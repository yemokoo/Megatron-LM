#!/usr/bin/env python3
"""Self-generated replay probes for the TRACE v3 LoRA-MoE checkpoint.

Question: with a strong pretrained backbone (Llama-3.1-8B-Instruct) plus 8
task-specific LoRA experts behind one shared top-1 router, what makes the model
generate a replay corpus that covers all 8 old tasks?

Three strategies, one script:
  --mode bos      just <|begin_of_text|>                     (does the router spread on its own?)
  --mode chat     the llama3 chat header the model always saw at document start
  --mode anchor   chat header + a short task-identifying prefix (--prefix-text)
  --mode force    chat header (or BOS) + router top-1 forced to --force-expert in every layer

Every mode records, per generated token, which expert each layer's router picked,
so "did all 8 experts fire" is answered directly rather than inferred from text.

Outputs under --out-dir:
  text.jsonl      one generated sample per line
  tokens.npz      int32 [N, T] generated ids (prompt stripped)
  routing.json    per-layer expert histograms + overall top-1 counts
  stats.json      config, timing, degenerate-repetition rates
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]                    # .../trace
IMPL = REPO / "implementations" / "llmcl_benchmark"
sys.path.insert(0, str(IMPL))

from model.Ours_LoRA_MoE_V3 import (                          # noqa: E402
    load_v3_checkpoint, shared_router_layers)
from transformers import AutoConfig, AutoTokenizer            # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_profile import chat_profile                         # noqa: E402

TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
         "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base-model", default="/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct")
    p.add_argument("--mode", choices=["bos", "chat", "anchor", "force", "bos_token"], required=True)
    p.add_argument("--last-bias-file", default="",
                   help="last_bias.pt from train_bos_token.py --method last_bias: adds the trained bias at the "
                        "last decoder layer (any mode)")
    p.add_argument("--bos-repeat", type=int, default=1,
                   help="bos mode: number of <|begin_of_text|> tokens (training documents carry 2)")
    p.add_argument("--bos-token-file", default="",
                   help="bos_token mode: bos_token.pt from scripts/bos_token/train_bos_token.py; the prompt is "
                        "that single token and its trained embedding is written into the input embedding row")
    p.add_argument("--prefix-text", default="", help="anchor mode: text appended after the chat header")
    p.add_argument("--prefix-histogram", default="",
                   help="anchor mode: JSON {suffix: weight}; each sequence draws its own suffix, "
                        "appended after --prefix-text (used to separate tasks that share an instruction)")
    p.add_argument("--anchor-jsonl", default="",
                   help="anchor mode: per-sequence anchors, one JSON object per line "
                        "{\"anchor\": text, \"anchor_ids\": [token ids] (optional; used verbatim "
                        "after the chat header when present), ...}.  Sequences are grouped by "
                        "prompt length so every batch is padding-free.  --num-seqs caps the "
                        "number of lines used (0 = all).  Overrides --prefix-text/--prefix-histogram.")
    p.add_argument("--per-seq-routing", action="store_true",
                   help="also save routing_per_seq.npz: int8 top1[N, T_gen, L] (the layer-L top-1 "
                        "expert of every generated token, -1 after EOS/pad) so per-sample routing "
                        "statistics can be computed; needs the routing probe on")
    p.add_argument("--force-expert", type=int, default=-1, help="force mode: expert index forced in every layer")
    p.add_argument("--force-from", choices=["bos", "chat"], default="chat",
                   help="force mode: what to feed as the prompt while routing is forced")
    p.add_argument("--num-seqs", type=int, default=256)
    p.add_argument("--no-routing-probe", action="store_true",
                   help="skip the per-layer top-1 capture; it syncs GPU->CPU on every layer at "
                        "every decode step and dominates wall time when routing stats are not needed")
    p.add_argument("--target-tokens", type=int, default=0,
                   help="if >0, keep generating until this many generated tokens exist "
                        "(matches the real task's token budget); --num-seqs then only sets "
                        "the batch planning granularity and is raised as needed")
    p.add_argument("--max-seqs", type=int, default=100000,
                   help="hard cap on sequences when --target-tokens is used")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--label", default="")
    return p.parse_args()


def install_router_probe(model, force_expert=None, capture=True, trace=None):
    """Wrap every layer's router: record top-1 picks, optionally force the pick.

    With top_k=1 and straight_through_topk the inference-time expert weight is
    softmax over a single logit = 1.0 regardless of the logit value, so forcing
    only has to replace the index -- the weight stays exactly what a natural
    top-1 route would have used.

    ``trace`` (optional) is a dict; when it holds a list under key ``li`` the
    per-call top-1 indices of layer ``li`` are appended to it as int8 arrays of
    shape [batch, tokens] (prefill: [B, L_prompt]; each decode step: [B, 1]).
    Callers that want per-sequence routing reset those lists between batches.
    The aggregate ``counts`` behave exactly as before.
    """
    layers = shared_router_layers(model)
    counts = {}          # layer_index -> np.array[num_experts]
    for li, layer in enumerate(layers):
        router = layer.shared_expert_router
        counts[li] = np.zeros(router.num_experts, dtype=np.int64)
        original = router.forward

        def wrapped(hidden_states, _router=router, _li=li, _orig=original):
            ctx = _orig(hidden_states)
            if force_expert is not None:
                ctx.expert_indices = torch.full_like(ctx.expert_indices, force_expert)
                ctx._route_cache.clear()
            if capture:
                idx = ctx.expert_indices[:, 0].detach().reshape(-1).cpu().numpy()
                np.add.at(counts[_li], idx, 1)
                if trace is not None and _li in trace:
                    trace[_li].append(idx.reshape(
                        hidden_states.shape[0], -1).astype(np.int8))
            return ctx

        router.forward = wrapped
    return counts, len(layers)


def trace_to_gen_top1(trace, num_layers, bsz, gen_len):
    """Turn per-call router traces into int8 [B, gen_len, L] over generated tokens.

    Generated token t (t >= 1) was routed at decode step t-1's forward; token 0
    was routed by the last prompt position of the prefill, so the generated
    tokens' *own* routing (what each generated token does when it is fed back)
    is prefill-tail excluded: decode step j routes generated token j (it is the
    input of that step).  The last generated token is never fed back, so its
    slot is -1.
    """
    out = np.full((bsz, gen_len, num_layers), -1, dtype=np.int8)
    for li in range(num_layers):
        calls = trace[li]
        if not calls:
            continue
        decode = [c for c in calls[1:] if c.shape[1] == 1]
        if decode:
            steps = np.concatenate(decode, axis=1)[:, :gen_len]
            # decode step j has generated token j as input -> routing of token j
            out[:, :steps.shape[1], li] = steps
    return out


def repetition_rate(seq, n=4):
    seen, rep, total = set(), 0, 0
    for i in range(len(seq) - n + 1):
        g = tuple(seq[i:i + n])
        rep += g in seen
        seen.add(g)
        total += 1
    return rep / max(total, 1)


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, use_fast=False, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    profile = chat_profile(tokenizer, AutoConfig.from_pretrained(
        args.base_model, local_files_only=True).model_type)
    print(f"[v3bos] chat profile {profile.name}", flush=True)

    print(f"[v3bos] loading {args.checkpoint}", flush=True)
    sys.path.insert(0, str(REPO / "scripts" / "residual"))
    from residual_expert import load_v3_any_checkpoint   # residual-from-task-0 aware
    model, meta = load_v3_any_checkpoint(
        args.checkpoint, tokenizer, args.base_model, device="cuda", dtype=torch.bfloat16)
    model.eval()
    num_experts = meta["num_experts"]
    print(f"[v3bos] experts={num_experts} top_k={meta['top_k']} mode={meta['routing_weight_mode']}", flush=True)

    bos_token_name = None
    if args.mode == "bos_token":
        if not args.bos_token_file:
            raise SystemExit("--mode bos_token needs --bos-token-file")
        bt = torch.load(args.bos_token_file, map_location="cpu", weights_only=False)
        bos_token_name = bt["token_name"]
        tid = tokenizer.convert_tokens_to_ids(bos_token_name)
        if tid != bt["token_id"]:
            raise SystemExit(f"token id mismatch: tokenizer {tid} vs file {bt['token_id']}")
        emb = model.get_input_embeddings().weight
        with torch.no_grad():
            emb[tid] = bt["embedding"].to(emb.dtype).to(emb.device)
        print(f"[v3bos] bos_token {bt.get('alias')}={bos_token_name} id={tid} step={bt.get('step')} "
              f"|e-init|={float((bt['embedding'] - bt['init_embedding']).norm()):.3f}", flush=True)

    if args.last_bias_file:
        sys.path.insert(0, str(REPO / "scripts" / "bos_token"))
        from train_bos_token import install_last_bias
        lb = torch.load(args.last_bias_file, map_location="cpu", weights_only=False)
        install_last_bias(model, lb["site"], lb["bias"].cuda())
        print(f"[v3bos] last_bias site={lb['site']} layer={lb['layer_index']} |b|={float(lb['bias'].norm()):.3f} "
              f"step={lb.get('step')}", flush=True)

    force = args.force_expert if args.mode == "force" else None
    if force is not None and not (0 <= force < num_experts):
        raise SystemExit(f"--force-expert must be in [0,{num_experts})")
    trace = None
    if args.per_seq_routing:
        if args.no_routing_probe:
            raise SystemExit("--per-seq-routing needs the routing probe (drop --no-routing-probe)")
        trace = {li: [] for li in range(len(shared_router_layers(model)))}
    counts, num_layers = install_router_probe(
        model, force, capture=not args.no_routing_probe, trace=trace)
    per_seq_top1 = []      # list of int8 [B, T_gen, L] per batch (only with --per-seq-routing)
    anchor_id_rows = []    # per sequence: the prompt ids after the chat header

    if args.anchor_jsonl:
        if args.mode != "anchor":
            raise SystemExit("--anchor-jsonl is only meaningful with --mode anchor")
        return run_anchor_jsonl(args, out, tokenizer, model, meta, counts, num_layers,
                                trace, t0=time.time())

    if args.mode == "bos" or (args.mode == "force" and args.force_from == "bos"):
        base_prompt = "<|begin_of_text|>" * args.bos_repeat
    elif args.mode == "bos_token":
        base_prompt = bos_token_name
    elif args.mode == "anchor":
        base_prompt = profile.header + args.prefix_text
    else:                                    # chat, or force-from-chat
        base_prompt = profile.header

    # One prompt, or a per-sequence anchor drawn from a histogram.  Sequences are
    # grouped by their anchor so every batch has a single prompt length (no padding).
    if args.prefix_histogram:
        hist = json.loads(Path(args.prefix_histogram).read_text())
        suffixes = list(hist.keys())
        weights = np.asarray([float(hist[k]) for k in suffixes], dtype=np.float64)
        weights /= weights.sum()
        rng = np.random.default_rng(args.seed)
        draw = rng.choice(len(suffixes), size=args.num_seqs, p=weights)
        plan = [(base_prompt + suffixes[i], int((draw == i).sum()),
                 args.prefix_text + suffixes[i]) for i in range(len(suffixes))]
        plan = [(p_, n, tail) for p_, n, tail in plan if n > 0]
        print(f"[v3bos] anchor histogram: {len(plan)} distinct, "
              f"{ {suffixes[i]: int((draw == i).sum()) for i in range(len(suffixes))} }", flush=True)
    else:
        plan = [(base_prompt, args.num_seqs,
                 args.prefix_text if args.mode == "anchor" else "")]
    print(f"[v3bos] base prompt: {base_prompt!r}", flush=True)

    eos_ids = list(profile.stop_token_ids)

    all_rows, t0, done = [], time.time(), 0
    prompt_len = None
    anchor_tails = []
    produced_tokens = 0

    def budget_met():
        if args.target_tokens <= 0:
            return False
        return produced_tokens >= args.target_tokens

    header_len = header_length(tokenizer, base_prompt)
    header_ids = tokenizer(base_prompt[:0] + (CHAT_HEADER if base_prompt.startswith(CHAT_HEADER) else base_prompt),
                           add_special_tokens=False).input_ids
    warned = []

    def record_batch(batch_prompt_ids, gen):
        # extra bookkeeping (anchor ids + optional per-sequence routing); does not
        # touch the legacy outputs
        if not warned and batch_prompt_ids[0, :header_len].tolist() != header_ids:
            print("[v3bos] WARNING: header/anchor token boundary merged; anchor_ids may be off by one", flush=True)
            warned.append(1)
        anchor_id_rows.extend(batch_prompt_ids[:, header_len:].tolist())
        if trace is not None:
            per_seq_top1.append(trace_to_gen_top1(trace, num_layers, gen.shape[0], gen.shape[1]))
            for li in trace:
                trace[li].clear()

    for prompt, want, anchor_tail in plan:
        prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        if prompt_len is None:
            prompt_len = prompt_ids.shape[1]
        made = 0
        while made < want:
            bsz = min(args.batch, want - made)
            batch_ids = prompt_ids.repeat(bsz, 1).to(model.device)
            with torch.no_grad():
                output = model.generate(
                    input_ids=batch_ids,
                    attention_mask=torch.ones_like(batch_ids),
                    do_sample=True, temperature=args.temperature, top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=eos_ids,
                    use_cache=True)
            gen = output[:, prompt_ids.shape[1]:].cpu().numpy().astype(np.int32)
            all_rows.append(gen)
            anchor_tails.extend([anchor_tail] * bsz)
            record_batch(prompt_ids.repeat(bsz, 1), gen)
            made += bsz
            done += bsz
            produced_tokens += int((gen != tokenizer.pad_token_id).sum())
            if args.target_tokens > 0:
                print(f"[v3bos] {done} seqs, {produced_tokens:,}/{args.target_tokens:,} tok, "
                      f"{time.time() - t0:.0f}s", flush=True)
            else:
                print(f"[v3bos] {done}/{args.num_seqs} seqs, {time.time() - t0:.0f}s", flush=True)

    wave = 0
    while args.target_tokens > 0 and not budget_met() and done < args.max_seqs:
        wave += 1
        print(f"[v3bos] token budget not met ({produced_tokens:,}/{args.target_tokens:,}); wave {wave}", flush=True)
        for prompt, want, anchor_tail in plan:
            if budget_met() or done >= args.max_seqs:
                break
            prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
            made = 0
            while made < want and not budget_met() and done < args.max_seqs:
                bsz = min(args.batch, want - made, args.max_seqs - done)
                batch_ids = prompt_ids.repeat(bsz, 1).to(model.device)
                with torch.no_grad():
                    output = model.generate(
                        input_ids=batch_ids,
                        attention_mask=torch.ones_like(batch_ids),
                        do_sample=True, temperature=args.temperature, top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=eos_ids,
                        use_cache=True)
                gen = output[:, prompt_ids.shape[1]:].cpu().numpy().astype(np.int32)
                all_rows.append(gen)
                anchor_tails.extend([anchor_tail] * bsz)
                record_batch(prompt_ids.repeat(bsz, 1), gen)
                made += bsz
                done += bsz
                produced_tokens += int((gen != tokenizer.pad_token_id).sum())
                print(f"[v3bos] {done} seqs, {produced_tokens:,}/{args.target_tokens:,} tok, "
                      f"{time.time() - t0:.0f}s", flush=True)

    finalize(args, out, tokenizer, model, counts, num_layers, force, all_rows, anchor_tails,
             anchor_id_rows, per_seq_top1, base_prompt, prompt_len, done, t0)


def header_length(tokenizer, base_prompt):
    """Number of prompt tokens that belong to the chat header / BOS, i.e. everything
    that is not the task anchor.  Tokenized jointly with the anchor to stay exact."""
    if base_prompt.startswith(CHAT_HEADER):
        return tokenizer(CHAT_HEADER, add_special_tokens=False, return_tensors="pt").input_ids.shape[1]
    return tokenizer(base_prompt, add_special_tokens=False, return_tensors="pt").input_ids.shape[1]


def run_anchor_jsonl(args, out, tokenizer, model, meta, counts, num_layers, trace, t0):
    """Per-sequence anchors from --anchor-jsonl (one sequence per line).

    Every line becomes exactly one sequence whose prompt is CHAT_HEADER + anchor
    (anchor_ids used verbatim when given, otherwise the anchor text is tokenized
    together with the header).  Sequences are grouped by prompt length and each
    group is generated in --batch sized, padding-free batches.  Output files are
    the same as the other modes; text.jsonl additionally carries "anchor_ids" and
    any extra keys of the input line (e.g. a source index).
    """
    force = None
    rows = [json.loads(l) for l in Path(args.anchor_jsonl).open() if l.strip()]
    if args.num_seqs > 0:
        rows = rows[:args.num_seqs]
    header_ids = tokenizer(CHAT_HEADER, add_special_tokens=False).input_ids
    seqs = []                                  # (prompt_ids list, row)
    for r in rows:
        if r.get("anchor_ids"):
            ids = header_ids + [int(t) for t in r["anchor_ids"]]
        else:
            ids = tokenizer(CHAT_HEADER + r.get("anchor", ""), add_special_tokens=False).input_ids
        seqs.append((ids, r))
    groups = {}
    for j, (ids, r) in enumerate(seqs):
        groups.setdefault(len(ids), []).append(j)
    print(f"[v3bos] anchor-jsonl: {len(seqs)} seqs, {len(groups)} prompt lengths "
          f"{sorted((L - len(header_ids), len(v)) for L, v in groups.items())}", flush=True)

    eos_ids = [tokenizer.eos_token_id]
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot >= 0:
        eos_ids.append(eot)

    order, all_rows, anchor_tails, anchor_id_rows, per_seq_top1, done = [], [], [], [], [], 0
    for L in sorted(groups):
        members = groups[L]
        for s0 in range(0, len(members), args.batch):
            chunk = members[s0:s0 + args.batch]
            batch_ids = torch.tensor([seqs[j][0] for j in chunk], dtype=torch.long, device=model.device)
            with torch.no_grad():
                output = model.generate(
                    input_ids=batch_ids, attention_mask=torch.ones_like(batch_ids),
                    do_sample=True, temperature=args.temperature, top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=eos_ids, use_cache=True)
            gen = output[:, L:].cpu().numpy().astype(np.int32)
            all_rows.append(gen)
            anchor_id_rows.extend(batch_ids[:, len(header_ids):].tolist())
            anchor_tails.extend([tokenizer.decode(seqs[j][0][len(header_ids):], skip_special_tokens=False)
                                 for j in chunk])
            order.extend(chunk)
            if trace is not None:
                per_seq_top1.append(trace_to_gen_top1(trace, num_layers, gen.shape[0], gen.shape[1]))
                for li in trace:
                    trace[li].clear()
            done += len(chunk)
            print(f"[v3bos] {done}/{len(seqs)} seqs (anchor len {L - len(header_ids)}), "
                  f"{time.time() - t0:.0f}s", flush=True)
    extra = [{k: v for k, v in seqs[j][1].items() if k not in ("anchor", "anchor_ids")} for j in order]
    finalize(args, out, tokenizer, model, counts, num_layers, force, all_rows, anchor_tails,
             anchor_id_rows, per_seq_top1, CHAT_HEADER + "<per-seq anchor>", None, done, t0,
             extra_fields=extra, source_order=order)


def finalize(args, out, tokenizer, model, counts, num_layers, force, all_rows, anchor_tails,
             anchor_id_rows, per_seq_top1, base_prompt, prompt_len, done, t0,
             extra_fields=None, source_order=None):
    num_experts = counts[0].shape[0]
    total_seqs = done
    width = max(r.shape[1] for r in all_rows)
    padded = np.full((total_seqs, width), tokenizer.pad_token_id, dtype=np.int32)
    row = 0
    for chunk in all_rows:
        padded[row:row + chunk.shape[0], :chunk.shape[1]] = chunk
        row += chunk.shape[0]
    if prompt_len is None:                     # per-seq anchors: variable prompt length
        prompt_len = -1
    np.savez_compressed(out / "tokens.npz", tokens=padded, prompt_len=prompt_len)
    if per_seq_top1:
        pw = max(a.shape[1] for a in per_seq_top1)
        top1 = np.full((total_seqs, pw, num_layers), -1, dtype=np.int8)
        row = 0
        for a in per_seq_top1:
            top1[row:row + a.shape[0], :a.shape[1]] = a
            row += a.shape[0]
        np.savez_compressed(out / "routing_per_seq.npz", top1=top1)

    texts = []
    with (out / "text.jsonl").open("w") as fh:
        for i, seq in enumerate(padded):
            body = [int(t) for t in seq if int(t) not in (tokenizer.pad_token_id,)]
            text = tokenizer.decode(body, skip_special_tokens=False)
            texts.append(text)
            rec = {"i": i, "anchor": anchor_tails[i] if i < len(anchor_tails) else "",
                   "text": text}
            if i < len(anchor_id_rows):
                rec["anchor_ids"] = [int(t) for t in anchor_id_rows[i]]
            if extra_fields is not None and i < len(extra_fields):
                rec.update(extra_fields[i])
            if source_order is not None and i < len(source_order):
                rec["src_line"] = int(source_order[i])
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    per_layer = {str(li): counts[li].tolist() for li in counts}
    total = np.sum([counts[li] for li in counts], axis=0)
    routing = {
        "num_layers": num_layers, "num_experts": num_experts,
        "forced_expert": force,
        "per_layer_top1_counts": per_layer,
        "total_top1_counts": total.tolist(),
        "total_top1_fraction": (total / max(total.sum(), 1)).tolist(),
        "experts_used": int((total > 0).sum()),
        "layers_with_all_experts": int(sum(
            1 for li in counts if (counts[li] > 0).sum() == num_experts)),
    }
    (out / "routing.json").write_text(json.dumps(routing, indent=1))

    flat = [int(t) for seq in padded for t in seq if int(t) != tokenizer.pad_token_id]
    stats = {
        "label": args.label or args.mode, "mode": args.mode, "checkpoint": args.checkpoint,
        "prompt": base_prompt, "prompt_len": prompt_len,
        "prefix_histogram": args.prefix_histogram, "prefix_text": args.prefix_text,
        "anchor_jsonl": args.anchor_jsonl, "per_seq_routing": bool(per_seq_top1),
        "force_expert": force, "num_seqs": total_seqs, "requested_seqs": args.num_seqs,
        "target_tokens": args.target_tokens,
        "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
        "top_p": args.top_p, "seed": args.seed,
        "tokens_generated": len(flat),
        "unique_token_ratio": len(set(flat)) / max(len(flat), 1),
        "repeat_4gram_rate": float(np.mean([
            repetition_rate([int(t) for t in seq if int(t) != tokenizer.pad_token_id])
            for seq in padded])),
        "wall_seconds": time.time() - t0,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print("[v3bos] routing:", json.dumps({k: routing[k] for k in (
        "experts_used", "layers_with_all_experts", "total_top1_fraction")}), flush=True)
    print("[v3bos] stats:", json.dumps({k: stats[k] for k in (
        "tokens_generated", "unique_token_ratio", "repeat_4gram_rate", "wall_seconds")}), flush=True)
    print(f"[v3bos] DONE {out}", flush=True)


if __name__ == "__main__":
    main()
