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
from transformers import AutoTokenizer                        # noqa: E402

CHAT_HEADER = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a helpful assistant.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n")
TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
         "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--router-from", default="", help="checkpoint whose routers (task-k) replace the loaded ones")
    p.add_argument("--base-model", default="/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct")
    p.add_argument("--mode", choices=["bos", "chat", "anchor", "force"], required=True)
    p.add_argument("--prefix-text", default="", help="anchor mode: text appended after the chat header")
    p.add_argument("--prefix-histogram", default="",
                   help="anchor mode: JSON {suffix: weight}; each sequence draws its own suffix, "
                        "appended after --prefix-text (used to separate tasks that share an instruction)")
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


def install_router_probe(model, force_expert=None, capture=True):
    """Wrap every layer's router: record top-1 picks, optionally force the pick.

    With top_k=1 and straight_through_topk the inference-time expert weight is
    softmax over a single logit = 1.0 regardless of the logit value, so forcing
    only has to replace the index -- the weight stays exactly what a natural
    top-1 route would have used.
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
            return ctx

        router.forward = wrapped
    return counts, len(layers)


def repetition_rate(seq, n=4):
    seen, rep, total = set(), 0, 0
    for i in range(len(seq) - n + 1):
        g = tuple(seq[i:i + n])
        rep += g in seen
        seen.add(g)
        total += 1
    return rep / max(total, 1)




def apply_router_snapshot(model, router_ckpt):
    """Router-snapshot generation: overwrite the shared routers with the rows saved
    at task-k time and restrict routing to the experts that existed then, so the
    generator reproduces the task-k model (experts are frozen after their task)."""
    import torch as _t
    sd = _t.load(str(Path(router_ckpt) / "pytorch_model.bin"), map_location="cpu", weights_only=True)
    n_k = None
    for i, layer in enumerate(shared_router_layers(model)):
        w = sd[f"model.layers.{i}.shared_expert_router.router.weight"]
        n_k = int(w.shape[0])
        with _t.no_grad():
            layer.shared_expert_router.router.weight[:n_k].copy_(w.to(layer.shared_expert_router.router.weight.dtype))
        layer.shared_expert_router._active_expert_count = n_k
    print(f"[router-snapshot] routers <- {router_ckpt} ({n_k} experts active)", flush=True)
    return n_k


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

    print(f"[v3bos] loading {args.checkpoint}", flush=True)
    model, meta = load_v3_checkpoint(
        args.checkpoint, tokenizer, args.base_model, device="cuda", dtype=torch.bfloat16)
    if args.router_from:
        apply_router_snapshot(model, args.router_from)
    model.eval()
    num_experts = meta["num_experts"]
    print(f"[v3bos] experts={num_experts} top_k={meta['top_k']} mode={meta['routing_weight_mode']}", flush=True)

    force = args.force_expert if args.mode == "force" else None
    if force is not None and not (0 <= force < num_experts):
        raise SystemExit(f"--force-expert must be in [0,{num_experts})")
    counts, num_layers = install_router_probe(
        model, force, capture=not args.no_routing_probe)

    if args.mode == "bos" or (args.mode == "force" and args.force_from == "bos"):
        base_prompt = "<|begin_of_text|>"
    elif args.mode == "anchor":
        base_prompt = CHAT_HEADER + args.prefix_text
    else:                                    # chat, or force-from-chat
        base_prompt = CHAT_HEADER

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

    eos_ids = [tokenizer.eos_token_id]
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot >= 0:
        eos_ids.append(eot)

    all_rows, t0, done = [], time.time(), 0
    prompt_len = None
    anchor_tails = []
    produced_tokens = 0

    def budget_met():
        if args.target_tokens <= 0:
            return False
        return produced_tokens >= args.target_tokens

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
                made += bsz
                done += bsz
                produced_tokens += int((gen != tokenizer.pad_token_id).sum())
                print(f"[v3bos] {done} seqs, {produced_tokens:,}/{args.target_tokens:,} tok, "
                      f"{time.time() - t0:.0f}s", flush=True)

    total_seqs = done
    width = max(r.shape[1] for r in all_rows)
    padded = np.full((total_seqs, width), tokenizer.pad_token_id, dtype=np.int32)
    row = 0
    for chunk in all_rows:
        padded[row:row + chunk.shape[0], :chunk.shape[1]] = chunk
        row += chunk.shape[0]
    np.savez_compressed(out / "tokens.npz", tokens=padded, prompt_len=prompt_len)

    texts = []
    with (out / "text.jsonl").open("w") as fh:
        for i, seq in enumerate(padded):
            body = [int(t) for t in seq if int(t) not in (tokenizer.pad_token_id,)]
            text = tokenizer.decode(body, skip_special_tokens=False)
            texts.append(text)
            fh.write(json.dumps(
                {"i": i, "anchor": anchor_tails[i] if i < len(anchor_tails) else "",
                 "text": text}, ensure_ascii=False) + "\n")

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
