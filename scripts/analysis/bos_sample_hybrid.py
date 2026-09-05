#!/usr/bin/env python3
"""Unconditional (BoS-only) sampling from a Megatron continual-learning checkpoint.

Feeds the model a single <|endoftext|> token (pythia's BOS == EOS == 0, and the
training streams are `--append-eod`, so "after EOD" is exactly "document start")
and samples `--bos-seq-len` tokens per sequence with nucleus sampling.  No KV
cache: the model is 9 layers x 1024 wide, so recomputing the full prefix each
step is cheap and avoids trusting custom layers with inference_params.

Outputs (rank 0), under --bos-out-dir:
  tokens.npz            int32 [N, 1 + seq_len], column 0 is the BOS
  text.jsonl            detokenized samples
  gen_text_document.bin/.idx   Megatron indexed dataset, one document per sample,
                        so the existing probe / router-usage / hidden-dump tooling
                        can read the generated set like any other corpus
  stats.json            EOD rate, repetition rate, unique-token ratio

Model construction and checkpoint loading follow plot_wiki_router_softmax_importance.py:
the usual MODEL_ARGS go on the command line, initialize_megatron() parses them.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
MEGATRON_ROOT = REPO_ROOT / "Megatron-LM"
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))

from megatron.core import InferenceParams  # noqa: E402
from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder  # noqa: E402
from megatron.core.enums import ModelType  # noqa: E402
from megatron.inference.text_generation.sampling import sample  # noqa: E402
from megatron.training import get_args, get_tokenizer, print_rank_0  # noqa: E402
from megatron.training.checkpointing import load_checkpoint  # noqa: E402
from megatron.training.initialize import initialize_megatron  # noqa: E402
from megatron.training.training import get_model  # noqa: E402
from megatron.training.utils import get_ltor_masks_and_position_ids, unwrap_model  # noqa: E402
from pretrain_gpt import model_provider  # noqa: E402


def add_bos_args(parser):
    group = parser.add_argument_group("bos-sample")
    group.add_argument("--bos-out-dir", required=True)
    group.add_argument("--bos-num-seqs", type=int, default=2048)
    group.add_argument("--bos-seq-len", type=int, default=512)
    group.add_argument("--bos-batch", type=int, default=64)
    group.add_argument("--bos-temperature", type=float, default=1.0)
    group.add_argument("--bos-top-p", type=float, default=0.95)
    group.add_argument("--bos-top-k", type=int, default=0)
    group.add_argument("--bos-seed", type=int, default=0)
    group.add_argument("--bos-label", default="")
    group.add_argument("--bos-prefix-text", default="", help="optional text placed after the BOS (minimal conditioning); still no stored data")
    group.add_argument("--bos-anchor-json", default="", help="json with {\"anchor_probs\": {token_id: prob}}: every sequence gets ONE anchor token after the BOS, drawn from this histogram (per-task document-first-token distribution)")
    group.add_argument("--bos-route-allow", default="", help="expert range lo:hi allowed in top-k (e.g. 0:8 = wiki experts only, 8:16 = code experts only); empty = all")
    group.add_argument("--bos-route-groups", default="", help="per-group top-k, e.g. 0:8=2,8:16=2 -> best 2 wiki experts + best 2 code experts per token (sum must equal top-k)")
    group.add_argument("--bos-kv-cache", action="store_true", help="incremental decoding with the stock InferenceParams KV cache (prefill once, then one token per step)")
    group.add_argument("--bos-kv-check", type=int, default=0, help=">0: before sampling, decode this many greedy steps with and without the KV cache on 8 sequences and report the logit/argmax agreement")
    group.add_argument("--bos-route-balance-eta", type=float, default=0.0, help=">0: load-balance all experts during generation — per-layer expert bias updated every step by -eta*sign(load-mean) (aux-loss-free balancing); selection uses logits+bias, gates are the natural probs renormalised over the selected experts")
    group.add_argument("--bos-route-gumbel-tau", type=float, default=0.0, help=">0: sample the top-k experts with Gumbel-top-k at this router temperature instead of argmax")
    return parser


@torch.no_grad()
def generate_batch(model, bsz, seq_len, eod, temperature, top_p, top_k, vocab_size, device, prefix_ids=()):
    start = [eod] + list(prefix_ids)
    tokens = torch.tensor([start] * bsz, dtype=torch.long, device=device)
    for _ in range(seq_len):
        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            tokens, eod, False, False, False)
        logits = model(tokens, position_ids, attention_mask, labels=None, runtime_gather_output=True)
        last = logits[:, -1, :].float()
        nxt = sample(last, top_k=top_k, top_p=top_p, temperature=temperature, vocab_size=vocab_size)
        tokens = torch.cat([tokens, nxt.view(bsz, 1).to(tokens.dtype)], dim=1)
    return tokens


def install_route_forcing(core, allow, tau, groups=None, balance_eta=0.0):
    """Replace every shared router's routing() so top-k is restricted to `allow` (bool [E]) and/or
    sampled with Gumbel-top-k (tau>0).  Gates keep the model's own pre-softmax semantics (softmax over
    the allowed experts, values at the selected ones, no renormalisation), so the restricted case is
    exactly 'natural routing within the subset'.  Also counts, per layer, the routing of the last
    position of every forward call (= the token being generated)."""
    from megatron.training.training import _collect_current_shared_routers
    routers = _collect_current_shared_routers([core])
    usage = {}; bias = {}
    for layer, router in routers.items():
        E = int(router.config.num_moe_experts); k = int(router.topk)
        mask = None
        if allow is not None:
            mask = torch.zeros(E, dtype=torch.bool, device=torch.cuda.current_device()); mask[allow] = True
            assert mask.sum() >= k, "fewer allowed experts than top-k"
        usage[layer] = torch.zeros(E, dtype=torch.long, device=torch.cuda.current_device())
        bias[layer] = torch.zeros(E, dtype=torch.float32, device=torch.cuda.current_device())
        if groups:
            assert sum(g[2] for g in groups) == k, "group top-k sizes must sum to top-k"
        def routing(logits, _E=E, _k=k, _mask=mask, _layer=layer, _groups=groups):
            seq, bsz = logits.shape[:2]
            lg = logits.view(-1, _E).float()
            if _mask is not None:
                lg = lg.masked_fill(~_mask, float("-inf"))
            probs = torch.softmax(lg, dim=-1)
            sel = lg if tau <= 0 else lg / tau - torch.log(-torch.log(torch.rand_like(lg).clamp_min(1e-20)))
            if balance_eta > 0:
                sel = sel + bias[_layer]
            if _groups:
                idx = torch.cat([lo + torch.topk(sel[:, lo:hi], k=kg, dim=-1).indices for lo, hi, kg in _groups], dim=-1)
            else:
                idx = torch.topk(sel, k=_k, dim=-1).indices
            g = probs.gather(1, idx)
            if balance_eta > 0:   # forced experts must get real weight: renormalise over the selected set, keep natural top-k mass
                g = torch.softmax(lg.gather(1, idx), dim=-1) * g.sum(-1, keepdim=True)
            gates = torch.zeros_like(probs).scatter(1, idx, g)
            rmap = torch.zeros_like(probs, dtype=torch.bool).scatter(1, idx, True)
            load = rmap[-bsz:].sum(0)
            usage[_layer] += load
            if balance_eta > 0:
                bias[_layer] -= balance_eta * torch.sign(load.float() - load.float().mean())
            return gates.type_as(logits), rmap
        router.routing = routing
    install_route_forcing.bias = bias
    return usage


@torch.no_grad()
def generate_batch_cached(model, bsz, seq_len, eod, temperature, top_p, top_k, vocab_size, device, prefix_ids=(),
                          return_logits=False, anchors=None):
    """Same sampling as generate_batch, but with the stock KV cache: the [BOS]+prefix is prefilled with a
    causal mask, then every step feeds only the new token (attention_mask=None, positions continue) and
    bumps inference_params.sequence_len_offset, exactly like megatron/inference/text_generation."""
    start = [eod] + list(prefix_ids); L = len(start); total = L + seq_len
    ip = InferenceParams(bsz, total)
    tokens = torch.full((bsz, total), eod, dtype=torch.long, device=device)
    tokens[:, :L] = torch.tensor(start, dtype=torch.long, device=device)
    if anchors is not None:                       # per-sequence single-token anchor right after the BOS
        assert not prefix_ids, "use either a fixed prefix or per-sequence anchors"
        tokens = torch.cat([tokens[:, :1], anchors.view(bsz, 1).to(tokens), tokens[:, 1:]], dim=1); L += 1; total += 1
        ip = InferenceParams(bsz, total)
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(tokens, eod, False, False, False)
    prev, step_logits = 0, []
    for cur in range(L, total):
        t2, p2 = tokens[:, prev:cur], position_ids[:, prev:cur]
        am = attention_mask[..., prev:cur, :cur] if prev == 0 else None
        logits = model(t2, p2, am, labels=None, runtime_gather_output=True, inference_params=ip)
        ip.sequence_len_offset += t2.size(1)
        last = logits[:, -1, :].float()
        if return_logits:
            step_logits.append(last[:, :vocab_size].clone())
        tokens[:, cur] = sample(last, top_k=top_k, top_p=top_p, temperature=temperature, vocab_size=vocab_size).view(bsz)
        prev = cur
    return (tokens, step_logits) if return_logits else tokens


@torch.no_grad()
def kv_cache_check(model, eod, vocab_size, device, prefix_ids, steps, bsz=8):
    """Greedy-decode `steps` tokens with the cache, then recompute every step's logits without the cache on
    the same tokens.  Reports max |dlogit| and argmax agreement; the two paths must agree up to bf16 noise."""
    toks, cached = generate_batch_cached(model, bsz, steps, eod, 1.0, 0.0, 1, vocab_size, device, prefix_ids, return_logits=True)
    L = 1 + len(prefix_ids); maxdiff, agree, n = 0.0, 0, 0
    for i, cur in enumerate(range(L, L + steps)):
        ctx = toks[:, :cur]
        am, _, pid = get_ltor_masks_and_position_ids(ctx, eod, False, False, False)
        full = model(ctx, pid, am, labels=None, runtime_gather_output=True)[:, -1, :vocab_size].float()
        maxdiff = max(maxdiff, float((full - cached[i]).abs().max()))
        agree += int((full.argmax(-1) == cached[i].argmax(-1)).sum()); n += bsz
    print_rank_0(f"[bos] kv-cache check: {steps} greedy steps x {bsz} seqs: max|dlogit|={maxdiff:.4f} argmax agreement={agree}/{n}")
    return agree == n


def repetition_stats(seqs: np.ndarray, n: int = 4):
    """Fraction of n-grams that repeat an earlier n-gram inside the same sequence."""
    rep, total = 0, 0
    for s in seqs:
        seen = set()
        for i in range(len(s) - n + 1):
            g = tuple(s[i:i + n].tolist())
            if g in seen:
                rep += 1
            seen.add(g)
            total += 1
    return rep / max(total, 1)


def main():
    initialize_megatron(extra_args_provider=add_bos_args,
                        args_defaults={"tokenizer_type": "GPT2BPETokenizer"})
    args = get_args()
    torch.manual_seed(args.bos_seed)
    tok = get_tokenizer()
    eod = int(tok.eod)
    vocab = int(getattr(tok, "vocab_size", 50277))

    model = get_model(model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)
    iteration, _ = load_checkpoint(model, None, None)
    core = unwrap_model(model)[0]
    core.eval()
    device = torch.cuda.current_device()
    print_rank_0(f"[bos] loaded iteration {iteration} from {args.load}; eod={eod} vocab={vocab}")
    print_rank_0(f"[bos] sampling N={args.bos_num_seqs} T={args.bos_seq_len} batch={args.bos_batch} "
                 f"temp={args.bos_temperature} top_p={args.bos_top_p} top_k={args.bos_top_k} seed={args.bos_seed}")

    out = Path(args.bos_out_dir)
    rank0 = torch.distributed.get_rank() == 0
    if rank0:
        out.mkdir(parents=True, exist_ok=True)

    allow = None
    if args.bos_route_allow:
        lo, hi = (int(x) for x in args.bos_route_allow.split(":")); allow = list(range(lo, hi))
    groups = None
    if args.bos_route_groups:
        groups = []
        for part in args.bos_route_groups.split(","):
            rng_, kg = part.split("="); lo, hi = (int(x) for x in rng_.split(":")); groups.append((lo, hi, int(kg)))
    forced_usage = None
    if allow is not None or groups or args.bos_route_gumbel_tau > 0 or args.bos_route_balance_eta > 0:
        forced_usage = install_route_forcing(core, allow, args.bos_route_gumbel_tau, groups, args.bos_route_balance_eta)
        print_rank_0(f"[bos] route forcing: allow={args.bos_route_allow or 'all'} groups={args.bos_route_groups or '-'} gumbel_tau={args.bos_route_gumbel_tau} balance_eta={args.bos_route_balance_eta} layers={sorted(forced_usage)}")
    prefix_ids = tok.tokenize(args.bos_prefix_text) if args.bos_prefix_text else []
    anchor_ids, anchor_p = None, None
    if args.bos_anchor_json:
        assert args.bos_kv_cache, "--bos-anchor-json needs --bos-kv-cache"
        hist = json.loads(Path(args.bos_anchor_json).read_text())["anchor_probs"]
        anchor_ids = torch.tensor([int(k) for k in hist], dtype=torch.long, device=device)
        anchor_p = torch.tensor([float(v) for v in hist.values()], dtype=torch.float32, device=device); anchor_p /= anchor_p.sum()
        print_rank_0(f"[bos] per-sequence anchors from {args.bos_anchor_json}: {len(hist)} tokens, top: "
                     + ", ".join(f"{tok.detokenize([int(k)])!r}:{v:.2f}" for k, v in list(hist.items())[:6]))
    if prefix_ids:
        print_rank_0(f"[bos] prefix {args.bos_prefix_text!r} -> {prefix_ids}")
    if args.bos_kv_check > 0:
        ok = kv_cache_check(core, eod, vocab, device, prefix_ids, args.bos_kv_check)
        if not ok:
            raise SystemExit("[bos] KV-cache check FAILED (argmax disagreement); refusing to sample with the cache")
    all_tokens = []
    t0 = time.time()
    done = 0
    while done < args.bos_num_seqs:
        bsz = min(args.bos_batch, args.bos_num_seqs - done)
        if anchor_ids is not None:
            anc = anchor_ids[torch.multinomial(anchor_p, bsz, replacement=True)]
            toks = generate_batch_cached(core, bsz, args.bos_seq_len, eod, args.bos_temperature,
                                         args.bos_top_p, args.bos_top_k, vocab, device, anchors=anc)
        else:
            gen_fn = generate_batch_cached if args.bos_kv_cache else generate_batch
            toks = gen_fn(core, bsz, args.bos_seq_len, eod, args.bos_temperature,
                          args.bos_top_p, args.bos_top_k, vocab, device, prefix_ids)
        all_tokens.append(toks.cpu().numpy().astype(np.int32))
        done += bsz
        print_rank_0(f"[bos] {done}/{args.bos_num_seqs} sequences, {time.time() - t0:.0f}s")

    if not rank0:
        return
    tokens = np.concatenate(all_tokens, axis=0)              # [N, 1 + T]
    gen = tokens[:, 1:]                                      # drop the BOS column (prefix tokens, if any, stay)
    np.savez_compressed(out / "tokens.npz", tokens=tokens)

    with (out / "text.jsonl").open("w") as f:
        for i, s in enumerate(gen):
            f.write(json.dumps({"i": i, "text": tok.detokenize(s.tolist())}, ensure_ascii=False) + "\n")

    builder = IndexedDatasetBuilder(str(out / "gen_text_document.bin"), dtype=np.int32)
    for s in gen:
        builder.add_item(torch.from_numpy(s.astype(np.int32)))
        builder.end_document()
    builder.finalize(str(out / "gen_text_document.idx"))

    flat = gen.reshape(-1)
    stats = {
        "label": args.bos_label,
        "load": args.load,
        "checkpoint_iteration": int(iteration),
        "num_seqs": int(gen.shape[0]),
        "seq_len": int(gen.shape[1]),
        "temperature": args.bos_temperature,
        "top_p": args.bos_top_p,
        "top_k": args.bos_top_k,
        "seed": args.bos_seed,
        "prefix_text": args.bos_prefix_text, "anchor_json": args.bos_anchor_json,
        "route_allow": args.bos_route_allow, "route_groups": args.bos_route_groups, "route_gumbel_tau": args.bos_route_gumbel_tau,
        "route_balance_eta": args.bos_route_balance_eta, "kv_cache": bool(args.bos_kv_cache),
        "final_bias": {str(l): b.tolist() for l, b in install_route_forcing.bias.items()} if forced_usage and args.bos_route_balance_eta > 0 else None,
        "forced_usage": {str(l): (u / u.sum().clamp_min(1)).tolist() for l, u in forced_usage.items()} if forced_usage else None,
        "eod_token_rate": float((flat == eod).mean()),
        "seqs_with_eod": float((gen == eod).any(axis=1).mean()),
        "unique_token_ratio": float(len(np.unique(flat)) / vocab),
        "repeat_4gram_rate": repetition_stats(gen, 4),
        "repeat_8gram_rate": repetition_stats(gen, 8),
        "wall_seconds": time.time() - t0,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print_rank_0("[bos] " + json.dumps(stats))


if __name__ == "__main__":
    main()
