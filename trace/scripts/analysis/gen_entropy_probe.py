#!/usr/bin/env python3
"""Generate N anchored samples from a V3 checkpoint and record the model's
next-token entropy at every generation position.

Entropy is computed from the RAW logits (temperature 1, no top-p) at each step,
i.e. the model's own predictive distribution, while the tokens themselves are
sampled with the production settings (T, top-p).  Output:

  entropy.npz   entropy [N, T] float32 (nan after the sequence's EOS),
                tokens [N, T] int32, active [N, T] bool
  text.jsonl    decoded samples
  meta.json     config

Per-position averages (mean over samples still active at position t) are
left to the analysis script so different runs can be compared consistently.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
IMPL = REPO / "implementations" / "llmcl_benchmark"
sys.path.insert(0, str(IMPL))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from model.Ours_LoRA_MoE_V3 import load_v3_checkpoint      # noqa: E402
from transformers import AutoTokenizer                    # noqa: E402
from bos_sample_v3 import CHAT_HEADER                     # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base-model", default="/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct")
    p.add_argument("--prefix-text", default="", help="anchor appended after the chat header")
    p.add_argument("--num-seqs", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--label", default="")
    return p.parse_args()


@torch.no_grad()
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

    print(f"[entropy] loading {args.checkpoint}", flush=True)
    model, meta = load_v3_checkpoint(
        args.checkpoint, tokenizer, args.base_model, device="cuda", dtype=torch.bfloat16)
    model.eval()

    prompt = CHAT_HEADER + args.prefix_text
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
    L = prompt_ids.shape[1]
    eos_ids = [tokenizer.eos_token_id]
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot >= 0:
        eos_ids.append(eot)
    eos_set = set(eos_ids)

    T = args.max_new_tokens
    ent_all = np.full((args.num_seqs, T), np.nan, dtype=np.float32)
    tok_all = np.full((args.num_seqs, T), -1, dtype=np.int32)
    act_all = np.zeros((args.num_seqs, T), dtype=bool)
    texts = []
    done = 0
    while done < args.num_seqs:
        b = min(args.batch, args.num_seqs - done)
        batch_ids = prompt_ids.expand(b, -1).contiguous()
        outp = model.generate(
            input_ids=batch_ids, attention_mask=torch.ones_like(batch_ids),
            do_sample=True, temperature=args.temperature, top_p=args.top_p,
            max_new_tokens=T, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids, use_cache=True,
            output_logits=True, return_dict_in_generate=True)
        seqs = outp.sequences[:, L:]                       # [b, t_gen]
        t_gen = seqs.shape[1]
        # outp.logits: tuple of t_gen tensors [b, V] (raw, pre-warp)
        for t, lg in enumerate(outp.logits):
            logp = torch.log_softmax(lg.float(), dim=-1)
            H = -(logp.exp() * logp).sum(-1)               # [b]
            ent_all[done:done + b, t] = H.cpu().numpy()
        gen = seqs.cpu().numpy()
        tok_all[done:done + b, :t_gen] = gen
        # active[i, t] = position t was generated before (and including) EOS
        for i in range(b):
            row = gen[i]
            stop = t_gen
            for t in range(t_gen):
                if int(row[t]) in eos_set:
                    stop = t + 1
                    break
            act_all[done + i, :stop] = True
            ent_all[done + i, stop:] = np.nan
            texts.append(tokenizer.decode(row[:stop], skip_special_tokens=False))
        done += b
        print(f"[entropy] {done}/{args.num_seqs}", flush=True)

    np.savez_compressed(out / "entropy.npz", entropy=ent_all, tokens=tok_all, active=act_all)
    with open(out / "text.jsonl", "w") as f:
        for i, tx in enumerate(texts):
            f.write(json.dumps({"i": i, "text": tx}, ensure_ascii=False) + "\n")
    (out / "meta.json").write_text(json.dumps({
        "label": args.label, "checkpoint": args.checkpoint, "prefix_text": args.prefix_text,
        "num_seqs": args.num_seqs, "max_new_tokens": T, "temperature": args.temperature,
        "top_p": args.top_p, "seed": args.seed, "num_experts": meta.get("num_experts"),
        "entropy_def": "H_t = -sum p log p over vocab, p = softmax(raw logits at step t), nats",
        "mean_active_len": float(act_all.sum(1).mean()),
    }, indent=1))
    print(f"[entropy] DONE {out} mean_len={act_all.sum(1).mean():.1f} "
          f"H[0]={np.nanmean(ent_all[:,0]):.3f} H[10]={np.nanmean(ent_all[:,10]):.3f}", flush=True)


if __name__ == "__main__":
    main()
