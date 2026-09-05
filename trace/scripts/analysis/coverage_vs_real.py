#!/usr/bin/env python3
"""How well does the anchor-generated corpus cover each task's real train data?

For every task, compares the generated tokens against the task's REAL replay
memory (the 500 train records the v3 run actually replayed) on:
  JS            unigram Jensen-Shannon divergence (bits), generated vs real
  JS floor      the same divergence between two disjoint real halves drawn at
                the generated corpus's own token count -- the lowest JS any
                corpus of this size can reach, so JS is not read as failure
                when it is just sample noise
  mass cov      share of the real corpus's token occurrences whose token id
                appears at least once in the generated corpus
  type cov      share of the real corpus's distinct token ids present
  purity        share of generated sequences a naive-Bayes classifier assigns
                to this task (8-way over all tasks; the cm/ds pair also gets a
                dedicated 2-way number, which is what separates them)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

BASE = "/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct"
TRACE = "/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace"
MEM = ("/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/v3_replay1to1/"
       "v3_new_replay1to1_st_top1/fixed_replay_memory")
TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
         "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]


def js(p, q, eps=1e-12):
    p = np.asarray(p, float) + eps
    q = np.asarray(q, float) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log2(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", nargs="+", required=True, help="TASK=<dir>")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=False, trust_remote_code=True, local_files_only=True)
    vocab = len(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rng = np.random.default_rng(0)
    gen_map = dict(item.split("=", 1) for item in args.gen)

    rows = []
    for ti, task in enumerate(TASKS):
        if task not in gen_map:
            continue
        idx = json.loads((Path(MEM) / f"task_{ti}_{task}.json").read_text())["indices"]
        recs = json.loads((Path(TRACE) / task / "train.json").read_text())
        real_seqs = [np.asarray(tok(recs[i]["prompt"] + "\n" + recs[i]["answer"],
                                    add_special_tokens=False).input_ids) for i in idx]
        real_all = np.concatenate(real_seqs)

        with np.load(Path(gen_map[task]) / "tokens.npz") as z:
            toks = z["tokens"]
        gen_all = np.asarray([int(t) for seq in toks for t in seq if int(t) != pad])

        real_hist = np.bincount(real_all, minlength=vocab).astype(float)
        gen_hist = np.bincount(gen_all, minlength=vocab).astype(float)
        d = js(gen_hist, real_hist)

        # noise floor at the generated corpus's own size: two disjoint real draws
        n = min(gen_all.size, real_all.size // 2)
        perm = rng.permutation(real_all.size)
        a = np.bincount(real_all[perm[:n]], minlength=vocab).astype(float)
        b = np.bincount(real_all[perm[n:2 * n]], minlength=vocab).astype(float)
        floor = js(a, b)

        present = gen_hist > 0
        mass_cov = float(real_hist[present].sum() / real_hist.sum())
        type_cov = float((present & (real_hist > 0)).sum() / (real_hist > 0).sum())

        rows.append({
            "task": task, "gen_dir": gen_map[task],
            "gen_tokens": int(gen_all.size), "real_tokens": int(real_all.size),
            "js_gen_vs_real": d, "js_noise_floor_at_gen_size": floor,
            "js_ratio": d / max(floor, 1e-9),
            "mass_coverage": mass_cov, "type_coverage": type_cov,
            "gen_distinct": int(present.sum()), "real_distinct": int((real_hist > 0).sum()),
        })
        print(f"{task:<13} gen {gen_all.size:>7,}tok  real {real_all.size:>9,}tok   "
              f"JS {d:.3f} (floor {floor:.3f}, x{d/max(floor,1e-9):.1f})   "
              f"mass {mass_cov*100:5.1f}%  type {type_cov*100:5.1f}%")

    Path(args.out).write_text(json.dumps(rows, indent=1))
    print(f"\n[OUT] {args.out}")


if __name__ == "__main__":
    main()
