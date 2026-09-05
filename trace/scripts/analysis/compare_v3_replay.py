#!/usr/bin/env python3
"""Which self-generation strategy fills a TRACE replay buffer best?

Reference = the exact replay memory the v3 run used: for each of the 8 tasks,
the 500 train records named by fixed_replay_memory/task_*.json indices.

For each generated corpus we ask two different questions:
  content coverage -- per sequence, which task does the text look like
                      (unigram naive Bayes over the 8 reference sets), and how
                      evenly are the 8 tasks covered (fraction per task,
                      normalized entropy, smallest task share)
  routing coverage -- which experts the router actually picked while generating
                      (from routing.json), and how evenly

A perfect replay generator would sit near uniform on both.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
         "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]
BASE = "/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct"
TRACE = "/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace"
MEM = ("/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/v3_replay1to1/"
       "v3_new_replay1to1_st_top1/fixed_replay_memory")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gen", nargs="+", required=True, help="label=<dir> of a bos_sample_v3.py run")
    p.add_argument("--out", required=True)
    p.add_argument("--cache", default=None, help="npz cache of the reference unigram table")
    return p.parse_args()


def load_reference(tokenizer, cache_path):
    if cache_path and Path(cache_path).is_file():
        with np.load(cache_path) as z:
            return z["counts"], list(z["tasks"])
    vocab = len(tokenizer)
    counts = np.zeros((len(TASKS), vocab), dtype=np.float64)
    for ti, task in enumerate(TASKS):
        idx = json.loads((Path(MEM) / f"task_{ti}_{task}.json").read_text())["indices"]
        rows = json.loads((Path(TRACE) / task / "train.json").read_text())
        for i in idx:
            text = rows[i]["prompt"] + "\n" + rows[i]["answer"]
            ids = tokenizer(text, add_special_tokens=False).input_ids
            np.add.at(counts[ti], np.asarray(ids), 1)
        print(f"[ref] {task}: {len(idx)} records, {int(counts[ti].sum())} tokens", flush=True)
    if cache_path:
        np.savez_compressed(cache_path, counts=counts, tasks=np.array(TASKS))
    return counts, TASKS


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        BASE, use_fast=False, trust_remote_code=True, local_files_only=True)
    counts, tasks = load_reference(tokenizer, args.cache)
    # add-1 smoothed unigram log-probs per task
    probs = (counts + 1.0) / (counts + 1.0).sum(axis=1, keepdims=True)
    logp = np.log(probs)

    report = {}
    for item in args.gen:
        label, path = item.split("=", 1)
        d = Path(path)
        rows = [json.loads(l) for l in (d / "text.jsonl").open()]
        with np.load(d / "tokens.npz") as z:
            toks = z["tokens"]
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        assign, per_seq_len = [], []
        for seq in toks:
            ids = np.asarray([int(t) for t in seq if int(t) != pad], dtype=np.int64)
            if ids.size == 0:
                continue
            scores = logp[:, ids].sum(axis=1)          # [8] per-task log-likelihood
            assign.append(int(scores.argmax()))
            per_seq_len.append(int(ids.size))
        assign = np.asarray(assign)
        share = np.array([(assign == t).mean() for t in range(len(TASKS))])
        nz = share[share > 0]
        entropy = float(-(nz * np.log(nz)).sum() / np.log(len(TASKS)))
        routing = json.loads((d / "routing.json").read_text())
        rf = np.asarray(routing["total_top1_fraction"])
        rnz = rf[rf > 0]
        report[label] = {
            "n_seqs": int(assign.size),
            "mean_len": float(np.mean(per_seq_len)),
            "content_share": {TASKS[i]: round(float(share[i]), 4) for i in range(len(TASKS))},
            "content_tasks_covered": int((share > 0).sum()),
            "content_min_share": float(share.min()),
            "content_entropy_norm": entropy,
            "routing_share": {TASKS[i]: round(float(rf[i]), 4) for i in range(len(TASKS))},
            "routing_experts_used": routing["experts_used"],
            "routing_entropy_norm": float(-(rnz * np.log(rnz)).sum() / np.log(len(TASKS))),
            "forced_expert": routing.get("forced_expert"),
            "stats": json.loads((d / "stats.json").read_text()),
        }
        print(f"\n== {label}  n={report[label]['n_seqs']} len={report[label]['mean_len']:.0f}")
        print("   content:", {k: v for k, v in report[label]["content_share"].items() if v > 0})
        print(f"   covered={report[label]['content_tasks_covered']}/8 "
              f"min={report[label]['content_min_share']:.3f} H={entropy:.3f}")
        print("   routing:", {k: v for k, v in report[label]["routing_share"].items() if v >= 0.01})
        print(f"   experts_used={routing['experts_used']}/8 H={report[label]['routing_entropy_norm']:.3f}")

    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\n[OUT] {args.out}")


if __name__ == "__main__":
    main()
