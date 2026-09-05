#!/usr/bin/env python3
"""Can generated NumGLUE text be told apart as cm vs ds?

The 8-way naive Bayes in compare_v3_replay.py is too coarse for this pair (they
share an instruction and both are math word problems), so this builds a
dedicated 2-way classifier from the two tasks' replay memories and first
reports its accuracy on held-out REAL data -- that number is the ceiling any
generated corpus can be scored against.
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
PAIR = [("NumGLUE-cm", 5), ("NumGLUE-ds", 6)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gen", nargs="*", default=[], help="label=<dir>")
    p.add_argument("--out", required=True)
    return p.parse_args()


def question_body(prompt):
    return prompt.split("Question:\n", 1)[-1].replace("\nAnswer:", "").strip()


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=False, trust_remote_code=True, local_files_only=True)
    vocab = len(tok)

    train_texts, held_out = {}, {}
    for task, ti in PAIR:
        idx = set(json.loads((Path(MEM) / f"task_{ti}_{task}.json").read_text())["indices"])
        rows = json.loads((Path(TRACE) / task / "train.json").read_text())
        train_texts[task] = [question_body(rows[i]["prompt"]) for i in sorted(idx)]
        rest = [i for i in range(len(rows)) if i not in idx][:500]
        held_out[task] = [question_body(rows[i]["prompt"]) for i in rest]

    counts = np.zeros((2, vocab), dtype=np.float64)
    for k, (task, _) in enumerate(PAIR):
        for text in train_texts[task]:
            np.add.at(counts[k], np.asarray(tok(text, add_special_tokens=False).input_ids), 1)
    logp = np.log((counts + 1.0) / (counts + 1.0).sum(axis=1, keepdims=True))

    def classify(texts_or_ids, are_ids=False):
        out = []
        for item in texts_or_ids:
            ids = item if are_ids else np.asarray(tok(item, add_special_tokens=False).input_ids)
            if len(ids) == 0:
                continue
            out.append(int(logp[:, ids].sum(axis=1).argmax()))
        return np.asarray(out)

    report = {"classifier_ceiling": {}}
    for k, (task, _) in enumerate(PAIR):
        pred = classify(held_out[task])
        acc = float((pred == k).mean())
        report["classifier_ceiling"][task] = {"held_out_n": int(pred.size), "accuracy": acc}
        print(f"[ceiling] real held-out {task}: {acc*100:.1f}% correct (n={pred.size})")

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for item in args.gen:
        label, path = item.split("=", 1)
        with np.load(Path(path) / "tokens.npz") as z:
            toks = z["tokens"]
        ids_list = [np.asarray([int(t) for t in seq if int(t) != pad]) for seq in toks]
        ids_list = [x for x in ids_list if x.size > 0]
        pred = classify(ids_list, are_ids=True)
        share_cm = float((pred == 0).mean())
        report[label] = {"n": int(pred.size), "cm_share": share_cm, "ds_share": 1.0 - share_cm}
        print(f"[gen] {label:<18} cm {share_cm*100:5.1f}%   ds {(1-share_cm)*100:5.1f}%   (n={pred.size})")

    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"[OUT] {args.out}")


if __name__ == "__main__":
    main()
