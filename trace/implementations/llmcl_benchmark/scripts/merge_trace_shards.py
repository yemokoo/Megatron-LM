#!/usr/bin/env python
"""Merge collision-free TRACE sample shards and score the full task."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_eval import TRACE_SCORING_PROTOCOL, normalize_predictions, score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--with_sari", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = {}
    for shard_id in range(args.num_shards):
        suffix = f".shard{shard_id}-of-{args.num_shards}"
        path = os.path.join(
            args.input_dir, f"results-{args.task}{suffix}.json")
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        indices = payload.get("sample_indices")
        if indices is None:
            raise ValueError(f"missing sample_indices in {path}")
        columns = (
            indices,
            payload["prompts"],
            payload["results"],
            payload["labels"],
        )
        if len({len(column) for column in columns}) != 1:
            raise ValueError(f"length mismatch in {path}")
        for index, prompt, prediction, label in zip(*columns):
            if index in rows:
                raise ValueError(f"duplicate sample index {index}")
            rows[index] = (prompt, prediction, label)

    indices = sorted(rows)
    if indices != list(range(len(indices))):
        raise ValueError(
            f"shards do not cover a contiguous dataset: "
            f"first={indices[:3]} last={indices[-3:]} n={len(indices)}")
    prompts = [rows[index][0] for index in indices]
    raw_predictions = [rows[index][1] for index in indices]
    labels = [rows[index][2] for index in indices]
    predictions = normalize_predictions(args.task, raw_predictions)
    result = score(
        args.task, prompts, predictions, labels, args.with_sari)

    output = {
        "eval": result,
        "prompts": prompts,
        "results": raw_predictions,
        "results_scored": predictions,
        "labels": labels,
        "sample_indices": indices,
        "scoring_protocol": TRACE_SCORING_PROTOCOL,
        "merged_from_shards": args.num_shards,
    }
    output_path = os.path.join(
        args.input_dir, f"results-{args.task}.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False)

    summary_path = os.path.join(args.input_dir, "summary.json")
    summary = {}
    if os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
    summary[args.task] = result
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"[{args.task}] merged {len(indices)} samples -> {result}")
    print(f"wrote {output_path}")
    print(f"updated {summary_path}")


if __name__ == "__main__":
    main()
