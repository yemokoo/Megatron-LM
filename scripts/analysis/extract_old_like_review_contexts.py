#!/usr/bin/env python3
"""Extract CPU-only context samples for manual old-like score review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--code-subsample", required=True, type=Path)
    p.add_argument("--subsample-metadata", required=True, type=Path)
    p.add_argument("--cut-decisions", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--per-group", type=int, default=100)
    p.add_argument("--context-radius", type=int, default=8)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = None
    if args.tokenizer is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    with np.load(args.code_subsample, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in ("sample_ids", "positions", "token_ids", "shard_ordinals", "cosine", "reference_rms")}
    metadata = json.loads(args.subsample_metadata.read_text())
    manifest = {int(r["ordinal"]): r for r in metadata["code_shard_manifest"]}
    score = arrays["cosine"].mean(axis=1, dtype=np.float64)
    q99, q90 = np.quantile(score, [.99, .90])
    # Deterministic ordering inside tied bins.
    key = arrays["sample_ids"].astype(np.uint64) * np.uint64(512) + arrays["positions"].astype(np.uint64)
    tie = key ^ np.uint64(0x9E3779B97F4A7C15)
    groups = {
        "raw_cosine_mean_top_0_to_1_percent": np.flatnonzero(score >= q99),
        "raw_cosine_mean_top_5_to_10_percent": np.flatnonzero((score >= q90) & (score < np.quantile(score, .95))),
    }
    for layer_index, layer in enumerate(range(2, 10)):
        lo, hi = np.quantile(arrays["reference_rms"][:, layer_index], [.001, .01])
        groups[f"layer_{layer}_reference_rms_low_tail_q0p1_to_q1"] = np.flatnonzero(
            (arrays["reference_rms"][:, layer_index] >= lo) & (arrays["reference_rms"][:, layer_index] < hi)
        )

    selected = []
    for group, candidates in groups.items():
        chosen = candidates[np.argsort(tie[candidates], kind="stable")[:args.per_group]]
        selected.extend((group, int(i)) for i in chosen)

    cache: dict[int, np.ndarray] = {}
    output = args.output_dir / "manual_review_context_samples.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for group, i in selected:
            ordinal = int(arrays["shard_ordinals"][i])
            row = manifest[ordinal]
            if ordinal not in cache:
                with np.load(row["path"], allow_pickle=False) as shard:
                    cache[ordinal] = shard["input_token_ids"].copy()
            local_sample = int(arrays["sample_ids"][i]) - int(row["start_sample"])
            seq = cache[ordinal][local_sample]
            position = int(arrays["positions"][i])
            start, end = max(0, position - args.context_radius), min(seq.size, position + args.context_radius + 1)
            record = {
                "group": group,
                "sample_id": int(arrays["sample_ids"][i]), "position": position,
                "token_id": int(arrays["token_ids"][i]), "shard_ordinal": ordinal,
                "context_start": start, "context_token_ids": [int(v) for v in seq[start:end]],
                "target_offset_in_context": position - start,
                "raw_cosine_mean_l2_l9": float(score[i]),
                "layer_cosines_l2_l9": [float(v) for v in arrays["cosine"][i]],
                "reference_rms_l2_l9": [float(v) for v in arrays["reference_rms"][i]],
            }
            if tokenizer is not None:
                context_ids = record["context_token_ids"]
                record["context_decoded"] = tokenizer.decode(context_ids, skip_special_tokens=False)
                record["target_token_decoded"] = tokenizer.decode([record["token_id"]], skip_special_tokens=False)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.output_dir / "below_active_cut_samples.json").write_text(json.dumps({
        "samples": [],
        "reason": "No active absolute norm cut: all layer distributions were clean-unimodal. Low-tail diagnostic samples are in manual_review_context_samples.jsonl."
    }, indent=2) + "\n")
    print(json.dumps({"complete": True, "records": len(selected), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
