#!/usr/bin/env python3
"""Uniform deterministic context sample from packed old-like GT positives."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = values.astype(np.uint64) + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260812)
    p.add_argument("--context-radius", type=int, default=12)
    args = p.parse_args()
    metadata = json.loads((args.gt_root / "metadata.json").read_text())
    candidates: list[tuple[int, int, int]] = []
    for row in metadata["partitions"]:
        packed = np.load(args.gt_root / row["rank"] / row["mask_file"], mmap_mode="r")
        start_sample = int(row["partition_start_sample"])
        for begin in range(0, packed.shape[0], 16_384):
            bits = np.unpackbits(packed[begin:begin + 16_384], axis=1, bitorder="little")
            dense = np.flatnonzero(bits.reshape(-1))
            if not dense.size:
                continue
            sample_ids = start_sample + begin + dense // 512
            positions = dense % 512
            occurrence = sample_ids.astype(np.uint64) * np.uint64(512) + positions.astype(np.uint64)
            priority = splitmix64(occurrence, args.seed)
            take = min(args.samples, priority.size)
            local = np.argpartition(priority, take - 1)[:take] if take < priority.size else np.arange(take)
            candidates.extend((int(priority[i]), int(sample_ids[i]), int(positions[i])) for i in local)
            if len(candidates) > args.samples * 8:
                candidates = sorted(candidates)[:args.samples]
    candidates = sorted(candidates)[:args.samples]

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    source_root = Path(metadata["source_root"])
    partition_by_sample = []
    for row in metadata["partitions"]:
        source_rank = source_root / row["rank"]
        shards = sorted((source_rank / "token_metrics").glob("shard_*.npz"))
        partition_by_sample.append((int(row["partition_start_sample"]), int(row["partition_samples"]), shards))
    grouped = defaultdict(list)
    for priority, sample_id, position in candidates:
        for start, count, shards in partition_by_sample:
            if start <= sample_id < start + count:
                shard_index = (sample_id - start) // 4800
                grouped[shards[shard_index]].append((priority, sample_id, position))
                break
        else:
            raise RuntimeError(f"sample {sample_id} outside GT partitions")
    records = []
    for shard, group in grouped.items():
        with np.load(shard, allow_pickle=False) as z:
            sample_ids = z["sample_ids"].copy()
            inputs = z["input_token_ids"].copy()
            cosine = z["cosine"][..., 1:].copy()
        for priority, sample_id, position in group:
            local = sample_id - int(sample_ids[0])
            sequence = inputs[local]
            left, right = max(0, position - args.context_radius), min(512, position + args.context_radius + 1)
            ids = [int(v) for v in sequence[left:right]]
            record = {
                "sample_id": sample_id, "position": position, "global_token_offset": sample_id * 512 + position,
                "token_id": int(sequence[position]), "context_start": left, "context_token_ids": ids,
                "target_offset_in_context": position - left,
                "cosine_l2_l9": [float(v) for v in cosine[local, position]],
                "cosine_mean_l2_l9": float(cosine[local, position].mean()),
                "selection_priority": priority,
            }
            if tokenizer:
                record["target_token_decoded"] = tokenizer.decode([record["token_id"]], skip_special_tokens=False)
                record["context_decoded"] = tokenizer.decode(ids, skip_special_tokens=False)
            records.append(record)
    records.sort(key=lambda row: row["selection_priority"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"complete": True, "samples": len(candidates), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
