#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split pre-tokenized shard_*.bin/.idx files into two physical directories by cumulative .bin size."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-seen-dir", required=True)
    parser.add_argument("--output-holdout-dir", required=True)
    parser.add_argument("--seen-steps", type=int, default=1800)
    parser.add_argument("--holdout-steps", type=int, default=74)
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Use hard links by default to avoid duplicating the large shard files.",
    )
    return parser.parse_args()


def collect_shards(input_dir: Path) -> list[dict]:
    shards = []
    for bin_path in sorted(input_dir.glob("shard_*_text_document.bin")):
        idx_path = bin_path.with_suffix(".idx")
        if not idx_path.exists():
            raise FileNotFoundError(f"Missing idx for {bin_path}")
        shards.append(
            {
                "prefix": bin_path.stem.replace("_text_document", ""),
                "bin_path": bin_path,
                "idx_path": idx_path,
                "bin_size": bin_path.stat().st_size,
                "idx_size": idx_path.stat().st_size,
            }
        )

    if not shards:
        raise RuntimeError(f"No shard_*.bin files found under {input_dir}")
    return shards


def choose_split_index(shards: list[dict], seen_ratio: float) -> tuple[int, int, int]:
    total_bytes = sum(shard["bin_size"] for shard in shards)
    target_bytes = total_bytes * seen_ratio

    cumulative = 0
    best_index = 1
    best_distance = None
    best_seen_bytes = 0

    for index, shard in enumerate(shards[:-1], start=1):
        cumulative += shard["bin_size"]
        distance = abs(cumulative - target_bytes)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_index = index
            best_seen_bytes = cumulative

    return best_index, best_seen_bytes, total_bytes


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    if mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def materialize_split(shards: list[dict], output_dir: Path, mode: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        link_or_copy(shard["bin_path"], output_dir / shard["bin_path"].name, mode)
        link_or_copy(shard["idx_path"], output_dir / shard["idx_path"].name, mode)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_seen_dir = Path(args.output_seen_dir)
    output_holdout_dir = Path(args.output_holdout_dir)

    total_steps = args.seen_steps + args.holdout_steps
    seen_ratio = args.seen_steps / total_steps

    shards = collect_shards(input_dir)
    split_index, seen_bytes, total_bytes = choose_split_index(shards, seen_ratio)

    seen_shards = shards[:split_index]
    holdout_shards = shards[split_index:]

    materialize_split(seen_shards, output_seen_dir, args.link_mode)
    materialize_split(holdout_shards, output_holdout_dir, args.link_mode)

    metadata = {
        "input_dir": str(input_dir),
        "seen_steps": args.seen_steps,
        "holdout_steps": args.holdout_steps,
        "seen_ratio_target": seen_ratio,
        "link_mode": args.link_mode,
        "total_shards": len(shards),
        "seen_shards": len(seen_shards),
        "holdout_shards": len(holdout_shards),
        "total_bin_bytes": total_bytes,
        "seen_bin_bytes": seen_bytes,
        "holdout_bin_bytes": total_bytes - seen_bytes,
        "seen_bin_ratio_actual": seen_bytes / total_bytes,
        "seen_prefixes": [shard["prefix"] for shard in seen_shards],
        "holdout_prefixes": [shard["prefix"] for shard in holdout_shards],
    }

    for output_dir in (output_seen_dir, output_holdout_dir):
        (output_dir / "split_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
