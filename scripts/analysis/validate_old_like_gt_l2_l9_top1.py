#!/usr/bin/env python3
"""Validate packed old-like GT identity and optionally all source cosine shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from build_old_like_gt_l2_l9_top1 import THRESHOLDS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def validate_rank(args: tuple[dict, str, bool]) -> dict:
    row, root_string, deep = args
    root = Path(root_string)
    rank_dir = root / row["rank"]
    metadata = json.loads((rank_dir / "metadata.json").read_text())
    mask_path = rank_dir / metadata["mask_file"]
    if sha256(mask_path) != metadata["mask_sha256"] or metadata["mask_sha256"] != row["mask_sha256"]:
        raise RuntimeError(f"{row['rank']}: mask SHA mismatch")
    packed = np.load(mask_path, mmap_mode="r")
    expected_shape = (int(row["partition_samples"]), 64)
    if packed.shape != expected_shape or packed.dtype != np.uint8:
        raise RuntimeError(f"{row['rank']}: mask shape/dtype {packed.shape}/{packed.dtype}")
    selected = int(np.unpackbits(packed, axis=1, bitorder="little").sum(dtype=np.uint64))
    if selected != int(row["selected_count"]) or selected != int(metadata["selected_count"]):
        raise RuntimeError(f"{row['rank']}: selected count mismatch {selected}")
    checked_shards = 0
    if deep:
        source = Path(metadata["source_rank_dir"])
        start = int(metadata["partition_start_sample"])
        for shard in sorted((source / "token_metrics").glob("shard_*.npz")):
            with np.load(shard, allow_pickle=False) as z:
                ids = z["sample_ids"].astype(np.int64, copy=False)
                expected = np.all(z["cosine"][..., 1:] >= THRESHOLDS.reshape(1, 1, 8), axis=2)
                expected &= z["valid_mask"].astype(bool, copy=False)
            begin = int(ids[0] - start)
            actual = np.unpackbits(packed[begin:begin + ids.size], axis=1, bitorder="little").astype(bool)
            if not np.array_equal(actual, expected):
                mismatch = int(np.count_nonzero(actual != expected))
                raise RuntimeError(f"{row['rank']} {shard.name}: {mismatch} GT mismatches")
            checked_shards += 1
    return {"rank": row["rank"], "selected_count": selected, "checked_shards": checked_shards}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-root", required=True, type=Path)
    p.add_argument("--deep", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    root = args.gt_root.resolve()
    metadata = json.loads((root / "metadata.json").read_text())
    rows = metadata["partitions"]
    expected_start = 0
    for row in rows:
        if int(row["partition_start_sample"]) != expected_start:
            raise RuntimeError("partition gap/overlap")
        expected_start += int(row["partition_samples"])
    with ProcessPoolExecutor(max_workers=min(args.workers, len(rows))) as pool:
        results = list(pool.map(validate_rank, [(row, str(root), args.deep) for row in rows]))
    selected = sum(row["selected_count"] for row in results)
    if selected != int(metadata["selected_count"]):
        raise RuntimeError(f"root selected mismatch {selected}")
    report = {"complete": True, "deep": args.deep, "selected_count": selected,
              "total_samples": expected_start, "ranks": results}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
