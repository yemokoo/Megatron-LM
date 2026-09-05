#!/usr/bin/env python3
"""Materialize contextual Code-token pseudo-GT from fixed L2--L9 cosine cuts.

GT=1 iff every residual-included layer output cosine for Layers 2--9 is at
least its fixed full-Code top-1% threshold.  Output is packed little-endian
bits, one 64-byte row for each 512-token GPTDataset sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


SCHEMA = "old_like_gt_l2_l9_all_code_top1_raw_v1"
LAYERS = np.arange(2, 10, dtype=np.int16)
THRESHOLDS = np.asarray(
    [0.99985, 0.99015, 0.95665, 0.93955, 0.91845, 0.88835, 0.86075, 0.85235],
    dtype=np.float32,
)
SEQUENCE_LENGTH = 512
PACKED_BYTES = SEQUENCE_LENGTH // 8


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def config_hash(source: Path) -> str:
    payload = {
        "schema": SCHEMA,
        "source": str(source.resolve()),
        "layers": LAYERS.tolist(),
        "thresholds": [float(v) for v in THRESHOLDS],
        "comparison": "cosine >= threshold for every layer",
        "sequence_length": SEQUENCE_LENGTH,
        "bitorder": "little",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def dataset_identity(source: Path) -> dict:
    source_metadata = json.loads((source / "rank_000" / "metadata.json").read_text())
    prefix = Path(source_metadata["code_data_path"])
    idx_path = Path(str(prefix) + ".idx")
    bin_path = Path(str(prefix) + ".bin")
    if not idx_path.is_file() or not bin_path.is_file():
        raise RuntimeError(f"paired-extraction dataset files are missing for {prefix}")
    return {
        "paired_extraction_dataset_prefix": str(prefix),
        "idx_sha256": file_sha256(idx_path),
        "idx_bytes": idx_path.stat().st_size,
        "bin_bytes": bin_path.stat().st_size,
        "seed": int(source_metadata["seed"]),
        "split": source_metadata["dataset_split"],
        "source_config_sha256": source_metadata["config_sha256"],
    }


def scan_rank(rank_dir_string: str, output_root_string: str, digest: str, force: bool) -> dict:
    rank_dir = Path(rank_dir_string)
    output_dir = Path(output_root_string) / rank_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    source_metadata = json.loads((rank_dir / "metadata.json").read_text())
    start_sample = int(source_metadata["partition_start_sample"])
    sample_count = int(source_metadata["partition_samples"])
    shards = sorted((rank_dir / "token_metrics").glob("shard_*.npz"))
    if len(shards) != int(source_metadata["shards"]):
        raise RuntimeError(f"{rank_dir.name}: expected {source_metadata['shards']} shards, got {len(shards)}")
    final_mask = output_dir / "old_like_gt_packed.npy"
    partial_mask = output_dir / "old_like_gt_packed.inprogress.npy"
    progress_path = output_dir / "progress.json"
    final_metadata = output_dir / "metadata.json"
    if force:
        for path in (final_mask, partial_mask, progress_path, final_metadata):
            if path.exists():
                path.unlink()
    if final_mask.is_file() and final_metadata.is_file():
        old = json.loads(final_metadata.read_text())
        if old.get("complete") and old.get("config_sha256") == digest:
            return old

    next_shard = 0
    selected_count = 0
    layer_pass_counts = np.zeros(8, dtype=np.uint64)
    stable_count_hist = np.zeros(9, dtype=np.uint64)
    if progress_path.is_file() and partial_mask.is_file():
        progress = json.loads(progress_path.read_text())
        if progress.get("config_sha256") == digest:
            next_shard = int(progress["next_shard"])
            selected_count = int(progress["selected_count"])
            layer_pass_counts[:] = progress["layer_pass_counts"]
            stable_count_hist[:] = progress["stable_count_hist"]
            mask = np.lib.format.open_memmap(partial_mask, mode="r+")
            if mask.shape != (sample_count, PACKED_BYTES) or mask.dtype != np.uint8:
                raise RuntimeError(f"{rank_dir.name}: incompatible partial mask {mask.shape} {mask.dtype}")
        else:
            raise RuntimeError(f"{rank_dir.name}: incompatible progress; use --force")
    else:
        mask = np.lib.format.open_memmap(
            partial_mask, mode="w+", dtype=np.uint8, shape=(sample_count, PACKED_BYTES)
        )

    for shard_index in range(next_shard, len(shards)):
        shard = shards[shard_index]
        with np.load(shard, allow_pickle=False) as data:
            if not np.array_equal(data["layer_numbers"], np.arange(1, 10)):
                raise RuntimeError(f"{shard}: unexpected layer numbers")
            sample_ids = data["sample_ids"].astype(np.int64, copy=False)
            valid = data["valid_mask"].astype(bool, copy=False)
            cosine = data["cosine"][..., 1:]
            expected = np.arange(sample_ids[0], sample_ids[0] + sample_ids.size, dtype=np.int64)
            if not np.array_equal(sample_ids, expected):
                raise RuntimeError(f"{shard}: sample IDs are not contiguous")
            local_start = int(sample_ids[0] - start_sample)
            local_end = local_start + sample_ids.size
            if local_start < 0 or local_end > sample_count:
                raise RuntimeError(f"{shard}: sample range outside partition")
            layer_pass = cosine >= THRESHOLDS.reshape(1, 1, 8)
            stable_count = layer_pass.sum(axis=2, dtype=np.uint8)
            selected = layer_pass.all(axis=2) & valid
            mask[local_start:local_end] = np.packbits(selected, axis=1, bitorder="little")
            selected_count += int(selected.sum(dtype=np.uint64))
            layer_pass_counts += (layer_pass & valid[..., None]).sum(axis=(0, 1), dtype=np.uint64)
            stable_count_hist += np.bincount(stable_count[valid], minlength=9).astype(np.uint64)
        mask.flush()
        atomic_json(progress_path, {
            "schema": SCHEMA, "config_sha256": digest, "next_shard": shard_index + 1,
            "selected_count": selected_count,
            "layer_pass_counts": [int(v) for v in layer_pass_counts],
            "stable_count_hist": [int(v) for v in stable_count_hist],
        })

    del mask
    os.replace(partial_mask, final_mask)
    if progress_path.exists():
        progress_path.unlink()
    total_tokens = int(stable_count_hist.sum(dtype=np.uint64))
    metadata = {
        "schema": SCHEMA,
        "complete": True,
        "config_sha256": digest,
        "rank": rank_dir.name,
        "source_rank_dir": str(rank_dir.resolve()),
        "partition_start_sample": start_sample,
        "partition_samples": sample_count,
        "sequence_length": SEQUENCE_LENGTH,
        "total_valid_tokens": total_tokens,
        "selected_count": selected_count,
        "selected_fraction": selected_count / total_tokens,
        "layer_pass_counts": [int(v) for v in layer_pass_counts],
        "stable_count_hist": [int(v) for v in stable_count_hist],
        "mask_file": final_mask.name,
        "mask_shape": [sample_count, PACKED_BYTES],
        "mask_dtype": "uint8",
        "bitorder": "little",
        "bit_semantics": "unpackbits(row, bitorder='little')[position] == old_like_gt",
        "mask_sha256": file_sha256(final_mask),
    }
    atomic_json(final_metadata, metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ranks = sorted(source.glob("rank_[0-9][0-9][0-9]"))
    if len(ranks) != 8:
        raise RuntimeError(f"expected 8 source ranks, got {len(ranks)}")
    digest = config_hash(source)
    rows = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(ranks))) as pool:
        futures = {pool.submit(scan_rank, str(rank), str(output), digest, args.force): rank for rank in ranks}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"[{row['rank']}] selected={row['selected_count']:,} ({row['selected_fraction']:.6%})", flush=True)
    rows.sort(key=lambda row: row["partition_start_sample"])
    expected_start = 0
    for row in rows:
        if row["partition_start_sample"] != expected_start:
            raise RuntimeError(f"partition gap/overlap before {row['rank']}")
        expected_start += row["partition_samples"]
    selected = sum(row["selected_count"] for row in rows)
    total = sum(row["total_valid_tokens"] for row in rows)
    root_metadata = {
        "schema": SCHEMA,
        "complete": True,
        "gt_name": "old_like_l2_l9_all_code_top1_raw",
        "gt_positive": "all eight residual-included layer-output cosines meet fixed per-layer raw cuts",
        "gt_negative": "at least one of Layers 2-9 falls below its cut",
        "semantic_claim": "Wiki-like stability pseudo-GT; not a semantic Wiki-domain ground truth",
        "source_root": str(source),
        "dataset_identity": dataset_identity(source),
        "config_sha256": digest,
        "layers": LAYERS.tolist(),
        "thresholds": [float(v) for v in THRESHOLDS],
        "comparison": ">=",
        "total_samples": expected_start,
        "sequence_length": SEQUENCE_LENGTH,
        "total_valid_tokens": total,
        "selected_count": selected,
        "selected_fraction": selected / total,
        "partitions": [{k: row[k] for k in (
            "rank", "partition_start_sample", "partition_samples", "total_valid_tokens",
            "selected_count", "selected_fraction", "mask_file", "mask_sha256"
        )} for row in rows],
    }
    atomic_json(output / "metadata.json", root_metadata)
    print(json.dumps(root_metadata, indent=2))


if __name__ == "__main__":
    main()
