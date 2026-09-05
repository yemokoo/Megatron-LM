#!/usr/bin/env python3
"""Build task-token pseudo-GT from per-layer full-stream cosine top 1%.

The source is a completed paired extraction.  Pass 1 computes a fixed
20,000-bin histogram over raw cosine in [-1, 1] for residual-included Layers
2--9.  Each layer cut is the center of the bin containing its upper 1% point,
matching the rule used by the Code experiment.  Pass 2 compares the original
float32 cosine values to those cuts.  GT=1 only when all eight layers pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


SCHEMA = "old_like_gt_l2_l9_all_task_top1_raw_v1"
LAYERS = list(range(2, 10))


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def load_rank_metadata(rank: Path) -> dict:
    metadata = json.loads((rank / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("completed") is not True:
        raise RuntimeError(f"paired rank is incomplete: {rank}")
    if metadata.get("layers") != list(range(1, 10)):
        raise RuntimeError(f"paired rank does not contain Layers 1..9: {rank}")
    return metadata


def rank_shards(rank: Path, metadata: dict) -> list[Path]:
    shards = sorted((rank / "token_metrics").glob("shard_*.npz"))
    if len(shards) != int(metadata["shards"]):
        raise RuntimeError(f"{rank.name}: expected {metadata['shards']} shards, got {len(shards)}")
    return shards


def histogram_rank(rank_string: str, bins: int) -> dict:
    rank = Path(rank_string)
    metadata = load_rank_metadata(rank)
    counts = np.zeros((8, bins), dtype=np.uint64)
    valid_tokens = 0
    for shard_path in rank_shards(rank, metadata):
        with np.load(shard_path, allow_pickle=False) as shard:
            if not np.array_equal(shard["layer_numbers"], np.arange(1, 10)):
                raise RuntimeError(f"unexpected layers in {shard_path}")
            valid = shard["valid_mask"].astype(bool, copy=False)
            cosine = shard["cosine"][..., 1:]
            if not np.isfinite(cosine[valid]).all():
                raise RuntimeError(f"non-finite cosine in {shard_path}")
            for layer_index in range(8):
                hist, _ = np.histogram(cosine[..., layer_index][valid], bins=bins, range=(-1.0, 1.0))
                counts[layer_index] += hist.astype(np.uint64)
            valid_tokens += int(valid.sum(dtype=np.uint64))
    if not np.all(counts.sum(axis=1, dtype=np.uint64) == valid_tokens):
        raise RuntimeError(f"histogram count mismatch in {rank}")
    return {
        "rank": rank.name,
        "counts": counts,
        "valid_tokens": valid_tokens,
        "start": int(metadata["partition_start_sample"]),
        "samples": int(metadata["partition_samples"]),
    }


def derive_thresholds(counts: np.ndarray, bins: int, top_fraction: float) -> tuple[list[float], list[dict]]:
    width = 2.0 / bins
    thresholds = []
    details = []
    for layer_index, layer_counts in enumerate(counts):
        total = int(layer_counts.sum(dtype=np.uint64))
        target = int(math.ceil(total * top_fraction))
        reverse_cumulative = np.cumsum(layer_counts[::-1], dtype=np.uint64)
        reverse_index = int(np.searchsorted(reverse_cumulative, target, side="left"))
        bin_index = bins - 1 - reverse_index
        threshold = -1.0 + (bin_index + 0.5) * width
        above_bin = int(layer_counts[bin_index + 1 :].sum(dtype=np.uint64))
        boundary = int(layer_counts[bin_index])
        thresholds.append(float(np.float32(threshold)))
        details.append({
            "layer": LAYERS[layer_index],
            "total": total,
            "target_top_count": target,
            "threshold_bin_index": bin_index,
            "threshold_bin_center": threshold,
            "count_strictly_above_boundary_bin": above_bin,
            "boundary_bin_count": boundary,
        })
    return thresholds, details


def config_hash(source: Path, task: str, bins: int, top_fraction: float, thresholds: list[float]) -> str:
    payload = {
        "schema": SCHEMA,
        "source": str(source),
        "task": task,
        "layers": LAYERS,
        "histogram_bins": bins,
        "histogram_range": [-1.0, 1.0],
        "top_fraction": top_fraction,
        "thresholds": thresholds,
        "comparison": "raw float32 cosine >= per-layer threshold for every layer",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def mask_rank(rank_string: str, output_string: str, thresholds: list[float], digest: str) -> dict:
    rank = Path(rank_string)
    output = Path(output_string) / rank.name
    output.mkdir(parents=True, exist_ok=True)
    metadata = load_rank_metadata(rank)
    start = int(metadata["partition_start_sample"])
    samples = int(metadata["partition_samples"])
    sequence_length = int(metadata["sequence_length"])
    packed_bytes = (sequence_length + 7) // 8
    shards = rank_shards(rank, metadata)
    final_mask = output / "old_like_gt_packed.npy"
    final_metadata = output / "metadata.json"
    if final_mask.is_file() and final_metadata.is_file():
        previous = json.loads(final_metadata.read_text(encoding="utf-8"))
        if previous.get("complete") and previous.get("config_sha256") == digest:
            return previous
        raise RuntimeError(f"incompatible completed GT output: {output}")

    partial = output / "old_like_gt_packed.inprogress.npy"
    mask = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint8, shape=(samples, packed_bytes))
    cuts = np.asarray(thresholds, dtype=np.float32).reshape(1, 1, 8)
    selected_count = 0
    layer_pass_counts = np.zeros(8, dtype=np.uint64)
    stable_count_hist = np.zeros(9, dtype=np.uint64)
    local_offset = 0
    for shard_path in shards:
        with np.load(shard_path, allow_pickle=False) as shard:
            sample_ids = shard["sample_ids"].astype(np.int64, copy=False)
            expected = np.arange(start + local_offset, start + local_offset + sample_ids.size)
            if not np.array_equal(sample_ids, expected):
                raise RuntimeError(f"sample ID mismatch in {shard_path}")
            valid = shard["valid_mask"].astype(bool, copy=False)
            layer_pass = shard["cosine"][..., 1:] >= cuts
            stable_count = layer_pass.sum(axis=2, dtype=np.uint8)
            selected = layer_pass.all(axis=2) & valid
            end = local_offset + sample_ids.size
            mask[local_offset:end] = np.packbits(selected, axis=1, bitorder="little")
            selected_count += int(selected.sum(dtype=np.uint64))
            layer_pass_counts += (layer_pass & valid[..., None]).sum(axis=(0, 1), dtype=np.uint64)
            stable_count_hist += np.bincount(stable_count[valid], minlength=9).astype(np.uint64)
            local_offset = end
    if local_offset != samples:
        raise RuntimeError(f"rank sample coverage mismatch: {rank}")
    mask.flush()
    del mask
    os.replace(partial, final_mask)
    total_tokens = int(stable_count_hist.sum(dtype=np.uint64))
    result = {
        "schema": SCHEMA,
        "complete": True,
        "config_sha256": digest,
        "rank": rank.name,
        "partition_start_sample": start,
        "partition_samples": samples,
        "sequence_length": sequence_length,
        "total_valid_tokens": total_tokens,
        "selected_count": selected_count,
        "selected_fraction": selected_count / total_tokens,
        "layer_pass_counts": [int(v) for v in layer_pass_counts],
        "stable_count_hist": [int(v) for v in stable_count_hist],
        "mask_file": final_mask.name,
        "mask_shape": [samples, packed_bytes],
        "mask_dtype": "uint8",
        "bitorder": "little",
        "mask_sha256": sha256(final_mask),
    }
    atomic_json(final_metadata, result)
    return result


def dataset_identity(source: Path) -> dict:
    metadata = load_rank_metadata(source / "rank_000")
    blend = metadata.get("data_blend") or metadata.get("code_data_blend")
    if not blend:
        raise RuntimeError("paired metadata has no valid data blend")
    # Paired extraction supports both the historical explicit weighted form
    # [weight, prefix, ...] and an exhaustive form [prefix, ...].  The latter
    # lets Megatron derive weights from each shard's actual one-epoch length,
    # avoiding repeated small shards in a corpus-equivalent pass.
    unweighted = all(Path(str(prefix) + ".bin").is_file() for prefix in blend)
    if unweighted:
        entries = [("exhaustive", prefix) for prefix in blend]
    else:
        if len(blend) % 2:
            raise RuntimeError("weighted paired data blend must contain weight/prefix pairs")
        entries = [(blend[index - 1], blend[index]) for index in range(1, len(blend), 2)]
    shards = []
    for weight, raw_prefix in entries:
        prefix = Path(raw_prefix)
        bin_path = Path(str(prefix) + ".bin")
        idx_path = Path(str(prefix) + ".idx")
        if not bin_path.is_file() or not idx_path.is_file():
            raise RuntimeError(f"dataset shard missing: {prefix}")
        shards.append({
            "weight": str(weight),
            "prefix": str(prefix),
            "bin_bytes": bin_path.stat().st_size,
            "idx_bytes": idx_path.stat().st_size,
            "idx_sha256": sha256(idx_path),
        })
    return {
        "blend_mode": "exhaustive" if unweighted else "explicit_weights",
        "seed": int(metadata["seed"]),
        "split": metadata["dataset_split"],
        "source_config_sha256": metadata["config_sha256"],
        "shards": shards,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bins", type=int, default=20_000)
    parser.add_argument("--top-fraction", type=float, default=0.01)
    args = parser.parse_args()
    if args.bins < 200 or not 0.0 < args.top_fraction < 1.0:
        raise SystemExit("invalid histogram bins/top fraction")
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    ranks = sorted(source.glob("rank_[0-9][0-9][0-9]"))
    if not ranks:
        raise RuntimeError(f"no paired ranks under {source}")
    output.mkdir(parents=True, exist_ok=True)

    histogram = np.zeros((8, args.bins), dtype=np.uint64)
    rows = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(ranks))) as pool:
        futures = {pool.submit(histogram_rank, str(rank), args.bins): rank for rank in ranks}
        for future in as_completed(futures):
            row = future.result()
            histogram += row.pop("counts")
            rows.append(row)
            print(f"[HIST] {row['rank']} tokens={row['valid_tokens']:,}", flush=True)
    rows.sort(key=lambda row: row["start"])
    expected_start = 0
    for row in rows:
        if row["start"] != expected_start:
            raise RuntimeError(f"partition gap/overlap before {row['rank']}")
        expected_start += row["samples"]
    thresholds, threshold_details = derive_thresholds(histogram, args.bins, args.top_fraction)
    digest = config_hash(source, args.task, args.bins, args.top_fraction, thresholds)
    threshold_payload = {
        "schema": SCHEMA,
        "task": args.task,
        "layers": LAYERS,
        "histogram_bins": args.bins,
        "histogram_range": [-1.0, 1.0],
        "top_fraction": args.top_fraction,
        "threshold_method": "center of fixed-width bin containing upper-tail target",
        "thresholds": thresholds,
        "details": threshold_details,
        "config_sha256": digest,
    }
    atomic_json(output / "thresholds.json", threshold_payload)
    np.savez(output / "layer_cosine_histograms.npz", counts=histogram, layers=LAYERS)
    print("[THRESHOLDS] " + json.dumps(thresholds), flush=True)

    mask_rows = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(ranks))) as pool:
        futures = {
            pool.submit(mask_rank, str(rank), str(output), thresholds, digest): rank
            for rank in ranks
        }
        for future in as_completed(futures):
            row = future.result()
            mask_rows.append(row)
            print(f"[GT] {row['rank']} selected={row['selected_count']:,}", flush=True)
    mask_rows.sort(key=lambda row: row["partition_start_sample"])
    selected = sum(row["selected_count"] for row in mask_rows)
    total = sum(row["total_valid_tokens"] for row in mask_rows)
    metadata = {
        "schema": SCHEMA,
        "complete": True,
        "task": args.task,
        "gt_name": f"old_like_l2_l9_all_{args.task}_top1_raw",
        "gt_positive": "all eight residual-included layer-output cosines meet task-specific per-layer full-stream top-1% cuts",
        "semantic_claim": "old-like stability pseudo-GT; not semantic-domain ground truth",
        "source_root": str(source),
        "dataset_identity": dataset_identity(source),
        "config_sha256": digest,
        "layers": LAYERS,
        "thresholds": thresholds,
        "comparison": ">= on original float32 cosine",
        "threshold_method": threshold_payload["threshold_method"],
        "total_samples": expected_start,
        "sequence_length": int(mask_rows[0]["sequence_length"]),
        "total_valid_tokens": total,
        "selected_count": selected,
        "selected_fraction": selected / total,
        "partitions": [{k: row[k] for k in (
            "rank", "partition_start_sample", "partition_samples", "total_valid_tokens",
            "selected_count", "selected_fraction", "mask_file", "mask_sha256"
        )} for row in mask_rows],
    }
    atomic_json(output / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
