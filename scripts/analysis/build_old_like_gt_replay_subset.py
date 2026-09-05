#!/usr/bin/env python3
"""Build the fixed contextual-sample subset that carries old-like GT tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


GT_SCHEMA = "old_like_gt_l2_l9_all_code_top1_raw_v1"
SUBSET_SCHEMA = "old_like_gt_positive_sample_subset_v1"
SUBSET_FILE = "replay_positive_sample_ids.npy"
SUBSET_METADATA = "replay_subset_metadata.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--virtual-samples", type=int, default=4_147_200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.gt_root.expanduser().resolve()
    root_metadata_path = root / "metadata.json"
    root_metadata = json.loads(root_metadata_path.read_text(encoding="utf-8"))
    if root_metadata.get("schema") != GT_SCHEMA or root_metadata.get("complete") is not True:
        raise SystemExit(f"invalid/incomplete old-like GT root: {root_metadata_path}")

    output = root / SUBSET_FILE
    metadata_output = root / SUBSET_METADATA
    if output.exists() or metadata_output.exists():
        raise SystemExit(
            "refusing to overwrite an existing replay subset; validate or move it first: "
            f"{output}, {metadata_output}"
        )

    positive_parts: list[np.ndarray] = []
    selected_tokens = 0
    expected_start = 0
    for partition in root_metadata["partitions"]:
        start = int(partition["partition_start_sample"])
        count = int(partition["partition_samples"])
        if start != expected_start:
            raise SystemExit(
                f"non-contiguous GT partition: expected {expected_start}, got {start}"
            )
        mask_path = root / partition["rank"] / partition["mask_file"]
        packed = np.load(mask_path, mmap_mode="r", allow_pickle=False)
        if packed.shape != (count, 64) or packed.dtype != np.uint8:
            raise SystemExit(f"unexpected packed mask shape/dtype: {mask_path}: {packed.shape}/{packed.dtype}")
        positive_local = np.flatnonzero(np.any(packed != 0, axis=1)).astype(np.int64)
        positive_parts.append(positive_local + start)
        selected_tokens += int(
            np.unpackbits(packed, axis=1, bitorder="little", count=512).sum(dtype=np.uint64)
        )
        expected_start += count

    total_samples = int(root_metadata["total_samples"])
    if expected_start != total_samples:
        raise SystemExit(f"GT partitions cover {expected_start}, expected {total_samples}")
    positive_ids = np.concatenate(positive_parts).astype(np.int32, copy=False)
    if positive_ids.size == 0 or np.unique(positive_ids).size != positive_ids.size:
        raise SystemExit("positive replay sample IDs are empty or duplicated")
    if selected_tokens != int(root_metadata["selected_count"]):
        raise SystemExit(
            f"selected-token count mismatch: masks={selected_tokens}, metadata={root_metadata['selected_count']}"
        )

    # Fixed permutation prevents every replay epoch from following raw corpus
    # order while remaining byte-identical across the LM and hidden-MSE runs.
    rng = np.random.default_rng(args.seed)
    positive_ids = positive_ids[rng.permutation(positive_ids.size)]

    temporary = output.with_name(output.name + ".inprogress")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, positive_ids, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        check = np.load(temporary, mmap_mode="r", allow_pickle=False)
        if check.shape != positive_ids.shape or check.dtype != np.int32:
            raise RuntimeError("replay subset round-trip shape/dtype mismatch")
        if not np.array_equal(check, positive_ids):
            raise RuntimeError("replay subset round-trip content mismatch")
        del check
        os.replace(temporary, output)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise

    payload = {
        "schema": SUBSET_SCHEMA,
        "complete": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_gt_root": str(root),
        "source_gt_schema": GT_SCHEMA,
        "source_gt_config_sha256": root_metadata.get("config_sha256"),
        "source_gt_metadata_sha256": sha256_file(root_metadata_path),
        "source_total_samples": total_samples,
        "source_selected_token_count": selected_tokens,
        "positive_sample_count": int(positive_ids.size),
        "positive_sample_fraction": float(positive_ids.size / total_samples),
        "mean_selected_tokens_per_positive_sample": float(selected_tokens / positive_ids.size),
        "ordering": "numpy_default_rng_permutation",
        "ordering_seed": args.seed,
        "subset_file": SUBSET_FILE,
        "dtype": "int32",
        "subset_sha256": sha256_file(output),
        "virtual_samples_per_1800_step_run": args.virtual_samples,
        "effective_positive_sample_epochs": float(args.virtual_samples / positive_ids.size),
    }
    atomic_json(metadata_output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
