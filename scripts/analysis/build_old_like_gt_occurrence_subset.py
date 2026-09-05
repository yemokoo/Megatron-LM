#!/usr/bin/env python3
"""Build one deterministic replay item per contextual old-like GT token."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


GT_SCHEMAS = {
    "old_like_gt_l2_l9_all_code_top1_raw_v1",
    "old_like_gt_l2_l9_all_task_top1_raw_v1",
}
SCHEMA = "old_like_gt_token_occurrence_subset_v1"
SAMPLE_FILE = "replay_occurrence_sample_ids.npy"
POSITION_FILE = "replay_occurrence_positions.npy"
METADATA_FILE = "replay_occurrence_metadata.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".inprogress")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        check = np.load(temporary, mmap_mode="r", allow_pickle=False)
        if check.shape != value.shape or check.dtype != value.dtype:
            raise RuntimeError(f"round-trip shape/dtype mismatch for {path}")
        del check
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--virtual-samples", type=int, default=4_147_200)
    args = parser.parse_args()

    root = args.gt_root.expanduser().resolve()
    root_metadata_path = root / "metadata.json"
    root_metadata = json.loads(root_metadata_path.read_text(encoding="utf-8"))
    source_gt_schema = root_metadata.get("schema")
    if source_gt_schema not in GT_SCHEMAS or root_metadata.get("complete") is not True:
        raise SystemExit(f"invalid/incomplete GT root: {root_metadata_path}")

    sample_output = root / SAMPLE_FILE
    position_output = root / POSITION_FILE
    metadata_output = root / METADATA_FILE
    if any(path.exists() for path in (sample_output, position_output, metadata_output)):
        raise SystemExit("refusing to overwrite an existing token-occurrence replay artifact")

    sample_parts: list[np.ndarray] = []
    position_parts: list[np.ndarray] = []
    expected_start = 0
    for partition in root_metadata["partitions"]:
        start = int(partition["partition_start_sample"])
        count = int(partition["partition_samples"])
        if start != expected_start:
            raise SystemExit(f"non-contiguous partition: expected {expected_start}, got {start}")
        packed = np.load(
            root / partition["rank"] / partition["mask_file"],
            mmap_mode="r",
            allow_pickle=False,
        )
        unpacked = np.unpackbits(packed, axis=1, bitorder="little", count=512)
        local_samples, positions = np.nonzero(unpacked)
        sample_parts.append((local_samples.astype(np.int64) + start).astype(np.int32))
        position_parts.append(positions.astype(np.uint16))
        expected_start += count

    sample_ids = np.concatenate(sample_parts)
    positions = np.concatenate(position_parts)
    expected_count = int(root_metadata["selected_count"])
    if sample_ids.shape != (expected_count,) or positions.shape != (expected_count,):
        raise SystemExit(
            f"occurrence count mismatch: {sample_ids.size}/{positions.size}, expected {expected_count}"
        )

    permutation = np.random.default_rng(args.seed).permutation(expected_count)
    sample_ids = sample_ids[permutation]
    positions = positions[permutation]
    atomic_npy(sample_output, sample_ids)
    atomic_npy(position_output, positions)

    payload = {
        "schema": SCHEMA,
        "complete": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_gt_root": str(root),
        "source_gt_schema": source_gt_schema,
        "source_gt_config_sha256": root_metadata.get("config_sha256"),
        "source_gt_metadata_sha256": file_sha256(root_metadata_path),
        "source_total_samples": int(root_metadata["total_samples"]),
        "occurrence_count": expected_count,
        "sample_file": SAMPLE_FILE,
        "position_file": POSITION_FILE,
        "sample_dtype": "int32",
        "position_dtype": "uint16",
        "sample_sha256": file_sha256(sample_output),
        "position_sha256": file_sha256(position_output),
        "ordering": "numpy_default_rng_permutation",
        "ordering_seed": args.seed,
        "virtual_samples_per_1800_step_run": args.virtual_samples,
        "effective_occurrence_epochs": float(args.virtual_samples / expected_count),
        "direct_supervised_positions_per_replay_item": 1,
    }
    atomic_json(metadata_output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
