#!/usr/bin/env python3
"""Materialize only contextual old-like GT token IDs as an indexed dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEGATRON_ROOT = PROJECT_ROOT / "Megatron-LM"
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))

from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder


OUTPUT_SCHEMA = "old_like_gt_token_packed_indexed_dataset_v1"


def sha256(path: Path) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--paired-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="train_text_document")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--target-train-fraction", type=float, default=0.20)
    parser.add_argument(
        "--target-reference-token-count",
        type=int,
        default=None,
        help=(
            "Token count whose fraction defines replay exposure. By default this is the "
            "paired extraction token count; set it when extraction covers one corpus epoch "
            "but downstream training consumes a different number of tokens."
        ),
    )
    args = parser.parse_args()

    gt_root = args.gt_root.expanduser().resolve()
    paired_root = args.paired_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / args.output_prefix
    bin_path = prefix.with_suffix(".bin")
    idx_path = prefix.with_suffix(".idx")
    metadata_path = output_dir / "miniset_metadata.json"
    if any(path.exists() for path in (bin_path, idx_path, metadata_path)):
        raise SystemExit(f"refusing to overwrite existing miniset output: {output_dir}")

    gt_metadata_path = gt_root / "metadata.json"
    gt_metadata = json.loads(gt_metadata_path.read_text(encoding="utf-8"))
    if gt_metadata.get("complete") is not True:
        raise SystemExit(f"invalid GT metadata: {gt_metadata_path}")
    if gt_metadata.get("layers") != list(range(2, 10)):
        raise SystemExit(f"GT must use residual layer outputs 2..9: {gt_metadata_path}")
    sequence_length = int(gt_metadata["sequence_length"])
    if sequence_length != args.sequence_length:
        raise SystemExit(
            f"sequence length mismatch: GT={sequence_length}, requested={args.sequence_length}"
        )

    selected_parts: list[np.ndarray] = []
    occurrence_hasher = hashlib.sha256()
    expected_global_sample = 0
    rank_summaries = []
    for partition in gt_metadata["partitions"]:
        rank_name = partition["rank"]
        start = int(partition["partition_start_sample"])
        samples = int(partition["partition_samples"])
        if start != expected_global_sample:
            raise SystemExit(
                f"non-contiguous GT partition: expected {expected_global_sample}, got {start}"
            )
        packed = np.load(
            gt_root / rank_name / partition["mask_file"], mmap_mode="r", allow_pickle=False
        )
        if packed.shape != (samples, (sequence_length + 7) // 8):
            raise SystemExit(f"bad packed GT shape for {rank_name}: {packed.shape}")

        local_selected = 0
        local_offset = 0
        shard_paths = sorted((paired_root / rank_name / "token_metrics").glob("shard_*.npz"))
        if not shard_paths:
            raise SystemExit(f"no paired shards found for {rank_name}")
        for shard_path in shard_paths:
            with np.load(shard_path, allow_pickle=False) as shard:
                sample_ids = shard["sample_ids"]
                token_ids = shard["input_token_ids"]
                valid = shard["valid_mask"].astype(bool, copy=False)
                shard_samples = int(sample_ids.size)
                expected_ids = np.arange(
                    start + local_offset,
                    start + local_offset + shard_samples,
                    dtype=sample_ids.dtype,
                )
                if not np.array_equal(sample_ids, expected_ids):
                    raise SystemExit(f"sample ID mismatch in {shard_path}")
                mask = np.unpackbits(
                    packed[local_offset : local_offset + shard_samples],
                    axis=1,
                    bitorder="little",
                    count=sequence_length,
                ).astype(bool, copy=False)
                mask &= valid
                chosen = np.asarray(token_ids[mask], dtype=np.int32)
                selected_parts.append(chosen)
                occurrence_hasher.update(chosen.tobytes(order="C"))
                local_selected += int(chosen.size)
                local_offset += shard_samples
        if local_offset != samples:
            raise SystemExit(
                f"paired shard coverage mismatch for {rank_name}: {local_offset} != {samples}"
            )
        if local_selected != int(partition["selected_count"]):
            raise SystemExit(
                f"selected count mismatch for {rank_name}: "
                f"{local_selected} != {partition['selected_count']}"
            )
        rank_summaries.append({"rank": rank_name, "selected_tokens": local_selected})
        expected_global_sample += samples

    selected_tokens = np.concatenate(selected_parts)
    selected_count = int(gt_metadata["selected_count"])
    if selected_tokens.shape != (selected_count,):
        raise SystemExit(
            f"global selected-token count mismatch: {selected_tokens.size} != {selected_count}"
        )

    temporary_bin = bin_path.with_name(bin_path.name + ".inprogress")
    temporary_idx = idx_path.with_name(idx_path.name + ".inprogress")
    try:
        builder = IndexedDatasetBuilder(str(temporary_bin), dtype=np.int32, multimodal=False)
        # A single ordered document preserves the GT occurrence stream while
        # allowing GPTDataset to form contiguous 512-token training samples.
        tensor = torch.from_numpy(selected_tokens)
        builder.add_document(tensor, [selected_count])
        builder.finalize(str(temporary_idx))
        os.replace(temporary_bin, bin_path)
        os.replace(temporary_idx, idx_path)
    except BaseException:
        for path in (temporary_bin, temporary_idx):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise

    indexed = IndexedDataset(str(prefix), mmap=True)
    if len(indexed) != 1 or int(indexed.sequence_lengths[0]) != selected_count:
        raise SystemExit(
            f"indexed dataset round-trip mismatch: docs={len(indexed)} "
            f"tokens={indexed.sequence_lengths.tolist()}"
        )
    roundtrip = np.asarray(indexed[0], dtype=np.int32)
    if not np.array_equal(roundtrip, selected_tokens):
        raise SystemExit("indexed dataset token round-trip mismatch")

    packed_samples = (selected_count - 1) // sequence_length
    extracted_tokens = int(gt_metadata["total_valid_tokens"])
    full_train_tokens = (
        int(args.target_reference_token_count)
        if args.target_reference_token_count is not None
        else extracted_tokens
    )
    if full_train_tokens <= 0:
        raise SystemExit("target reference token count must be positive")
    target_tokens = args.target_train_fraction * full_train_tokens
    repeat_epochs = target_tokens / selected_count
    target_sequence_exposures = int(round(target_tokens / sequence_length))
    payload = {
        "schema": OUTPUT_SCHEMA,
        "complete": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_gt_root": str(gt_root),
        "source_gt_metadata_sha256": sha256(gt_metadata_path),
        "source_paired_root": str(paired_root),
        "ordering": "global_sample_then_position",
        "selected_token_count": selected_count,
        "selected_token_sha256": occurrence_hasher.hexdigest(),
        "document_count": 1,
        "document_tokens": selected_count,
        "sequence_length": sequence_length,
        "available_nonoverlapping_gpt_samples": packed_samples,
        "source_extraction_token_count": extracted_tokens,
        "full_train_token_count": full_train_tokens,
        "target_train_fraction": args.target_train_fraction,
        "target_supervised_tokens": target_tokens,
        "effective_token_epochs": repeat_epochs,
        "target_sequence_exposures": target_sequence_exposures,
        "global_batch_size_2304_steps": int(round(target_sequence_exposures / 2304)),
        "bin_file": bin_path.name,
        "idx_file": idx_path.name,
        "bin_sha256": sha256(bin_path),
        "idx_sha256": sha256(idx_path),
        "rank_summaries": rank_summaries,
    }
    atomic_json(metadata_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
