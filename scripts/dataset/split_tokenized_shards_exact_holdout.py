#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEGATRON_ROOT = PROJECT_ROOT / "Megatron-LM"
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))

from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split tokenized shards into an exact prefix seen set and the immediately following exact holdout set."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-seen-dir", required=True)
    parser.add_argument("--output-holdout-dir", required=True)
    parser.add_argument("--seen-tokens", type=int, default=2123366400)
    parser.add_argument("--holdout-tokens", type=int, default=87293952)
    parser.add_argument("--link-full-shards", action="store_true", default=True)
    return parser.parse_args()


def shard_prefixes(input_dir: Path) -> list[Path]:
    prefixes = sorted(path.with_suffix("") for path in input_dir.glob("shard_*_text_document.bin"))
    if not prefixes:
        raise RuntimeError(f"No shard bin files found under {input_dir}")
    return prefixes


def hardlink_pair(src_prefix: Path, dst_prefix: Path) -> None:
    dst_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".bin", ".idx"):
        src = src_prefix.with_suffix(suffix)
        dst = dst_prefix.with_suffix(suffix)
        if dst.exists():
            continue
        os.link(src, dst)


def get_builder(builders: dict[str, IndexedDatasetBuilder], output_prefix: Path) -> IndexedDatasetBuilder:
    key = str(output_prefix)
    if key not in builders:
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        builders[key] = IndexedDatasetBuilder(str(output_prefix.with_suffix(".bin")), multimodal=False)
    return builders[key]


def finalize_builders(builders: dict[str, IndexedDatasetBuilder]) -> None:
    for output_prefix, builder in builders.items():
        builder.finalize(str(Path(output_prefix).with_suffix(".idx")))


def write_piece(builder: IndexedDatasetBuilder, piece: numpy.ndarray) -> int:
    if piece.size == 0:
        return 0
    tensor = torch.from_numpy(numpy.asarray(piece, dtype=numpy.int32))
    builder.add_document(tensor, [int(tensor.numel())])
    return int(tensor.numel())


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_seen_dir = Path(args.output_seen_dir)
    output_holdout_dir = Path(args.output_holdout_dir)

    seen_end = args.seen_tokens
    holdout_end = args.seen_tokens + args.holdout_tokens

    prefixes = shard_prefixes(input_dir)
    shards = []
    total_tokens = 0
    for prefix in prefixes:
        ds = IndexedDataset(str(prefix), multimodal=False, mmap=True)
        shard_tokens = int(ds.sequence_lengths.sum())
        shards.append({"prefix": prefix, "dataset": ds, "tokens": shard_tokens})
        total_tokens += shard_tokens

    if holdout_end > total_tokens:
        raise ValueError(
            f"Requested seen+holdout tokens ({holdout_end}) exceeds total dataset tokens ({total_tokens})"
        )

    seen_partial_builders: dict[str, IndexedDatasetBuilder] = {}
    holdout_partial_builders: dict[str, IndexedDatasetBuilder] = {}
    actual_seen_tokens = 0
    actual_holdout_tokens = 0
    boundary_shards: list[str] = []

    global_start = 0
    for shard in shards:
        shard_start = global_start
        shard_end = global_start + shard["tokens"]
        dst_name = shard["prefix"].name

        if shard_end <= seen_end:
            hardlink_pair(shard["prefix"], output_seen_dir / dst_name)
            actual_seen_tokens += shard["tokens"]
        elif shard_start >= holdout_end:
            break
        elif shard_start >= seen_end and shard_end <= holdout_end:
            hardlink_pair(shard["prefix"], output_holdout_dir / dst_name)
            actual_holdout_tokens += shard["tokens"]
        else:
            boundary_shards.append(dst_name)
            ds = shard["dataset"]
            shard_token_offset = 0
            seen_builder = None
            holdout_builder = None

            for doc_idx in range(len(ds)):
                doc = ds[doc_idx]
                doc_len = int(doc.shape[0])
                doc_global_start = shard_start + shard_token_offset
                doc_global_end = doc_global_start + doc_len

                seen_overlap_start = max(doc_global_start, 0)
                seen_overlap_end = min(doc_global_end, seen_end)
                if seen_overlap_end > seen_overlap_start:
                    local_start = seen_overlap_start - doc_global_start
                    local_end = seen_overlap_end - doc_global_start
                    seen_builder = get_builder(seen_partial_builders, output_seen_dir / dst_name)
                    actual_seen_tokens += write_piece(seen_builder, doc[local_start:local_end])

                holdout_overlap_start = max(doc_global_start, seen_end)
                holdout_overlap_end = min(doc_global_end, holdout_end)
                if holdout_overlap_end > holdout_overlap_start:
                    local_start = holdout_overlap_start - doc_global_start
                    local_end = holdout_overlap_end - doc_global_start
                    holdout_builder = get_builder(holdout_partial_builders, output_holdout_dir / dst_name)
                    actual_holdout_tokens += write_piece(holdout_builder, doc[local_start:local_end])

                shard_token_offset += doc_len
                if doc_global_start >= holdout_end:
                    break

        global_start = shard_end

    finalize_builders(seen_partial_builders)
    finalize_builders(holdout_partial_builders)

    metadata = {
        "input_dir": str(input_dir),
        "total_tokens": total_tokens,
        "seen_tokens_target": args.seen_tokens,
        "holdout_tokens_target": args.holdout_tokens,
        "seen_tokens_actual": actual_seen_tokens,
        "holdout_tokens_actual": actual_holdout_tokens,
        "discarded_suffix_tokens": total_tokens - actual_seen_tokens - actual_holdout_tokens,
        "boundary_shards": boundary_shards,
    }

    for output_dir in (output_seen_dir, output_holdout_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "split_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
