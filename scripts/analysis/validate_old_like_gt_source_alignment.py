#!/usr/bin/env python3
"""Validate contextual GT sample IDs against paired-extraction token records."""

import argparse
import json
import os
import tempfile
import types
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--paired-root", required=True)
    parser.add_argument("--data-cache", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--checks", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def paired_record(paired_root: Path, sample_id: int):
    for rank_dir in sorted(paired_root.glob("rank_*")):
        metadata_path = rank_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text())
        start = int(metadata["partition_start_sample"])
        count = int(metadata["partition_samples"])
        if start <= sample_id < start + count:
            shard_samples = int(metadata["shard_samples"])
            shard_index = (sample_id - start) // shard_samples
            path = rank_dir / "token_metrics" / f"shard_{shard_index:06d}.npz"
            with np.load(path) as payload:
                ids = payload["sample_ids"]
                rows = np.flatnonzero(ids == sample_id)
                if rows.size != 1:
                    raise RuntimeError(
                        f"sample {sample_id} occurs {rows.size} times in {path}"
                    )
                row = int(rows[0])
                return (
                    payload["input_token_ids"][row].astype(np.int64),
                    payload["label_token_ids"][row].astype(np.int64),
                    path,
                )
    raise KeyError(f"paired sample_id is outside all partitions: {sample_id}")


def main():
    args = parse_args()
    gt_root = Path(args.gt_root).resolve()
    paired_root = Path(args.paired_root).resolve()
    metadata = json.loads((gt_root / "metadata.json").read_text())
    identity = metadata["dataset_identity"]
    if identity.get("blend_mode") != "exhaustive":
        raise RuntimeError(f"expected exhaustive GT identity, got {identity.get('blend_mode')}")
    prefixes = [str(Path(x["prefix"]).resolve()) for x in identity["shards"]]

    # Transformer Engine queries CUDA capability at import time even though
    # this validation only builds CPU/mmap datasets.
    torch.cuda.current_device = lambda: 0
    torch.cuda.get_device_properties = lambda _=None: types.SimpleNamespace(major=8, minor=0)
    if not torch.distributed.is_initialized():
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        rendezvous = tempfile.NamedTemporaryFile(prefix="gt-align-", delete=False)
        rendezvous.close()
        torch.distributed.init_process_group(
            "gloo", rank=0, world_size=1, init_method=f"file://{rendezvous.name}"
        )

    from megatron.core.datasets.blended_megatron_dataset_builder import (
        BlendedMegatronDatasetBuilder,
    )
    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
    from megatron.training.tokenizer.tokenizer import _HuggingFaceTokenizer

    tokenizer = _HuggingFaceTokenizer(args.tokenizer_model)
    config = GPTDatasetConfig(
        random_seed=int(identity["seed"]),
        sequence_length=int(metadata["sequence_length"]),
        blend=(prefixes, None),
        blend_per_split=None,
        split=identity["split"],
        num_dataset_builder_threads=8,
        path_to_cache=str(Path(args.data_cache).resolve()),
        mmap_bin_files=True,
        tokenizer=tokenizer,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=False,
    )
    total_samples = int(metadata["total_samples"])
    dataset, _, _ = BlendedMegatronDatasetBuilder(
        GPTDataset, (total_samples, 0, 0), lambda: True, config
    ).build()
    if len(dataset) != total_samples:
        raise RuntimeError(f"dataset length mismatch: {len(dataset)} != {total_samples}")

    occurrence_ids = np.load(gt_root / "replay_occurrence_sample_ids.npy", mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    candidates = np.unique(
        np.concatenate(
            [
                np.asarray([0, total_samples - 1], dtype=np.int64),
                np.asarray(occurrence_ids[: min(8, occurrence_ids.size)], dtype=np.int64),
                np.asarray(
                    occurrence_ids[
                        rng.integers(0, occurrence_ids.size, size=max(0, args.checks - 10))
                    ],
                    dtype=np.int64,
                ),
            ]
        )
    )
    checked = []
    for sample_id in candidates:
        item = dataset[int(sample_id)]
        expected_tokens, expected_labels, source = paired_record(paired_root, int(sample_id))
        actual_tokens = np.asarray(item["tokens"], dtype=np.int64)
        if not np.array_equal(actual_tokens, expected_tokens):
            mismatch = int(np.flatnonzero(actual_tokens != expected_tokens)[0])
            raise RuntimeError(
                f"token mismatch sample={sample_id} position={mismatch}: "
                f"runtime={actual_tokens[mismatch]} paired={expected_tokens[mismatch]} source={source}"
            )
        actual_labels = np.asarray(item["labels"], dtype=np.int64)
        if not np.array_equal(actual_labels, expected_labels):
            mismatch = int(np.flatnonzero(actual_labels != expected_labels)[0])
            raise RuntimeError(
                f"label mismatch sample={sample_id} position={mismatch}: "
                f"runtime={actual_labels[mismatch]} paired={expected_labels[mismatch]} source={source}"
            )
        checked.append(int(sample_id))
    print(
        json.dumps(
            {
                "status": "PASS",
                "blend_mode": "exhaustive",
                "total_samples": total_samples,
                "checks": len(checked),
                "checked_sample_ids": checked,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
