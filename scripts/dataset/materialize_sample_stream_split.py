#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEGATRON_ROOT = PROJECT_ROOT / "Megatron-LM"
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
from megatron.core.datasets.utils import Split
from megatron.training.tokenizer.tokenizer import _HuggingFaceTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Materialize the GPT train sample stream into physical indexed datasets."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing tokenized shard_*.bin/.idx files.")
    parser.add_argument("--output-seen-dir", required=True, help="Output directory for the first split.")
    parser.add_argument("--output-holdout-dir", required=True, help="Output directory for the second split.")
    parser.add_argument("--tokenizer-model", default="EleutherAI/pythia-12b")
    parser.add_argument("--dataset-split", default="95,5,0")
    parser.add_argument(
        "--dataset-split-name",
        default="train",
        choices=("train", "valid", "test"),
    )
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--seen-samples", type=int, default=4147200)
    parser.add_argument("--holdout-samples", type=int, default=170496)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional directory to cache GPTDataset document/sample/shuffle indices.",
    )
    parser.add_argument(
        "--output-prefix",
        default="stream_text_document",
        help="Prefix name for the generated .bin/.idx inside each output dir.",
    )
    return parser.parse_args()


def list_data_prefixes(input_dir: Path) -> list[str]:
    prefixes = sorted(str(path.with_suffix("")) for path in input_dir.rglob("*.bin"))
    if not prefixes:
        raise RuntimeError(f"No .bin shards found under {input_dir}")
    return prefixes


def build_dataset(
    input_dir: Path,
    tokenizer_model: str,
    dataset_split: str,
    dataset_split_name: str,
    sequence_length: int,
    random_seed: int,
    total_samples: int,
    cache_dir: str | None,
):
    tokenizer = _HuggingFaceTokenizer(tokenizer_model)
    prefixes = list_data_prefixes(input_dir)
    split_name_to_index = {"train": 0, "valid": 1, "test": 2}
    split_idx = split_name_to_index[dataset_split_name]
    sizes = [0, 0, 0]
    sizes[split_idx] = total_samples

    config = GPTDatasetConfig(
        random_seed=random_seed,
        sequence_length=sequence_length,
        blend=(prefixes, [1.0] * len(prefixes)),
        blend_per_split=None,
        split=dataset_split,
        num_dataset_builder_threads=1,
        path_to_cache=cache_dir,
        mmap_bin_files=True,
        tokenizer=tokenizer,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=True,
        s3_cache_path=None,
    )

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset,
        sizes,
        lambda: True,
        config,
    ).build()
    dataset = (train_ds, valid_ds, test_ds)[split_idx]
    if dataset is None:
        raise RuntimeError(f"Failed to build {dataset_split_name} dataset")
    return dataset, prefixes


def ensure_distributed_initialized() -> Path | None:
    if not torch.distributed.is_available() or torch.distributed.is_initialized():
        return None

    rendezvous_dir = Path(tempfile.mkdtemp(prefix="materialize-split-dist-"))
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_dir / 'init'}",
        rank=0,
        world_size=1,
    )
    return rendezvous_dir


def materialize_range(dataset, start: int, count: int, output_prefix: Path):
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    builder = IndexedDatasetBuilder(str(output_prefix.with_suffix(".bin")), multimodal=False)
    total_tokens = 0

    progress = tqdm(range(start, start + count), desc=output_prefix.parent.name, dynamic_ncols=True)
    for sample_idx in progress:
        sample_tokens, _document_ids = query_sample_text(dataset, sample_idx)
        tensor = torch.from_numpy(sample_tokens.astype("int32", copy=False))
        builder.add_document(tensor, [int(tensor.numel())])
        total_tokens += int(tensor.numel())
        if (sample_idx - start + 1) % 1024 == 0:
            progress.set_postfix(samples=sample_idx - start + 1, tokens=total_tokens)

    builder.finalize(str(output_prefix.with_suffix(".idx")))
    return {
        "samples": count,
        "tokens": total_tokens,
        "prefix": output_prefix.name,
    }


def query_sample_text(dataset, sample_idx: int):
    if isinstance(dataset, GPTDataset):
        return dataset._query_document_sample_shuffle_indices(sample_idx)

    if isinstance(dataset, BlendedDataset):
        dataset_id = int(dataset.dataset_index[sample_idx])
        dataset_sample_id = int(dataset.dataset_sample_index[sample_idx])
        return dataset.datasets[dataset_id]._query_document_sample_shuffle_indices(dataset_sample_id)

    raise TypeError(f"Unsupported dataset type for materialization: {type(dataset).__name__}")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_seen_dir = Path(args.output_seen_dir)
    output_holdout_dir = Path(args.output_holdout_dir)
    rendezvous_dir = ensure_distributed_initialized()

    total_samples = args.seen_samples + args.holdout_samples
    dataset, prefixes = build_dataset(
        input_dir=input_dir,
        tokenizer_model=args.tokenizer_model,
        dataset_split=args.dataset_split,
        dataset_split_name=args.dataset_split_name,
        sequence_length=args.sequence_length,
        random_seed=args.random_seed,
        total_samples=total_samples,
        cache_dir=args.cache_dir,
    )

    seen_stats = materialize_range(dataset, 0, args.seen_samples, output_seen_dir / args.output_prefix)
    holdout_stats = materialize_range(
        dataset,
        args.seen_samples,
        args.holdout_samples,
        output_holdout_dir / args.output_prefix,
    )

    metadata = {
        "input_dir": str(input_dir),
        "dataset_split": args.dataset_split,
        "dataset_split_name": args.dataset_split_name,
        "random_seed": args.random_seed,
        "sequence_length": args.sequence_length,
        "tokenizer_model": args.tokenizer_model,
        "source_prefix_count": len(prefixes),
        "source_prefixes": prefixes,
        "seen_samples": args.seen_samples,
        "holdout_samples": args.holdout_samples,
        "seen": seen_stats,
        "holdout": holdout_stats,
    }

    for output_dir in (output_seen_dir, output_holdout_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    if rendezvous_dir is not None:
        try:
            rendezvous_dir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
