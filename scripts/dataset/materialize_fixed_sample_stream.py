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

from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
from megatron.training.tokenizer.tokenizer import _HuggingFaceTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Materialize a deterministic fixed GPT sample stream into one indexed dataset."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing source .bin/.idx files.")
    parser.add_argument("--output-dir", required=True, help="Output directory for the fixed sample stream.")
    parser.add_argument("--samples", type=int, required=True, help="Number of GPT samples to materialize.")
    parser.add_argument("--tokenizer-model", default="EleutherAI/pythia-12b")
    parser.add_argument("--dataset-split", default="100,0,0")
    parser.add_argument("--dataset-split-name", default="train", choices=("train", "valid", "test"))
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-prefix", default="train_text_document")
    parser.add_argument("--purpose", default="fixed_router_memory")
    parser.add_argument("--metadata-name", default="router_memory_metadata.json")
    return parser.parse_args()


def list_data_prefixes(input_dir: Path) -> list[str]:
    prefixes = sorted(str(path.with_suffix("")) for path in input_dir.rglob("*.bin"))
    if not prefixes:
        raise RuntimeError(f"No .bin shards found under {input_dir}")
    return prefixes


def ensure_distributed_initialized() -> Path | None:
    if not torch.distributed.is_available() or torch.distributed.is_initialized():
        return None

    rendezvous_dir = Path(tempfile.mkdtemp(prefix="fixed-sample-stream-dist-"))
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_dir / 'init'}",
        rank=0,
        world_size=1,
    )
    return rendezvous_dir


def build_dataset(args):
    tokenizer = _HuggingFaceTokenizer(args.tokenizer_model)
    prefixes = list_data_prefixes(Path(args.input_dir))
    split_name_to_index = {"train": 0, "valid": 1, "test": 2}
    split_idx = split_name_to_index[args.dataset_split_name]
    sizes = [0, 0, 0]
    sizes[split_idx] = args.samples

    config = GPTDatasetConfig(
        random_seed=args.random_seed,
        sequence_length=args.sequence_length,
        blend=(prefixes, [1.0] * len(prefixes)),
        blend_per_split=None,
        split=args.dataset_split,
        num_dataset_builder_threads=1,
        path_to_cache=args.cache_dir,
        mmap_bin_files=True,
        tokenizer=tokenizer,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=True,
        s3_cache_path=None,
    )

    datasets = BlendedMegatronDatasetBuilder(GPTDataset, sizes, lambda: True, config).build()
    dataset = datasets[split_idx]
    if dataset is None:
        raise RuntimeError(f"Failed to build {args.dataset_split_name} dataset")
    return dataset, prefixes


def query_sample_text(dataset, sample_idx: int):
    if isinstance(dataset, GPTDataset):
        return dataset._query_document_sample_shuffle_indices(sample_idx)

    if isinstance(dataset, BlendedDataset):
        dataset_id = int(dataset.dataset_index[sample_idx])
        dataset_sample_id = int(dataset.dataset_sample_index[sample_idx])
        return dataset.datasets[dataset_id]._query_document_sample_shuffle_indices(dataset_sample_id)

    raise TypeError(f"Unsupported dataset type for materialization: {type(dataset).__name__}")


def materialize(dataset, output_prefix: Path, samples: int):
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    builder = IndexedDatasetBuilder(str(output_prefix.with_suffix(".bin")), multimodal=False)
    total_tokens = 0

    progress = tqdm(range(samples), desc=output_prefix.parent.name, dynamic_ncols=True)
    for sample_idx in progress:
        sample_tokens, _document_ids = query_sample_text(dataset, sample_idx)
        tensor = torch.from_numpy(sample_tokens.astype("int32", copy=False))
        builder.add_document(tensor, [int(tensor.numel())])
        total_tokens += int(tensor.numel())
        if (sample_idx + 1) % 1024 == 0:
            progress.set_postfix(samples=sample_idx + 1, tokens=total_tokens)

    builder.finalize(str(output_prefix.with_suffix(".idx")))
    return {"samples": samples, "tokens": total_tokens, "prefix": output_prefix.name}


def main():
    args = parse_args()
    rendezvous_dir = ensure_distributed_initialized()
    output_dir = Path(args.output_dir)
    output_prefix = output_dir / args.output_prefix

    dataset, prefixes = build_dataset(args)
    stats = materialize(dataset, output_prefix, args.samples)
    metadata = {
        "purpose": args.purpose,
        "input_dir": str(args.input_dir),
        "dataset_split": args.dataset_split,
        "dataset_split_name": args.dataset_split_name,
        "random_seed": args.random_seed,
        "sequence_length": args.sequence_length,
        "tokenizer_model": args.tokenizer_model,
        "source_prefix_count": len(prefixes),
        "source_prefixes": prefixes,
        "memory": stats,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / args.metadata_name).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
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
