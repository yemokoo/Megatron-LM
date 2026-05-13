#!/usr/bin/env python3
"""Dump shared-router input hidden states for a fixed evaluation batch.

The router-memory KL experiment distills old-router outputs on hidden states
captured immediately before each shared router. This script captures the same
hidden tensors so we can visualize how a fixed Wiki batch drifts across
continual-learning checkpoints.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEGATRON_ROOT = PROJECT_ROOT / "Megatron-LM"
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.transformer.shared_router_hybrid import capture_shared_router_inputs
from megatron.training import get_args, get_model, get_tokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.legacy.data.data_samplers import build_pretraining_data_loader

from pretrain_gpt import get_batch, is_dataset_built_on_rank, model_provider


def add_args(parser):
    group = parser.add_argument_group(title="shared-router-hidden-drift")
    group.add_argument("--output-pt", type=str, required=True)
    group.add_argument("--compare-label", type=str, required=True)
    group.add_argument("--checkpoint-step-label", type=int, required=True)
    group.add_argument("--dataset-split", type=str, default="100,0,0")
    group.add_argument(
        "--dataset-split-name",
        type=str,
        default="train",
        choices=("train", "valid", "test"),
    )
    group.add_argument("--consumed-samples", type=int, default=0)
    group.add_argument("--max-batches", type=int, default=2)
    group.add_argument("--max-tokens-per-layer", type=int, default=4096)
    return parser


def build_eval_dataloader(data_path, num_batches):
    args = get_args()
    split_name_to_index = {"train": 0, "valid": 1, "test": 2}
    split_idx = split_name_to_index[args.dataset_split_name]
    requested_samples = args.consumed_samples + num_batches * args.global_batch_size
    config = GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(data_path),
        blend_per_split=None,
        split=args.dataset_split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=get_tokenizer(),
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        s3_cache_path=args.s3_cache_path,
    )

    dataset_type = MockGPTDataset if args.mock_data else GPTDataset
    sizes = [0, 0, 0]
    sizes[split_idx] = requested_samples
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        sizes,
        is_dataset_built_on_rank,
        config,
    ).build()
    selected_ds = (train_ds, valid_ds, test_ds)[split_idx]
    return build_pretraining_data_loader(selected_ds, args.consumed_samples)


def append_hidden(store, layer_number, hidden_states, max_tokens):
    layer_key = f"layer_{int(layer_number):02d}"
    hidden_flat = hidden_states.detach().reshape(-1, hidden_states.shape[-1])
    current = store.get(layer_key)
    current_count = 0 if current is None else current.shape[0]
    remaining = max_tokens - current_count
    if remaining <= 0:
        return
    chunk = hidden_flat[:remaining].float().cpu()
    if current is None:
        store[layer_key] = chunk
    else:
        store[layer_key] = torch.cat([current, chunk], dim=0)


def main():
    initialize_megatron(
        extra_args_provider=add_args,
        args_defaults={
            "no_load_rng": True,
            "no_load_optim": True,
            "exit_on_missing_checkpoint": True,
            "use_checkpoint_args": True,
        },
    )
    args = get_args()
    eval_iters = max(1, int(args.max_batches))

    model_list = get_model(model_provider, wrap_with_ddp=False)
    iteration, _ = load_checkpoint(model_list, None, None)
    assert len(model_list) == 1, "Expected a single GPT model shard."
    model = model_list[0]
    model.eval()

    dataloader = build_eval_dataloader(args.data_path, eval_iters)
    iterator = iter(dataloader)
    hidden_by_layer = {}

    progress = tqdm(
        range(eval_iters),
        desc=args.compare_label,
        dynamic_ncols=True,
        mininterval=5.0,
        disable=torch.distributed.get_rank() != 0,
    )
    with torch.no_grad():
        for _ in progress:
            tokens, _labels, _loss_mask, attention_mask, position_ids = get_batch(iterator)
            with capture_shared_router_inputs() as captured:
                _ = model(
                    tokens,
                    position_ids,
                    attention_mask,
                    labels=None,
                    runtime_gather_output=False,
                )
            for layer_number, hidden_states in captured:
                append_hidden(
                    hidden_by_layer,
                    layer_number,
                    hidden_states,
                    args.max_tokens_per_layer,
                )

    payload = {
        "compare_label": args.compare_label,
        "checkpoint_step_label": int(args.checkpoint_step_label),
        "loaded_iteration": int(iteration),
        "load": args.load,
        "data_path": args.data_path,
        "dataset_split": args.dataset_split,
        "dataset_split_name": args.dataset_split_name,
        "consumed_samples": int(args.consumed_samples),
        "max_batches": int(args.max_batches),
        "max_tokens_per_layer": int(args.max_tokens_per_layer),
        "seq_length": int(args.seq_length),
        "micro_batch_size": int(args.micro_batch_size),
        "global_batch_size": int(args.global_batch_size),
        "hidden_by_layer": hidden_by_layer,
        "token_count_by_layer": {
            layer_key: int(tensor.shape[0]) for layer_key, tensor in hidden_by_layer.items()
        },
        "hidden_size": {
            layer_key: int(tensor.shape[-1]) for layer_key, tensor in hidden_by_layer.items()
        },
        "metric_note": (
            "Hidden states are captured immediately before each shared router, "
            "using the same capture hook as router-memory KL."
        ),
    }

    output_path = Path(args.output_pt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    if torch.distributed.get_rank() == 0:
        summary = {k: v for k, v in payload.items() if k != "hidden_by_layer"}
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
