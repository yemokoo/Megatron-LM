#!/usr/bin/env python3
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
from megatron.core.transformer.shared_router_hybrid import SharedRouterHybridTransformerLayer
from megatron.training import get_args, get_model, get_tokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.legacy.data.data_samplers import build_pretraining_data_loader

from pretrain_gpt import get_batch, is_dataset_built_on_rank, model_provider


def add_args(parser):
    group = parser.add_argument_group(title="shared-router-pair-dump")
    group.add_argument("--output-pt", type=str, required=True)
    group.add_argument("--compare-label", type=str, required=True)
    group.add_argument("--dataset-split", type=str, default="100,0,0")
    group.add_argument(
        "--dataset-split-name",
        type=str,
        default="train",
        choices=("train", "valid", "test"),
    )
    group.add_argument("--consumed-samples", type=int, default=0)
    group.add_argument("--target-eval-tokens", type=int, default=1_000_000)
    group.add_argument("--max-batches", type=int, default=8)
    return parser


def normalize_blend_path_args(data_path_args):
    normalized = []
    for entry in data_path_args:
        candidate = Path(entry)
        if candidate.is_dir():
            prefixes = sorted(bin_path.with_suffix("") for bin_path in candidate.glob("*.bin"))
            if not prefixes:
                raise FileNotFoundError(f"No .bin files found under dataset directory {candidate}")
            for prefix in prefixes:
                normalized.extend(["1.0", str(prefix)])
        else:
            normalized.append(entry)
    return normalized


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


class RoutingPairCapture:
    def __init__(self, model):
        self.current_batch = {}
        self.layer_keys = []
        self.hooks = []
        for module in model.modules():
            if not isinstance(module, SharedRouterHybridTransformerLayer):
                continue
            if module.shared_expert_router is None:
                continue
            layer_key = f"layer_{int(module.layer_number):02d}"
            self.layer_keys.append(layer_key)
            self.hooks.append(
                module.shared_expert_router.register_forward_hook(self._make_hook(layer_key))
            )
        self.layer_keys = sorted(self.layer_keys)

    def _make_hook(self, layer_key):
        def _hook(_module, _inputs, outputs):
            _scores, routing_map = outputs
            topk = int(routing_map.sum(dim=-1).max().item())
            if topk != 2:
                raise ValueError(f"Expected top-2 routing, but found topk={topk} at {layer_key}")
            pair_idx = torch.topk(routing_map.to(torch.int32), k=topk, dim=-1).indices
            pair_idx = torch.sort(pair_idx, dim=-1).values.to(torch.int16).cpu()
            self.current_batch[layer_key] = pair_idx

        return _hook

    def reset(self):
        self.current_batch = {}

    def remove(self):
        for hook in self.hooks:
            hook.remove()


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
    args.data_path = normalize_blend_path_args(args.data_path)
    eval_iters = max(
        args.max_batches,
        math.ceil(args.target_eval_tokens / max(args.global_batch_size * args.seq_length, 1)),
    )

    model_list = get_model(model_provider, wrap_with_ddp=False)
    _ = load_checkpoint(model_list, None, None)
    assert len(model_list) == 1, "Expected a single GPT model shard."
    model = model_list[0]
    model.eval()

    capture = RoutingPairCapture(model)
    dataloader = build_eval_dataloader(args.data_path, eval_iters)
    iterator = iter(dataloader)
    pairs_by_layer = {layer_key: [] for layer_key in capture.layer_keys}

    progress = tqdm(
        range(eval_iters),
        desc=args.compare_label,
        dynamic_ncols=True,
        mininterval=5.0,
        disable=torch.distributed.get_rank() != 0,
    )
    with torch.no_grad():
        for _ in progress:
            capture.reset()
            tokens, _labels, _loss_mask, attention_mask, position_ids = get_batch(iterator)
            _ = model(
                tokens,
                position_ids,
                attention_mask,
                labels=None,
                runtime_gather_output=False,
            )
            for layer_key in capture.layer_keys:
                if layer_key not in capture.current_batch:
                    raise RuntimeError(f"Missing routing capture for {layer_key}")
                pairs_by_layer[layer_key].append(capture.current_batch[layer_key])

    capture.remove()

    serialized_pairs = {
        layer_key: torch.cat(chunks, dim=0) for layer_key, chunks in pairs_by_layer.items()
    }
    payload = {
        "compare_label": args.compare_label,
        "load": args.load,
        "iteration": args.iteration,
        "eval_iters": eval_iters,
        "seq_length": args.seq_length,
        "micro_batch_size": args.micro_batch_size,
        "global_batch_size": args.global_batch_size,
        "data_path": args.data_path,
        "dataset_split": args.dataset_split,
        "dataset_split_name": args.dataset_split_name,
        "pairs_by_layer": serialized_pairs,
        "token_count_by_layer": {
            layer_key: int(tensor.shape[0]) for layer_key, tensor in serialized_pairs.items()
        },
    }

    output_path = Path(args.output_pt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    if torch.distributed.get_rank() == 0:
        print(output_path)


if __name__ == "__main__":
    main()
