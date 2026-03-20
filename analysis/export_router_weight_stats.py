#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEGATRON_ROOT = PROJECT_ROOT / "Megatron-LM"
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))

from megatron.core.transformer.moe.router import Router
from megatron.training import get_args, get_model
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron

from pretrain_gpt import model_provider


def add_args(parser):
    group = parser.add_argument_group(title="router-weight-stats")
    group.add_argument("--output-json", required=True)
    group.add_argument("--model-label", required=True)
    group.add_argument("--old-expert-count", type=int, default=4)
    return parser


def tensor_stats(weight: torch.Tensor):
    weight_fp32 = weight.detach().float().cpu()
    return {
        "mean": float(weight_fp32.mean().item()),
        "abs_mean": float(weight_fp32.abs().mean().item()),
        "std": float(weight_fp32.std(unbiased=False).item()),
        "l2_norm": float(torch.linalg.vector_norm(weight_fp32).item()),
        "max_abs": float(weight_fp32.abs().max().item()),
    }


def collect_router_stats(model, old_expert_count: int):
    layer_records = []
    unnamed_layer_idx = 0

    for module in model.modules():
        if not isinstance(module, Router):
            continue

        layer_number = module.layer_number
        if layer_number is None:
            unnamed_layer_idx += 1
            layer_number = unnamed_layer_idx

        weight = module.weight.detach().float().cpu()
        expert_rows = []
        for expert_idx in range(weight.shape[0]):
            expert_rows.append(
                {
                    "expert_idx": expert_idx,
                    "group": "old" if expert_idx < old_expert_count else "new",
                    **tensor_stats(weight[expert_idx]),
                }
            )

        old_rows = weight[:old_expert_count]
        new_rows = weight[old_expert_count:]
        layer_records.append(
            {
                "layer_number": int(layer_number),
                "shape": list(weight.shape),
                "experts": expert_rows,
                "group_summary": {
                    "old": tensor_stats(old_rows),
                    "new": tensor_stats(new_rows) if new_rows.numel() > 0 else None,
                },
            }
        )

    layer_records.sort(key=lambda record: record["layer_number"])

    metric_names = ("mean", "abs_mean", "std", "l2_norm", "max_abs")
    overall = {
        "old": {metric: 0.0 for metric in metric_names},
        "new": {metric: 0.0 for metric in metric_names},
        "expert_means": [],
    }
    old_count = 0
    new_count = 0
    expert_accumulator = {}
    expert_counter = {}

    for layer in layer_records:
        for expert in layer["experts"]:
            idx = expert["expert_idx"]
            expert_accumulator.setdefault(idx, {metric: 0.0 for metric in metric_names})
            expert_counter[idx] = expert_counter.get(idx, 0) + 1
            for metric in metric_names:
                expert_accumulator[idx][metric] += expert[metric]

            target_group = "old" if idx < old_expert_count else "new"
            if target_group == "old":
                old_count += 1
            else:
                new_count += 1
            for metric in metric_names:
                overall[target_group][metric] += expert[metric]

    if old_count > 0:
        for metric in metric_names:
            overall["old"][metric] /= old_count
    if new_count > 0:
        for metric in metric_names:
            overall["new"][metric] /= new_count

    for expert_idx in sorted(expert_accumulator):
        overall["expert_means"].append(
            {
                "expert_idx": expert_idx,
                "group": "old" if expert_idx < old_expert_count else "new",
                **{
                    metric: expert_accumulator[expert_idx][metric] / expert_counter[expert_idx]
                    for metric in metric_names
                },
            }
        )

    return {
        "num_layers": len(layer_records),
        "old_expert_count": old_expert_count,
        "layers": layer_records,
        "overall": overall,
    }


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

    model_list = get_model(model_provider, wrap_with_ddp=False)
    _ = load_checkpoint(model_list, None, None)
    assert len(model_list) == 1, "Expected a single GPT model shard."
    model = model_list[0]
    stats = collect_router_stats(model, old_expert_count=args.old_expert_count)

    payload = {
        "model_label": args.model_label,
        "load": args.load,
        "stats": stats,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
