#!/usr/bin/env python3
"""Plot per-expert router softmax mass on a wiki-train miniset.

This intentionally measures the dense router softmax distribution before Top-K
selection, so it can be used to choose old/wiki experts to freeze later.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MEGATRON_ROOT = REPO_ROOT / "Megatron-LM"
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))

from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.enums import ModelType
from megatron.core.transformer.shared_router_hybrid import capture_shared_router_inputs
from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training.training import _collect_current_shared_routers, _router_logits, get_model

from pretrain_gpt import (
    _flatten_layer_hidden,
    build_pretraining_data_loader,
    get_batch,
    is_dataset_built_on_rank,
    model_provider,
)


def add_router_softmax_args(parser):
    group = parser.add_argument_group("router softmax importance analysis")
    group.add_argument(
        "--router-softmax-out-dir",
        type=str,
        required=True,
        help="Directory for CSV/JSON/plot outputs.",
    )
    group.add_argument(
        "--router-softmax-eval-iters",
        type=int,
        default=25,
        help="Number of dataloader iterations. Match probe eval iters by default.",
    )
    group.add_argument(
        "--router-softmax-data-path",
        nargs="+",
        default=None,
        help="Optional weighted indexed dataset prefix list. Defaults to --data-path.",
    )
    group.add_argument(
        "--router-softmax-layers",
        type=str,
        default="2,3,4,5,6,7,8,9",
        help="Comma-separated router layer numbers to report, or 'all'.",
    )
    group.add_argument(
        "--router-softmax-topn-highlight",
        type=int,
        default=4,
        help="Highlight this many experts in each bar plot.",
    )
    group.add_argument(
        "--router-softmax-mask-topk",
        type=int,
        default=4,
        help="Number of top experts per layer to save in the freeze-mask JSON.",
    )
    group.add_argument(
        "--router-softmax-cumulative-cutoff",
        type=float,
        default=0.0,
        help=(
            "If >0, highlight/save the smallest expert set whose sorted mean "
            "router softmax mass reaches this cutoff, e.g. 0.8 for 80%% mass."
        ),
    )
    group.add_argument(
        "--router-softmax-max-tokens",
        type=int,
        default=0,
        help=(
            "Optional approximate global cap on valid tokens to score. "
            "Use 1048576 for a 1M-token miniset. 0 means no cap."
        ),
    )
    return parser


def parse_layers(layer_spec: str, available_layers: Iterable[int]) -> list[int]:
    available = sorted(int(layer) for layer in available_layers)
    if layer_spec.lower() == "all":
        return available
    requested = sorted({int(x.strip()) for x in layer_spec.split(",") if x.strip()})
    missing = [layer for layer in requested if layer not in available]
    if missing:
        raise RuntimeError(f"Requested router layers not found: {missing}; available={available}")
    return requested


def build_train_miniset_dataloader():
    args = get_args()
    data_path = args.router_softmax_data_path or args.data_path
    if not data_path:
        raise RuntimeError("Provide --data-path or --router-softmax-data-path.")

    config = GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(data_path),
        blend_per_split=None,
        split="100,0,0",
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
    train_ds, _, _ = BlendedMegatronDatasetBuilder(
        dataset_type,
        (args.router_softmax_eval_iters * args.global_batch_size, 0, 0),
        is_dataset_built_on_rank,
        config,
    ).build()
    return build_pretraining_data_loader(train_ds, 0)


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    torch.distributed.all_reduce(tensor, group=mpu.get_data_parallel_group())
    return tensor


def collect_softmax_means(model):
    args = get_args()
    modules = model if isinstance(model, list) else [model]
    routers = _collect_current_shared_routers(modules)
    if not routers:
        raise RuntimeError("No shared-router modules found in the loaded model.")

    selected_layers = parse_layers(args.router_softmax_layers, routers.keys())
    num_experts = int(next(iter(routers.values())).weight.shape[0])
    device = torch.device("cuda", torch.cuda.current_device())
    sums = {
        layer: torch.zeros(num_experts, device=device, dtype=torch.float64)
        for layer in selected_layers
    }
    counts = {
        layer: torch.zeros(1, device=device, dtype=torch.float64)
        for layer in selected_layers
    }

    dataloader = build_train_miniset_dataloader()
    iterator = iter(dataloader)
    prior_states = [module.training for module in modules]
    for module in modules:
        module.eval()

    local_token_limit = 0
    if args.router_softmax_max_tokens and args.router_softmax_max_tokens > 0:
        dp_world_size = max(1, mpu.get_data_parallel_world_size())
        local_token_limit = math.ceil(args.router_softmax_max_tokens / dp_world_size)
    local_tokens = 0

    with torch.no_grad():
        for step in range(args.router_softmax_eval_iters):
            tokens, labels, loss_mask, attention_mask, position_ids = get_batch(iterator)
            flat_mask = loss_mask.reshape(-1).bool()
            candidate_indices = torch.nonzero(flat_mask, as_tuple=False).view(-1)
            if candidate_indices.numel() == 0:
                continue
            if local_token_limit > 0:
                remaining = local_token_limit - local_tokens
                if remaining <= 0:
                    break
                candidate_indices = candidate_indices[:remaining]

            with capture_shared_router_inputs() as captured:
                modules[0](tokens, position_ids, attention_mask, labels=labels)

            for layer_number, hidden_states in captured:
                layer = int(layer_number)
                if layer not in sums:
                    continue
                router = routers.get(layer)
                if router is None:
                    continue

                flat_hidden = _flatten_layer_hidden(hidden_states.detach(), labels)
                flat_hidden = flat_hidden.index_select(0, candidate_indices)
                logits = _router_logits(router, flat_hidden)
                probs = torch.softmax(logits.float(), dim=-1)
                sums[layer] += probs.double().sum(dim=0)
                counts[layer] += probs.shape[0]

            local_tokens += int(candidate_indices.numel())
            if torch.distributed.get_rank() == 0 and (step + 1) % max(1, args.log_interval) == 0:
                suffix = f", local_tokens={local_tokens}"
                if local_token_limit > 0:
                    suffix += f"/{local_token_limit}"
                print_rank_0(
                    f"[router-softmax] processed {step + 1}/{args.router_softmax_eval_iters}{suffix}"
                )
            if local_token_limit > 0 and local_tokens >= local_token_limit:
                break

    for module, was_training in zip(modules, prior_states):
        if was_training:
            module.train()

    result = {}
    for layer in selected_layers:
        reduce_tensor(sums[layer])
        reduce_tensor(counts[layer])
        result[layer] = (sums[layer] / counts[layer].clamp_min(1.0)).detach().cpu()
    return result


def ranked_rows(layer_means: dict[int, torch.Tensor]) -> tuple[list[dict], list[dict]]:
    rows = []
    for layer, values in layer_means.items():
        order = torch.argsort(values, descending=True).tolist()
        rank_by_idx = {expert_idx: rank + 1 for rank, expert_idx in enumerate(order)}
        for expert_idx, value in enumerate(values.tolist()):
            rows.append(
                {
                    "layer": layer,
                    "expert": expert_idx + 1,
                    "mean_softmax": float(value),
                    "rank_in_layer": rank_by_idx[expert_idx],
                }
            )

    stacked = torch.stack([values for _, values in sorted(layer_means.items())], dim=0)
    avg = stacked.mean(dim=0)
    order = torch.argsort(avg, descending=True).tolist()
    rank_by_idx = {expert_idx: rank + 1 for rank, expert_idx in enumerate(order)}
    avg_rows = [
        {
            "expert": expert_idx + 1,
            "mean_softmax": float(value),
            "rank": rank_by_idx[expert_idx],
        }
        for expert_idx, value in enumerate(avg.tolist())
    ]
    return rows, avg_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_topk_freeze_mask(layer_means: dict[int, torch.Tensor], topk: int) -> dict:
    num_experts = len(next(iter(layer_means.values())))
    topk = max(1, min(int(topk), num_experts))
    layers = {}
    for layer, values in sorted(layer_means.items()):
        order = torch.argsort(values, descending=True).tolist()
        layers[str(layer)] = [expert_idx + 1 for expert_idx in order[:topk]]
    return {
        "expert_index_base": 1,
        "selection": "topk_by_mean_router_softmax_pre_topk_per_layer",
        "topk": topk,
        "layers": layers,
    }


def build_cumulative_cutoff_selection(layer_means: dict[int, torch.Tensor], cutoff: float) -> tuple[dict, list[dict]]:
    cutoff = max(0.0, min(float(cutoff), 1.0))
    layers = {}
    layer_metadata = {}
    rows = []
    for layer, values in sorted(layer_means.items()):
        order = torch.argsort(values, descending=True).tolist()
        cumulative = 0.0
        selected = []
        for expert_idx in order:
            cumulative += float(values[expert_idx])
            selected.append(expert_idx)
            if cumulative >= cutoff:
                break

        selected_experts = [expert_idx + 1 for expert_idx in selected]
        layers[str(layer)] = selected_experts
        layer_metadata[str(layer)] = {
            "experts": selected_experts,
            "count": len(selected_experts),
            "mass": cumulative,
            "cutoff": cutoff,
        }
        for rank, expert_idx in enumerate(order, start=1):
            value = float(values[expert_idx])
            rows.append(
                {
                    "layer": layer,
                    "expert": expert_idx + 1,
                    "mean_softmax": value,
                    "rank_in_layer": rank,
                    "selected_for_cutoff": int(expert_idx in selected),
                    "selected_count": len(selected_experts),
                    "selected_mass": cumulative,
                    "cutoff": cutoff,
                }
            )

    return (
        {
            "expert_index_base": 1,
            "selection": "minimum_prefix_by_mean_router_softmax_cumulative_mass_pre_topk_per_layer",
            "cumulative_cutoff": cutoff,
            "layers": layers,
            "layer_metadata": layer_metadata,
        },
        rows,
    )


def plot_results(out_dir: Path, layer_means: dict[int, torch.Tensor], avg_rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    args = get_args()
    layers = sorted(layer_means)
    num_experts = len(next(iter(layer_means.values())))
    experts = list(range(1, num_experts + 1))
    highlight_n = max(1, min(int(args.router_softmax_topn_highlight), num_experts))
    cumulative_cutoff = max(0.0, min(float(args.router_softmax_cumulative_cutoff), 1.0))
    uniform = 1.0 / float(num_experts)

    fig, axes = plt.subplots(2, math.ceil(len(layers) / 2), figsize=(22, 9), sharey=True)
    axes = list(axes.reshape(-1))
    for ax, layer in zip(axes, layers):
        values = layer_means[layer].numpy()
        order = values.argsort()[::-1]
        if cumulative_cutoff > 0:
            cumulative = 0.0
            selected = []
            for expert_idx in order.tolist():
                cumulative += float(values[expert_idx])
                selected.append(expert_idx)
                if cumulative >= cumulative_cutoff:
                    break
            highlight = set(selected)
            title_suffix = f" | {len(selected)} exp, mass={cumulative:.3f}"
        else:
            highlight = set(order[:highlight_n].tolist())
            title_suffix = ""
        colors = ["#2563eb" if idx in highlight else "#cbd5e1" for idx in range(num_experts)]
        ax.bar(experts, values, color=colors, edgecolor="#0f172a", linewidth=0.6)
        ax.axhline(uniform, color="#ef4444", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_title(f"Layer {layer}{title_suffix}", fontsize=14, fontweight="bold")
        ax.set_xticks(experts)
        ax.set_ylim(0, max(values.max() * 1.18, uniform * 1.35))
        ax.grid(axis="y", alpha=0.22)
        for expert_idx in order:
            if expert_idx not in highlight:
                continue
            ax.text(
                expert_idx + 1,
                values[expert_idx],
                f"{values[expert_idx]:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#1e3a8a",
                fontweight="bold",
            )

    for ax in axes[len(layers):]:
        ax.axis("off")

    axes[0].set_ylabel("Mean router softmax probability", fontsize=12)
    fig.suptitle(
        "Wiki Train Miniset Router Softmax Importance by Layer (pre-TopK)",
        fontsize=22,
        fontweight="bold",
        y=0.99,
    )
    fig.legend(
        handles=[
            Patch(
                facecolor="#2563eb",
                edgecolor="#0f172a",
                label=(
                    f"Cumulative {cumulative_cutoff:.0%} mass experts"
                    if cumulative_cutoff > 0
                    else f"Top-{highlight_n} experts in layer"
                ),
            ),
            Patch(facecolor="#cbd5e1", edgecolor="#0f172a", label="Other experts"),
            Patch(facecolor="#ffffff", edgecolor="#ef4444", label=f"Uniform baseline = {uniform:.3f}"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        frameon=False,
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(out_dir / "router_softmax_by_layer.png", dpi=220)
    fig.savefig(out_dir / "router_softmax_by_layer.svg")
    plt.close(fig)

    avg_values = [row["mean_softmax"] for row in sorted(avg_rows, key=lambda row: row["expert"])]
    avg_order = sorted(avg_rows, key=lambda row: row["rank"])
    if cumulative_cutoff > 0:
        cumulative = 0.0
        top_experts = set()
        for row in avg_order:
            cumulative += float(row["mean_softmax"])
            top_experts.add(row["expert"])
            if cumulative >= cumulative_cutoff:
                break
        avg_label = f"Cumulative {cumulative_cutoff:.0%} layer-average experts ({len(top_experts)} exp)"
    else:
        top_experts = {row["expert"] for row in avg_order[:highlight_n]}
        avg_label = f"Top-{highlight_n} layer-average experts"
    colors = ["#16a34a" if expert in top_experts else "#d1d5db" for expert in experts]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(experts, avg_values, color=colors, edgecolor="#111827", linewidth=0.8)
    ax.axhline(uniform, color="#ef4444", linestyle="--", linewidth=1.4, label=f"Uniform baseline = {uniform:.3f}")
    ax.set_title("Layer-Average Wiki Router Softmax Importance (pre-TopK)", fontsize=20, fontweight="bold")
    ax.set_xlabel("Expert", fontsize=13)
    ax.set_ylabel("Mean router softmax probability", fontsize=13)
    ax.set_xticks(experts)
    ax.set_ylim(0, max(max(avg_values) * 1.22, uniform * 1.35))
    ax.grid(axis="y", alpha=0.25)
    for bar, value, expert in zip(bars, avg_values, experts):
        rank = next(row["rank"] for row in avg_rows if row["expert"] == expert)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"#{rank}\n{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold" if expert in top_experts else "normal",
            color="#14532d" if expert in top_experts else "#374151",
        )
    ax.legend(
        handles=[
            Patch(facecolor="#16a34a", edgecolor="#111827", label=avg_label),
            Patch(facecolor="#d1d5db", edgecolor="#111827", label="Other experts"),
            Patch(facecolor="#ffffff", edgecolor="#ef4444", label=f"Uniform baseline = {uniform:.3f}"),
        ],
        loc="upper right",
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "router_softmax_layer_average.png", dpi=220)
    fig.savefig(out_dir / "router_softmax_layer_average.svg")
    plt.close(fig)


def main() -> None:
    initialize_megatron(
        extra_args_provider=add_router_softmax_args,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
    args = get_args()

    model = get_model(model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)
    iteration, _ = load_checkpoint(model, None, None)
    print_rank_0(f"[router-softmax] loaded checkpoint iteration {iteration}")

    layer_means = collect_softmax_means(model)
    rows, avg_rows = ranked_rows(layer_means)

    out_dir = Path(args.router_softmax_out_dir)
    if torch.distributed.get_rank() == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(
            out_dir / "router_softmax_by_layer.csv",
            rows,
            ["layer", "expert", "mean_softmax", "rank_in_layer"],
        )
        write_csv(
            out_dir / "router_softmax_layer_average.csv",
            avg_rows,
            ["expert", "mean_softmax", "rank"],
        )
        metadata = {
            "load": args.load,
            "checkpoint_iteration": int(iteration),
            "data_path": args.router_softmax_data_path or args.data_path,
            "eval_iters": int(args.router_softmax_eval_iters),
            "max_tokens": int(args.router_softmax_max_tokens),
            "global_batch_size": int(args.global_batch_size),
            "sequence_length": int(args.seq_length),
            "layers": sorted(layer_means),
            "cumulative_cutoff": float(args.router_softmax_cumulative_cutoff),
            "measurement": "mean per-token router softmax probability before Top-K selection",
        }
        (out_dir / "router_softmax_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        mask = build_topk_freeze_mask(layer_means, args.router_softmax_mask_topk)
        mask["load"] = args.load
        mask["checkpoint_iteration"] = int(iteration)
        mask["max_tokens"] = int(args.router_softmax_max_tokens)
        mask["data_path"] = args.router_softmax_data_path or args.data_path
        (out_dir / f"router_softmax_top{mask['topk']}_freeze_mask.json").write_text(
            json.dumps(mask, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if args.router_softmax_cumulative_cutoff and args.router_softmax_cumulative_cutoff > 0:
            cutoff_mask, cutoff_rows = build_cumulative_cutoff_selection(
                layer_means,
                args.router_softmax_cumulative_cutoff,
            )
            cutoff_mask["load"] = args.load
            cutoff_mask["checkpoint_iteration"] = int(iteration)
            cutoff_mask["max_tokens"] = int(args.router_softmax_max_tokens)
            cutoff_mask["data_path"] = args.router_softmax_data_path or args.data_path
            cutoff_label = int(round(float(cutoff_mask["cumulative_cutoff"]) * 100))
            (out_dir / f"router_softmax_cumulative{cutoff_label}_selection.json").write_text(
                json.dumps(cutoff_mask, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            write_csv(
                out_dir / f"router_softmax_cumulative{cutoff_label}_by_layer.csv",
                cutoff_rows,
                [
                    "layer",
                    "expert",
                    "mean_softmax",
                    "rank_in_layer",
                    "selected_for_cutoff",
                    "selected_count",
                    "selected_mass",
                    "cutoff",
                ],
            )
        plot_results(out_dir, layer_means, avg_rows)
        print_rank_0(f"[router-softmax] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
