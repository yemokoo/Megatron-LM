#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEGATRON_ROOT = PROJECT_ROOT / "Megatron-LM"
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP
from megatron.core.transformer.moe.router import Router
from megatron.core.transformer.qv_lora_attention import QVLoraExpertRouter
from megatron.training import get_args, get_model, get_tokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.legacy.data.data_samplers import build_pretraining_data_loader

from pretrain_gpt import get_batch, is_dataset_built_on_rank, model_provider


def add_args(parser):
    group = parser.add_argument_group(title="expert-output-similarity")
    group.add_argument("--output-dir", type=str, required=True)
    group.add_argument("--compare-label", type=str, required=True)
    group.add_argument("--model-kind", type=str, required=True, choices=("ffn", "lora"))
    group.add_argument("--source-num-experts", type=int, default=4)
    group.add_argument("--dataset-split", type=str, default="100,0,0")
    group.add_argument(
        "--dataset-split-name",
        type=str,
        default="train",
        choices=("train", "valid", "test"),
    )
    group.add_argument("--consumed-samples", type=int, default=0)
    group.add_argument("--max-batches", type=int, default=2)
    group.add_argument("--max-tokens-per-layer", type=int, default=2048)
    group.add_argument("--plot-layers", type=str, default="")
    group.add_argument("--save-hidden-cache", action="store_true")
    group.add_argument("--hidden-cache-filename", type=str, default="hidden_states.pt")
    group.add_argument("--save-raw-expert-outputs", action="store_true")
    group.add_argument(
        "--raw-output-dtype",
        type=str,
        default="bf16",
        choices=("fp32", "fp16", "bf16"),
    )
    group.add_argument("--heatmap-vmin", type=float, default=-0.25)
    group.add_argument("--heatmap-vmax", type=float, default=0.75)
    return parser


def build_eval_dataloader():
    args = get_args()
    split_name_to_index = {"train": 0, "valid": 1, "test": 2}
    split_idx = split_name_to_index[args.dataset_split_name]
    requested_samples = (
        args.consumed_samples + args.max_batches * args.micro_batch_size * args.data_parallel_size
    )
    config = GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
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


def store_hidden(hidden_store, layer_number, hidden_states, max_tokens):
    layer_key = f"layer_{int(layer_number):02d}"
    hidden_flat = hidden_states.detach().reshape(-1, hidden_states.shape[-1])
    remaining = max_tokens - hidden_store.get(layer_key, torch.empty(0)).shape[0]
    if remaining <= 0:
        return
    chunk = hidden_flat[:remaining].detach().cpu()
    if layer_key not in hidden_store:
        hidden_store[layer_key] = chunk
    else:
        hidden_store[layer_key] = torch.cat([hidden_store[layer_key], chunk], dim=0)


def infer_layer_key(name, module):
    layer_number = getattr(module, "layer_number", None)
    if layer_number is not None:
        return f"layer_{int(layer_number):02d}"
    parts = name.split(".")
    if "layers" in parts:
        return f"layer_{int(parts[parts.index('layers') + 1]) + 1:02d}"
    return None


def install_hidden_collection_hooks(model, model_kind, hidden_store, max_tokens):
    hooks = []
    if model_kind == "ffn":
        for module in model.modules():
            if isinstance(module, Router):
                hooks.append(
                    module.register_forward_pre_hook(
                        lambda mod, inputs, layer=module.layer_number: store_hidden(
                            hidden_store, layer, inputs[0], max_tokens
                        )
                    )
                )
    else:
        for name, module in model.named_modules():
            if not isinstance(module, QVLoraExpertRouter):
                continue
            layer_key = infer_layer_key(name, module)
            if layer_key is None:
                continue
            hooks.append(
                module.register_forward_pre_hook(
                    lambda mod, inputs, layer=layer_key: store_hidden(
                        hidden_store, int(layer.split("_")[-1]), inputs[0], max_tokens
                    )
                )
            )
    return hooks


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


def collect_hidden_states(model):
    args = get_args()
    hidden_store = {}
    hooks = install_hidden_collection_hooks(model, args.model_kind, hidden_store, args.max_tokens_per_layer)

    dataloader = build_eval_dataloader()
    iterator = iter(dataloader)
    progress = tqdm(
        range(args.max_batches),
        desc=f"{args.compare_label}-hidden",
        dynamic_ncols=True,
        mininterval=5.0,
        disable=torch.distributed.get_rank() != 0,
    )
    with torch.no_grad():
        for _ in progress:
            tokens, _labels, _loss_mask, attention_mask, position_ids = get_batch(iterator)
            _ = model(tokens, position_ids, attention_mask, labels=None, runtime_gather_output=False)

    remove_hooks(hooks)
    return hidden_store


def mean_tokenwise_cosine_similarity_matrix(outputs: torch.Tensor):
    if outputs.dim() != 3:
        raise ValueError(f"Expected [num_experts, num_tokens, output_dim], got {tuple(outputs.shape)}")
    normalized = F.normalize(outputs, dim=-1)
    per_token = torch.einsum("etd,ftd->eft", normalized, normalized)
    return per_token.mean(dim=-1)


def aggregate_expert_outputs(outputs: torch.Tensor):
    if outputs.dim() != 3:
        raise ValueError(f"Expected [num_experts, num_tokens, output_dim], got {tuple(outputs.shape)}")
    return outputs.sum(dim=1)


def project_expert_vectors_2d(vectors: torch.Tensor):
    vectors = vectors.float()
    if vectors.shape[0] == 0:
        return vectors.new_zeros((0, 2))
    if vectors.shape[0] == 1:
        return vectors.new_zeros((1, 2))

    centered = vectors - vectors.mean(dim=0, keepdim=True)
    q = min(2, centered.shape[0], centered.shape[1])
    _u, _s, v = torch.pca_lowrank(centered, q=q, center=False)
    coords = centered @ v[:, :q]
    if q == 1:
        coords = torch.cat([coords, torch.zeros_like(coords)], dim=1)
    return coords[:, :2].cpu()


def cast_storage_dtype(tensor: torch.Tensor, dtype_name: str):
    if dtype_name == "fp32":
        return tensor.float().cpu()
    if dtype_name == "fp16":
        return tensor.to(torch.float16).cpu()
    if dtype_name == "bf16":
        return tensor.to(torch.bfloat16).cpu()
    raise ValueError(f"Unsupported storage dtype: {dtype_name}")


def save_raw_expert_outputs(output_dir: Path, layer_key: str, outputs: torch.Tensor, dtype_name: str):
    raw_dir = output_dir / "raw_expert_outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / f"{layer_key}_expert_outputs.pt"
    torch.save(cast_storage_dtype(outputs, dtype_name), file_path)
    return file_path.relative_to(output_dir).as_posix()


def masked_mean(matrix: torch.Tensor, row_slice: slice, col_slice: slice, diagonal: bool):
    block = matrix[row_slice, col_slice]
    if block.numel() == 0:
        return None
    if diagonal and block.shape[0] == block.shape[1]:
        mask = ~torch.eye(block.shape[0], dtype=torch.bool)
        values = block[mask]
    else:
        values = block.reshape(-1)
    if values.numel() == 0:
        return None
    return float(values.mean().item())


def summarize_similarity(matrix: torch.Tensor, old_expert_count: int):
    total = matrix.shape[0]
    old_end = min(old_expert_count, total)
    return {
        "within_old_mean": masked_mean(matrix, slice(0, old_end), slice(0, old_end), diagonal=True),
        "within_new_mean": masked_mean(matrix, slice(old_end, total), slice(old_end, total), diagonal=True),
        "cross_mean": masked_mean(matrix, slice(0, old_end), slice(old_end, total), diagonal=False),
    }


def collect_ffn_modules(model):
    modules = {}
    for module in model.modules():
        if isinstance(module, Router):
            modules.setdefault(f"layer_{int(module.layer_number):02d}", {})["router"] = module
        elif isinstance(module, GroupedMLP) or isinstance(module, SequentialMLP):
            pass
    for name, module in model.named_modules():
        if isinstance(module, GroupedMLP) or isinstance(module, SequentialMLP):
            parts = name.split(".")
            if "layers" in parts:
                layer_idx = int(parts[parts.index("layers") + 1]) + 1
                modules.setdefault(f"layer_{layer_idx:02d}", {})["experts"] = module
    return modules


def compute_ffn_outputs(experts_module, hidden_cpu):
    hidden = hidden_cpu.to(next(experts_module.parameters()).device).float()
    outputs = []
    if isinstance(experts_module, GroupedMLP):
        w1 = experts_module.weight1.detach().float().view(
            experts_module.num_local_experts, experts_module.config.hidden_size, -1
        )
        w2 = experts_module.weight2.detach().float().view(
            experts_module.num_local_experts, -1, experts_module.config.hidden_size
        )
        for expert_id in range(experts_module.num_local_experts):
            fc1 = hidden @ w1[expert_id]
            act = experts_module.activation_func(fc1)
            fc2 = act @ w2[expert_id]
            outputs.append(fc2.detach().cpu())
    elif isinstance(experts_module, SequentialMLP):
        for expert in experts_module.local_experts:
            out, _bias = expert(hidden)
            outputs.append(out.detach().cpu())
    else:
        raise TypeError(f"Unsupported FFN experts module: {type(experts_module)}")
    return torch.stack(outputs, dim=0)


def collect_lora_modules(model):
    modules = {}
    for name, module in model.named_modules():
        if not isinstance(module, QVLoraExpertRouter):
            continue
        layer_key = infer_layer_key(name, module)
        if layer_key is None:
            continue
        modules[layer_key] = module
    return modules


def compute_lora_outputs(router_module, hidden_cpu):
    hidden = hidden_cpu.to(router_module.router_weight.device).float()
    outputs = []
    for expert_id in range(router_module.num_experts):
        q_low_rank = hidden @ router_module.q_lora_a[expert_id].float()
        v_low_rank = hidden @ router_module.v_lora_a[expert_id].float()
        q_out = (q_low_rank @ router_module.q_lora_b[expert_id].float()) * router_module.scale
        v_out = (v_low_rank @ router_module.v_lora_b[expert_id].float()) * router_module.scale
        outputs.append(torch.cat([q_out, v_out], dim=-1).detach().cpu())
    return torch.stack(outputs, dim=0)


def save_layer_csvs(layer_results, output_dir: Path):
    csv_dir = output_dir / "layer_tables"
    csv_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for item in layer_results:
        layer = item["layer"]
        matrix = item["mean_tokenwise_cosine_similarity"]
        with (csv_dir / f"{layer}_mean_tokenwise_cosine_similarity.csv").open(
            "w", encoding="utf-8", newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["expert"] + [f"e{idx}" for idx in range(len(matrix))])
            for row_idx, row in enumerate(matrix):
                writer.writerow([f"e{row_idx}"] + [f"{float(value):.6f}" for value in row])

        summary = item.get("summary", {})
        summary_rows.append(
            {
                "layer": layer,
                "num_tokens": item["num_tokens"],
                "num_experts": item["num_experts"],
                "output_dim": item["output_dim"],
                "within_old_mean": summary.get("within_old_mean"),
                "within_new_mean": summary.get("within_new_mean"),
                "cross_mean": summary.get("cross_mean"),
            }
        )

    with (csv_dir / "layer_similarity_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "num_tokens",
                "num_experts",
                "output_dim",
                "within_old_mean",
                "within_new_mean",
                "cross_mean",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def save_projection_csvs(projection_cache, output_dir: Path):
    csv_dir = output_dir / "layer_tables"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for layer, payload in projection_cache.items():
        coords = payload["projection_2d"]
        with (csv_dir / f"{layer}_expert_projection_2d.csv").open(
            "w", encoding="utf-8", newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["expert", "x", "y"])
            for expert_idx, coord in enumerate(coords.tolist()):
                writer.writerow([f"e{expert_idx}", f"{coord[0]:.6f}", f"{coord[1]:.6f}"])


def save_heatmaps(layer_results, output_dir: Path, title_prefix: str, plot_layers, vmin: float, vmax: float):
    selected = [item for item in layer_results if item["layer"] in plot_layers]
    if not selected:
        return
    cols = min(3, len(selected))
    rows = math.ceil(len(selected) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.0 * rows))
    if hasattr(axes, "reshape"):
        flat_axes = list(axes.reshape(-1))
    else:
        flat_axes = [axes]
    for ax_idx, ax in enumerate(flat_axes):
        if ax_idx >= len(selected):
            ax.axis("off")
            continue
        item = selected[ax_idx]
        matrix = torch.tensor(item["mean_tokenwise_cosine_similarity"])
        im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(item["layer"])
        ax.set_xlabel("Expert")
        ax.set_ylabel("Expert")
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_yticks(range(matrix.shape[0]))
        ax.set_xticklabels([f"e{i}" for i in range(matrix.shape[1])], rotation=0)
        ax.set_yticklabels([f"e{i}" for i in range(matrix.shape[0])])
        midpoint = (vmin + vmax) / 2.0
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = float(matrix[row_idx, col_idx].item())
                text_color = "white" if value < midpoint else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title_prefix)
    fig.tight_layout()
    fig.savefig(output_dir / "expert_output_similarity_heatmaps.png", dpi=200)
    plt.close(fig)


def save_scatter_plots(layer_results, output_dir: Path, title_prefix: str, plot_layers, old_expert_count: int):
    selected = [item for item in layer_results if item["layer"] in plot_layers]
    if not selected:
        return
    cols = min(3, len(selected))
    rows = math.ceil(len(selected) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.0 * rows))
    if hasattr(axes, "reshape"):
        flat_axes = list(axes.reshape(-1))
    else:
        flat_axes = [axes]

    for ax_idx, ax in enumerate(flat_axes):
        if ax_idx >= len(selected):
            ax.axis("off")
            continue
        item = selected[ax_idx]
        coords = torch.tensor(item["expert_projection_2d"])
        num_experts = coords.shape[0]
        for expert_idx in range(num_experts):
            color = "#4c78a8" if expert_idx < old_expert_count else "#e45756"
            ax.scatter(coords[expert_idx, 0], coords[expert_idx, 1], s=80, color=color)
            ax.text(
                coords[expert_idx, 0],
                coords[expert_idx, 1],
                f"e{expert_idx}",
                fontsize=9,
                ha="left",
                va="bottom",
            )
        ax.set_title(item["layer"])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.axhline(0.0, color="lightgray", linewidth=0.8)
        ax.axvline(0.0, color="lightgray", linewidth=0.8)
    fig.suptitle(f"{title_prefix} (aggregate token-sum outputs)")
    fig.tight_layout()
    fig.savefig(output_dir / "expert_output_similarity_scatter.png", dpi=200)
    plt.close(fig)


def choose_plot_layers(layer_results, spec):
    if spec.strip():
        return [f"layer_{int(x):02d}" for x in spec.split(",") if x.strip()]
    layers = [item["layer"] for item in layer_results]
    return layers


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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_list = get_model(model_provider, wrap_with_ddp=False)
    _ = load_checkpoint(model_list, None, None)
    assert len(model_list) == 1, "Expected a single GPT model shard."
    model = model_list[0]
    model.eval()

    hidden_store = collect_hidden_states(model)
    if args.save_hidden_cache:
        torch.save(hidden_store, output_dir / args.hidden_cache_filename)
    layer_results = []
    aggregate_cache = {}
    projection_cache = {}

    if args.model_kind == "ffn":
        modules = collect_ffn_modules(model)
        for layer_key, hidden in sorted(hidden_store.items()):
            experts_module = modules.get(layer_key, {}).get("experts")
            if experts_module is None:
                continue
            outputs = compute_ffn_outputs(experts_module, hidden)
            raw_output_file = None
            if args.save_raw_expert_outputs:
                raw_output_file = save_raw_expert_outputs(
                    output_dir, layer_key, outputs, args.raw_output_dtype
                )
            cosine = mean_tokenwise_cosine_similarity_matrix(outputs).cpu()
            aggregate_vectors = aggregate_expert_outputs(outputs).cpu()
            projection_2d = project_expert_vectors_2d(aggregate_vectors)
            aggregate_cache[layer_key] = aggregate_vectors
            projection_cache[layer_key] = {
                "projection_2d": projection_2d,
            }
            layer_results.append(
                {
                    "layer": layer_key,
                    "num_tokens": int(hidden.shape[0]),
                    "num_experts": int(outputs.shape[0]),
                    "output_dim": int(outputs.shape[-1]),
                    "raw_expert_output_file": raw_output_file,
                    "mean_tokenwise_cosine_similarity": [[float(v) for v in row] for row in cosine.tolist()],
                    "expert_projection_2d": [[float(v) for v in row] for row in projection_2d.tolist()],
                    "summary": summarize_similarity(cosine, args.source_num_experts),
                }
            )
    else:
        modules = collect_lora_modules(model)
        for layer_key, hidden in sorted(hidden_store.items()):
            router_module = modules.get(layer_key)
            if router_module is None:
                continue
            outputs = compute_lora_outputs(router_module, hidden)
            raw_output_file = None
            if args.save_raw_expert_outputs:
                raw_output_file = save_raw_expert_outputs(
                    output_dir, layer_key, outputs, args.raw_output_dtype
                )
            cosine = mean_tokenwise_cosine_similarity_matrix(outputs).cpu()
            aggregate_vectors = aggregate_expert_outputs(outputs).cpu()
            projection_2d = project_expert_vectors_2d(aggregate_vectors)
            aggregate_cache[layer_key] = aggregate_vectors
            projection_cache[layer_key] = {
                "projection_2d": projection_2d,
            }
            layer_results.append(
                {
                    "layer": layer_key,
                    "num_tokens": int(hidden.shape[0]),
                    "num_experts": int(outputs.shape[0]),
                    "output_dim": int(outputs.shape[-1]),
                    "raw_expert_output_file": raw_output_file,
                    "mean_tokenwise_cosine_similarity": [[float(v) for v in row] for row in cosine.tolist()],
                    "expert_projection_2d": [[float(v) for v in row] for row in projection_2d.tolist()],
                    "summary": summarize_similarity(cosine, args.source_num_experts),
                }
            )

    plot_layers = choose_plot_layers(layer_results, args.plot_layers)
    save_layer_csvs(layer_results, output_dir)
    save_projection_csvs(projection_cache, output_dir)
    torch.save(aggregate_cache, output_dir / "expert_output_aggregate_vectors.pt")
    result = {
        "label": args.compare_label,
        "model_kind": args.model_kind,
        "load": args.load,
        "iteration": args.iteration,
        "source_num_experts": args.source_num_experts,
        "max_batches": args.max_batches,
        "micro_batch_size": args.micro_batch_size,
        "global_batch_size": args.global_batch_size,
        "data_parallel_size": args.data_parallel_size,
        "requested_samples": args.max_batches * args.micro_batch_size * args.data_parallel_size,
        "requested_token_count": (
            args.max_batches * args.micro_batch_size * args.data_parallel_size * args.seq_length
        ),
        "sample_seed": args.seed,
        "sampling_policy": (
            "deterministic random subset from the provided test dataset via GPTDataset train-index "
            "construction with a fixed seed"
        ),
        "max_tokens_per_layer": args.max_tokens_per_layer,
        "similarity_type": "mean_tokenwise_cosine",
        "same_hidden_input_per_expert": True,
        "saved_hidden_cache": bool(args.save_hidden_cache),
        "hidden_cache_filename": args.hidden_cache_filename if args.save_hidden_cache else None,
        "raw_expert_outputs_saved": bool(args.save_raw_expert_outputs),
        "raw_expert_output_dtype": args.raw_output_dtype if args.save_raw_expert_outputs else None,
        "raw_expert_output_dir": "raw_expert_outputs" if args.save_raw_expert_outputs else None,
        "recompute_note": "Reuse hidden_states.pt with the checkpoint to compute alternative metrics without rerunning dataset forward.",
        "aggregate_vector_method": "token_sum",
        "aggregate_vector_cache_filename": "expert_output_aggregate_vectors.pt",
        "heatmap_vmin": args.heatmap_vmin,
        "heatmap_vmax": args.heatmap_vmax,
        "layer_csv_dir": "layer_tables",
        "plot_layers": plot_layers,
        "layer_results": layer_results,
    }
    (output_dir / "expert_output_similarity.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    save_heatmaps(
        layer_results,
        output_dir,
        f"Expert output cosine similarity: {args.compare_label}",
        plot_layers,
        args.heatmap_vmin,
        args.heatmap_vmax,
    )
    save_scatter_plots(
        layer_results,
        output_dir,
        f"Expert output projection: {args.compare_label}",
        plot_layers,
        args.source_num_experts,
    )
    if torch.distributed.get_rank() == 0:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
