#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Recompute expert similarity from cached raw outputs.")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument(
        "--metric",
        type=str,
        default="linear_cka",
        choices=("linear_cka",),
    )
    parser.add_argument("--source-num-experts", type=int, default=4)
    parser.add_argument("--plot-layers", type=str, default="")
    parser.add_argument("--heatmap-vmin", type=float, default=0.0)
    parser.add_argument("--heatmap-vmax", type=float, default=1.0)
    parser.add_argument("--tsne-sample-per-expert", type=int, default=256)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-iterations", type=int, default=500)
    parser.add_argument("--tsne-learning-rate", type=float, default=200.0)
    parser.add_argument("--tsne-early-exaggeration", type=float, default=12.0)
    return parser.parse_args()


def discover_dataset_dirs(input_dir: Path):
    if (input_dir / "raw_expert_outputs").is_dir():
        return [input_dir]
    return sorted(
        path for path in input_dir.iterdir() if path.is_dir() and (path / "raw_expert_outputs").is_dir()
    )


def layer_sort_key(path: Path):
    stem = path.stem
    try:
        return int(stem.split("_")[1])
    except Exception:
        return stem


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    x = x.float()
    y = y.float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xty = x.transpose(0, 1) @ y
    hsic = (xty * xty).sum()
    x_norm = ((x.transpose(0, 1) @ x) ** 2).sum().sqrt()
    y_norm = ((y.transpose(0, 1) @ y) ** 2).sum().sqrt()
    denom = (x_norm * y_norm).clamp_min(eps)
    return float((hsic / denom).item())


def compute_linear_cka_matrix(outputs: torch.Tensor) -> torch.Tensor:
    num_experts = outputs.shape[0]
    matrix = torch.empty((num_experts, num_experts), dtype=torch.float32)
    for i in range(num_experts):
        for j in range(i, num_experts):
            value = linear_cka(outputs[i], outputs[j])
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def reduce_features_for_tsne(vectors: torch.Tensor, max_components: int = 50):
    vectors = vectors.float()
    if vectors.shape[0] <= 1:
        return vectors
    centered = vectors - vectors.mean(dim=0, keepdim=True)
    target_dims = min(max_components, centered.shape[0] - 1, centered.shape[1])
    if target_dims <= 0:
        return centered
    if target_dims >= centered.shape[1]:
        return centered
    _u, _s, v = torch.pca_lowrank(centered, q=target_dims, center=False)
    return centered @ v[:, :target_dims]


def _shannon_entropy_and_probs(dist_row: torch.Tensor, beta: float):
    probs = torch.exp(-dist_row * beta)
    prob_sum = probs.sum()
    if float(prob_sum.item()) <= 1e-12:
        normalized = torch.full_like(probs, 1.0 / max(probs.numel(), 1))
        return torch.tensor(0.0, dtype=dist_row.dtype), normalized
    probs = probs / prob_sum
    entropy = -torch.sum(probs * torch.log(probs.clamp_min(1e-12)))
    return entropy, probs


def compute_joint_probabilities(features: torch.Tensor, perplexity: float):
    num_points = features.shape[0]
    if num_points <= 1:
        return torch.zeros((num_points, num_points), dtype=torch.float32)

    sq_norms = (features**2).sum(dim=1, keepdim=True)
    distances = (sq_norms + sq_norms.transpose(0, 1) - 2.0 * (features @ features.transpose(0, 1))).clamp_min(0.0)
    conditional = torch.zeros((num_points, num_points), dtype=torch.float32)
    target_entropy = math.log(max(min(perplexity, num_points - 1), 1.0))

    for row_idx in range(num_points):
        row_dist = torch.cat([distances[row_idx, :row_idx], distances[row_idx, row_idx + 1 :]], dim=0).float()
        beta = 1.0
        beta_min = None
        beta_max = None
        probs = None
        for _ in range(50):
            entropy, probs = _shannon_entropy_and_probs(row_dist, beta)
            diff = float(entropy.item() - target_entropy)
            if abs(diff) < 1e-4:
                break
            if diff > 0.0:
                beta_min = beta
                beta = beta * 2.0 if beta_max is None else 0.5 * (beta + beta_max)
            else:
                beta_max = beta
                beta = beta / 2.0 if beta_min is None else 0.5 * (beta + beta_min)

        if probs is None:
            _, probs = _shannon_entropy_and_probs(row_dist, beta)

        full_row = torch.zeros(num_points, dtype=torch.float32)
        if row_idx > 0:
            full_row[:row_idx] = probs[:row_idx]
        if row_idx + 1 < num_points:
            full_row[row_idx + 1 :] = probs[row_idx:]
        conditional[row_idx] = full_row

    joint = (conditional + conditional.transpose(0, 1)) / (2.0 * num_points)
    return joint.clamp_min(1e-12)


def exact_tsne(
    features: torch.Tensor,
    perplexity: float,
    iterations: int,
    learning_rate: float,
    early_exaggeration: float,
    seed: int,
):
    num_points = features.shape[0]
    if num_points == 0:
        return torch.zeros((0, 2), dtype=torch.float32)
    if num_points == 1:
        return torch.zeros((1, 2), dtype=torch.float32)

    effective_perplexity = min(perplexity, max(1.0, float(num_points - 1)))
    reduced = reduce_features_for_tsne(features)
    joint = compute_joint_probabilities(reduced, effective_perplexity)
    exaggeration_steps = min(250, max(iterations // 2, 1))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    embedding = 1e-4 * torch.randn((num_points, 2), generator=generator, dtype=torch.float32)
    velocity = torch.zeros_like(embedding)

    for step in range(iterations):
        sq_norms = (embedding**2).sum(dim=1, keepdim=True)
        num = 1.0 / (1.0 + sq_norms + sq_norms.transpose(0, 1) - 2.0 * (embedding @ embedding.transpose(0, 1)))
        num.fill_diagonal_(0.0)
        q = num / num.sum().clamp_min(1e-12)
        p = joint * (early_exaggeration if step < exaggeration_steps else 1.0)

        coeff = (p - q) * num
        grad = 4.0 * (torch.diag(coeff.sum(dim=1)) - coeff) @ embedding
        momentum = 0.5 if step < exaggeration_steps else 0.8
        velocity = momentum * velocity - learning_rate * grad
        embedding = embedding + velocity
        embedding = embedding - embedding.mean(dim=0, keepdim=True)

    return embedding.cpu()


def sample_token_level_outputs(outputs: torch.Tensor, samples_per_expert: int, seed: int):
    num_experts, num_tokens, output_dim = outputs.shape
    sample_count = min(samples_per_expert, num_tokens)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    token_indices = torch.randperm(num_tokens, generator=generator)[:sample_count]
    sampled = outputs[:, token_indices, :]
    flat_outputs = sampled.reshape(num_experts * sample_count, output_dim).cpu()
    expert_indices = (
        torch.arange(num_experts, dtype=torch.long).unsqueeze(1).expand(num_experts, sample_count).reshape(-1)
    )
    repeated_token_indices = token_indices.unsqueeze(0).expand(num_experts, sample_count).reshape(-1)
    return {
        "flat_outputs": flat_outputs,
        "expert_indices": expert_indices,
        "token_indices": repeated_token_indices,
        "sample_count_per_expert": sample_count,
    }


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


def choose_plot_layers(layer_results, spec: str):
    if spec.strip():
        return [f"layer_{int(piece):02d}" for piece in spec.split(",") if piece.strip()]
    return [item["layer"] for item in layer_results]


def save_heatmaps(layer_results, output_dir: Path, title_prefix: str, plot_layers, vmin: float, vmax: float):
    selected = [item for item in layer_results if item["layer"] in plot_layers]
    if not selected:
        return
    cols = min(3, len(selected))
    rows = math.ceil(len(selected) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.0 * rows))
    flat_axes = list(axes.reshape(-1)) if hasattr(axes, "reshape") else [axes]
    for ax_idx, ax in enumerate(flat_axes):
        if ax_idx >= len(selected):
            ax.axis("off")
            continue
        item = selected[ax_idx]
        matrix = torch.tensor(item["matrix"])
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
    fig.savefig(output_dir / "expert_output_linear_cka_heatmaps.png", dpi=200)
    plt.close(fig)


def save_csvs(layer_results, output_dir: Path):
    csv_dir = output_dir / "layer_tables"
    csv_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for item in layer_results:
        layer = item["layer"]
        matrix = item["matrix"]
        with (csv_dir / f"{layer}_linear_cka.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["expert"] + [f"e{idx}" for idx in range(len(matrix))])
            for row_idx, row in enumerate(matrix):
                writer.writerow([f"e{row_idx}"] + [f"{float(value):.6f}" for value in row])
        summary_rows.append(
            {
                "layer": layer,
                "num_tokens": item["num_tokens"],
                "num_experts": item["num_experts"],
                "output_dim": item["output_dim"],
                "within_old_mean": item["summary"]["within_old_mean"],
                "within_new_mean": item["summary"]["within_new_mean"],
                "cross_mean": item["summary"]["cross_mean"],
            }
        )
    with (csv_dir / "layer_linear_cka_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
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


def save_token_tsne_csvs(token_tsne_cache, output_dir: Path, old_expert_count: int):
    csv_dir = output_dir / "layer_tables" / "token_tsne"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for layer, payload in token_tsne_cache.items():
        coords = payload["coords_2d"]
        experts = payload["expert_indices"]
        token_indices = payload["token_indices"]
        with (csv_dir / f"{layer}_token_tsne_2d.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["expert", "group", "token_index", "x", "y"])
            for row_idx in range(coords.shape[0]):
                expert_idx = int(experts[row_idx].item())
                group = "wiki_old" if expert_idx < old_expert_count else "code_new"
                writer.writerow(
                    [
                        f"e{expert_idx}",
                        group,
                        int(token_indices[row_idx].item()),
                        f"{float(coords[row_idx, 0].item()):.6f}",
                        f"{float(coords[row_idx, 1].item()):.6f}",
                    ]
                )


def save_token_tsne_plots(token_tsne_cache, output_dir: Path, title_prefix: str, plot_layers, old_expert_count: int):
    selected_layers = [layer for layer in plot_layers if layer in token_tsne_cache]
    if not selected_layers:
        return
    cols = min(3, len(selected_layers))
    rows = math.ceil(len(selected_layers) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.0 * rows))
    flat_axes = list(axes.reshape(-1)) if hasattr(axes, "reshape") else [axes]
    for ax_idx, ax in enumerate(flat_axes):
        if ax_idx >= len(selected_layers):
            ax.axis("off")
            continue
        layer = selected_layers[ax_idx]
        payload = token_tsne_cache[layer]
        coords = payload["coords_2d"]
        experts = payload["expert_indices"]
        old_mask = experts < old_expert_count
        new_mask = ~old_mask
        if bool(old_mask.any()):
            ax.scatter(
                coords[old_mask, 0],
                coords[old_mask, 1],
                s=10,
                alpha=0.35,
                color="#e45756",
                label="Wiki experts (e0-e3)",
            )
        if bool(new_mask.any()):
            ax.scatter(
                coords[new_mask, 0],
                coords[new_mask, 1],
                s=10,
                alpha=0.35,
                color="#4c78a8",
                label="Code experts (e4-e6)",
            )
        ax.set_title(layer)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        if ax_idx == 0:
            ax.legend(loc="best", fontsize=8)
    fig.suptitle(title_prefix)
    fig.tight_layout()
    tsne_dir = output_dir / "token_tsne"
    tsne_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(tsne_dir / "expert_output_similarity_tsne.png", dpi=200)
    plt.close(fig)


def recompute_for_dataset(dataset_dir: Path, args):
    raw_dir = dataset_dir / "raw_expert_outputs"
    layer_files = sorted(raw_dir.glob("layer_*_expert_outputs.pt"), key=layer_sort_key)
    if not layer_files:
        raise RuntimeError(f"No raw expert outputs found under {raw_dir}")

    metadata = {}
    metadata_path = dataset_dir / "expert_output_similarity.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    layer_results = []
    token_tsne_cache = {}
    for layer_file in layer_files:
        layer_key = layer_file.stem.replace("_expert_outputs", "")
        outputs = torch.load(layer_file, map_location="cpu")
        matrix = compute_linear_cka_matrix(outputs)
        token_sample = sample_token_level_outputs(
            outputs,
            args.tsne_sample_per_expert,
            1234 + int(layer_key.split("_")[-1]),
        )
        token_tsne = exact_tsne(
            token_sample["flat_outputs"],
            perplexity=args.tsne_perplexity,
            iterations=args.tsne_iterations,
            learning_rate=args.tsne_learning_rate,
            early_exaggeration=args.tsne_early_exaggeration,
            seed=2234 + int(layer_key.split("_")[-1]),
        )
        token_tsne_cache[layer_key] = {
            "coords_2d": token_tsne,
            "expert_indices": token_sample["expert_indices"],
            "token_indices": token_sample["token_indices"],
            "sample_count_per_expert": token_sample["sample_count_per_expert"],
        }
        layer_results.append(
            {
                "layer": layer_key,
                "num_tokens": int(outputs.shape[1]),
                "num_experts": int(outputs.shape[0]),
                "output_dim": int(outputs.shape[2]),
                "matrix": [[float(v) for v in row] for row in matrix.tolist()],
                "summary": summarize_similarity(matrix, args.source_num_experts),
                "raw_expert_output_file": layer_file.relative_to(dataset_dir).as_posix(),
                "token_tsne_csv_file": f"layer_tables/token_tsne/{layer_key}_token_tsne_2d.csv",
                "token_tsne_sample_count_per_expert": token_sample["sample_count_per_expert"],
            }
        )

    plot_layers = choose_plot_layers(layer_results, args.plot_layers)
    save_csvs(layer_results, dataset_dir)
    save_token_tsne_csvs(token_tsne_cache, dataset_dir, args.source_num_experts)
    save_heatmaps(
        layer_results,
        dataset_dir,
        f"Expert output linear CKA: {metadata.get('label', dataset_dir.name)}",
        plot_layers,
        args.heatmap_vmin,
        args.heatmap_vmax,
    )
    save_token_tsne_plots(
        token_tsne_cache,
        dataset_dir,
        f"Expert output token t-SNE: {metadata.get('label', dataset_dir.name)}",
        plot_layers,
        args.source_num_experts,
    )

    result = {
        "metric": args.metric,
        "source_num_experts": args.source_num_experts,
        "plot_layers": plot_layers,
        "heatmap_vmin": args.heatmap_vmin,
        "heatmap_vmax": args.heatmap_vmax,
        "token_tsne_sample_per_expert": args.tsne_sample_per_expert,
        "token_tsne_perplexity": args.tsne_perplexity,
        "token_tsne_iterations": args.tsne_iterations,
        "token_tsne_learning_rate": args.tsne_learning_rate,
        "token_tsne_early_exaggeration": args.tsne_early_exaggeration,
        "token_tsne_plot_filename": "token_tsne/expert_output_similarity_tsne.png",
        "recomputed_from_raw_outputs": True,
        "raw_output_dir": "raw_expert_outputs",
        "base_similarity_json": metadata_path.name if metadata_path.exists() else None,
        "layer_results": layer_results,
    }
    (dataset_dir / "expert_output_linear_cka.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    dataset_dirs = discover_dataset_dirs(input_dir)
    if not dataset_dirs:
        raise RuntimeError(
            f"No dataset directories with raw_expert_outputs were found under {input_dir}"
        )
    for dataset_dir in dataset_dirs:
        recompute_for_dataset(dataset_dir, args)
        print(f"Saved linear CKA outputs under: {dataset_dir}")


if __name__ == "__main__":
    main()
