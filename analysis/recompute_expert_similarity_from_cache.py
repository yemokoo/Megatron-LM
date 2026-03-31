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
    for layer_file in layer_files:
        layer_key = layer_file.stem.replace("_expert_outputs", "")
        outputs = torch.load(layer_file, map_location="cpu")
        matrix = compute_linear_cka_matrix(outputs)
        layer_results.append(
            {
                "layer": layer_key,
                "num_tokens": int(outputs.shape[1]),
                "num_experts": int(outputs.shape[0]),
                "output_dim": int(outputs.shape[2]),
                "matrix": [[float(v) for v in row] for row in matrix.tolist()],
                "summary": summarize_similarity(matrix, args.source_num_experts),
                "raw_expert_output_file": layer_file.relative_to(dataset_dir).as_posix(),
            }
        )

    plot_layers = choose_plot_layers(layer_results, args.plot_layers)
    save_csvs(layer_results, dataset_dir)
    save_heatmaps(
        layer_results,
        dataset_dir,
        f"Expert output linear CKA: {metadata.get('label', dataset_dir.name)}",
        plot_layers,
        args.heatmap_vmin,
        args.heatmap_vmax,
    )

    result = {
        "metric": args.metric,
        "source_num_experts": args.source_num_experts,
        "plot_layers": plot_layers,
        "heatmap_vmin": args.heatmap_vmin,
        "heatmap_vmax": args.heatmap_vmax,
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
