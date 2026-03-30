#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_dataset_from_name(name: str):
    parts = name.split("_")
    return next((part for part in parts if part in ("wiki", "code")), None)


def load_top1_routing(input_dir: Path):
    parsed = {}
    for path in sorted(input_dir.glob("*_top1_routing.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        dataset = parse_dataset_from_name(path.stem)
        if dataset is None:
            continue
        parsed[dataset] = data
    return parsed


def ordered_layers(layer_dict):
    return sorted(layer_dict.keys(), key=lambda item: int(item.split("_")[-1]))


def plot_old_vs_new(parsed, datasets, source_num_experts: int, output_path: Path):
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.5), squeeze=False)
    axes = axes[0]
    for ax, dataset in zip(axes, datasets):
        layer_map = parsed[dataset]
        layers = ordered_layers(layer_map)
        old_vals = []
        new_vals = []
        for layer in layers:
            fractions = layer_map[layer]["expert_assignment_fractions"]
            old_vals.append(float(sum(fractions[:source_num_experts])))
            new_vals.append(float(sum(fractions[source_num_experts:])))
        x = np.arange(len(layers))
        old_bars = ax.bar(x, old_vals, color="#4C78A8", label="Old/wiki experts")
        new_bars = ax.bar(x, new_vals, bottom=old_vals, color="#E45756", label="New/code experts")
        ax.set_title(f"{dataset} dataset")
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=30)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Top-1 selection fraction")
        for bar, value in zip(old_bars, old_vals):
            if value <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value / 2,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )
        for bar, old_value, new_value in zip(new_bars, old_vals, new_vals):
            if new_value <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                old_value + new_value / 2,
                f"{new_value:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Top-1 routing: old vs new expert usage")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_expert_heatmap(parsed, datasets, output_path: Path):
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), squeeze=False)
    axes = axes[0]
    for ax, dataset in zip(axes, datasets):
        layer_map = parsed[dataset]
        layers = ordered_layers(layer_map)
        matrix = np.array([layer_map[layer]["expert_assignment_fractions"] for layer in layers], dtype=float)
        im = ax.imshow(matrix, vmin=0.0, vmax=max(matrix.max(), 1e-8), cmap="Blues", aspect="auto")
        expert_labels = [f"e{i}" for i in range(matrix.shape[1])]
        ax.set_title(f"{dataset} dataset")
        ax.set_xticks(np.arange(len(expert_labels)))
        ax.set_xticklabels(expert_labels)
        ax.set_yticks(np.arange(len(layers)))
        ax.set_yticklabels(layers)
        ax.set_xlabel("Expert")
        ax.set_ylabel("Layer")
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                text_color = "white" if value > matrix.max() * 0.45 else "black"
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Top-1 routing: expert assignment fraction by layer")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--source-num-experts", type=int, default=4)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    parsed = load_top1_routing(input_dir)
    if not parsed:
        raise SystemExit(f"No *_top1_routing.json files found under {input_dir}")

    datasets = [dataset for dataset in ("wiki", "code") if dataset in parsed]
    plot_old_vs_new(parsed, datasets, args.source_num_experts, input_dir / "top1_old_vs_new.png")
    plot_expert_heatmap(parsed, datasets, input_dir / "top1_expert_heatmap.png")


if __name__ == "__main__":
    main()
