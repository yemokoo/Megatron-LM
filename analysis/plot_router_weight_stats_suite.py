#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = (
    ("a_to_b_full", "A->B Shared Unfreeze"),
    ("a_to_b_new_only", "A->B Shared Freeze"),
    ("b_to_a_full", "B->A Shared Unfreeze"),
    ("b_to_a_new_only", "B->A Shared Freeze"),
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def heatmap_array(model_payload, metric: str):
    layers = model_payload["stats"]["layers"]
    return np.array([[expert[metric] for expert in layer["experts"]] for layer in layers], dtype=float)


def render_heatmap_suite(payloads, metric: str, title: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    all_values = np.concatenate([heatmap_array(payloads[key], metric).ravel() for key, _ in MODEL_ORDER])
    vmin = float(all_values.min())
    vmax = float(all_values.max())
    if vmin == vmax:
        vmax = vmin + 1.0

    for ax, (key, label) in zip(axes.flat, MODEL_ORDER):
        data = heatmap_array(payloads[key], metric)
        image = ax.imshow(data, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel("Expert index")
        ax.set_ylabel("MoE layer")
        ax.set_xticks(range(data.shape[1]))
        ax.set_yticks(range(data.shape[0]))
        ax.set_yticklabels([str(layer["layer_number"]) for layer in payloads[key]["stats"]["layers"]])

    fig.suptitle(title, fontsize=16)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.86, label=metric)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def render_group_summary(payloads, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [label for _, label in MODEL_ORDER]
    x = np.arange(len(labels))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    abs_old = [payloads[key]["stats"]["overall"]["old"]["abs_mean"] for key, _ in MODEL_ORDER]
    abs_new = [payloads[key]["stats"]["overall"]["new"]["abs_mean"] for key, _ in MODEL_ORDER]
    norm_old = [payloads[key]["stats"]["overall"]["old"]["l2_norm"] for key, _ in MODEL_ORDER]
    norm_new = [payloads[key]["stats"]["overall"]["new"]["l2_norm"] for key, _ in MODEL_ORDER]

    axes[0].bar(x - width / 2, abs_old, width=width, label="Old experts (0-3)", color="#1f77b4")
    axes[0].bar(x + width / 2, abs_new, width=width, label="New experts (4-6)", color="#cc6b00")
    axes[0].set_title("Router abs-mean by expert group")
    axes[0].set_ylabel("Mean absolute weight")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x - width / 2, norm_old, width=width, label="Old experts (0-3)", color="#1f77b4")
    axes[1].bar(x + width / 2, norm_new, width=width, label="New experts (4-6)", color="#cc6b00")
    axes[1].set_title("Router L2 norm by expert group")
    axes[1].set_ylabel("Row L2 norm")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def render_per_model_bars(payload, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = payload["stats"]["overall"]["expert_means"]
    experts = [row["expert_idx"] for row in rows]
    abs_means = [row["abs_mean"] for row in rows]
    l2_norms = [row["l2_norm"] for row in rows]
    colors = ["#1f77b4" if row["group"] == "old" else "#cc6b00" for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].bar(experts, abs_means, color=colors)
    axes[0].set_title(f'{payload["model_label"]}: expert abs-mean')
    axes[0].set_xlabel("Expert index")
    axes[0].set_ylabel("Mean absolute weight")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(experts, l2_norms, color=colors)
    axes[1].set_title(f'{payload["model_label"]}: expert L2 norm')
    axes[1].set_xlabel("Expert index")
    axes[1].set_ylabel("Row L2 norm")
    axes[1].grid(axis="y", alpha=0.3)

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot router weight statistics for continual models.")
    parser.add_argument("--a-to-b-full", required=True)
    parser.add_argument("--a-to-b-new-only", required=True)
    parser.add_argument("--b-to-a-full", required=True)
    parser.add_argument("--b-to-a-new-only", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads = {
        "a_to_b_full": load_json(Path(args.a_to_b_full)),
        "a_to_b_new_only": load_json(Path(args.a_to_b_new_only)),
        "b_to_a_full": load_json(Path(args.b_to_a_full)),
        "b_to_a_new_only": load_json(Path(args.b_to_a_new_only)),
    }

    render_heatmap_suite(payloads, "abs_mean", "Router weight abs-mean heatmaps", output_dir / "router_abs_mean_heatmaps.svg")
    render_heatmap_suite(payloads, "l2_norm", "Router row L2 norm heatmaps", output_dir / "router_l2_norm_heatmaps.svg")
    render_heatmap_suite(payloads, "mean", "Router weight mean heatmaps", output_dir / "router_mean_heatmaps.svg")
    render_group_summary(payloads, output_dir / "router_group_summary.svg")

    for key, _label in MODEL_ORDER:
        render_per_model_bars(payloads[key], output_dir / f"{key}_expert_bars.svg")

    combined = {
        key: payloads[key]["stats"]["overall"]
        for key, _ in MODEL_ORDER
    }
    (output_dir / "router_weight_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
