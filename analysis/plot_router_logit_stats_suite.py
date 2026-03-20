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


def expert_matrix(payload, metric: str):
    layer_keys = sorted(payload["layers"].keys())
    rows = []
    for layer_key in layer_keys:
        rows.append([expert[metric] for expert in payload["layers"][layer_key]["experts"]])
    return np.array(rows, dtype=float), layer_keys


def render_heatmap_suite(payloads, metric: str, title: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    matrices = [expert_matrix(payloads[key], metric)[0] for key, _ in MODEL_ORDER]
    all_values = np.concatenate([matrix.ravel() for matrix in matrices])
    vmin = float(np.min(all_values))
    vmax = float(np.max(all_values))
    if vmin == vmax:
        vmax = vmin + 1.0

    for ax, (key, label) in zip(axes.flat, MODEL_ORDER):
        matrix, layer_keys = expert_matrix(payloads[key], metric)
        image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel("Expert index")
        ax.set_ylabel("Layer")
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_yticks(range(matrix.shape[0]))
        ax.set_yticklabels([layer_key.replace("layer_", "") for layer_key in layer_keys])

    fig.suptitle(title, fontsize=16)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.86, label=metric)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def render_group_histograms(payloads, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    for ax, (key, label) in zip(axes.flat, MODEL_ORDER):
        payload = payloads[key]
        hist_cfg = payload["histogram"]
        bins = hist_cfg["bins"]
        hist_min = hist_cfg["min"]
        hist_max = hist_cfg["max"]
        centers = np.linspace(hist_min, hist_max, bins, endpoint=False) + (hist_max - hist_min) / bins / 2

        old_hist = np.array(payload["overall"]["old_histogram"], dtype=float)
        new_hist = np.array(payload["overall"]["new_histogram"], dtype=float)
        if old_hist.sum() > 0:
            old_hist = old_hist / old_hist.sum()
        if new_hist.sum() > 0:
            new_hist = new_hist / new_hist.sum()

        ax.plot(centers, old_hist, color="#1f77b4", linewidth=2, label="Old experts (0-3)")
        ax.plot(centers, new_hist, color="#cc6b00", linewidth=2, label="New experts (4-6)")
        ax.set_title(label)
        ax.set_xlabel("Routing logit")
        ax.set_ylabel("Normalized density")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def render_group_summary(payloads, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [label for _, label in MODEL_ORDER]
    x = np.arange(len(labels))
    width = 0.36

    old_means = [payloads[key]["overall"]["old_mean"] for key, _ in MODEL_ORDER]
    new_means = [payloads[key]["overall"]["new_mean"] for key, _ in MODEL_ORDER]
    old_p95 = [payloads[key]["overall"]["old_p95_approx"] for key, _ in MODEL_ORDER]
    new_p95 = [payloads[key]["overall"]["new_p95_approx"] for key, _ in MODEL_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].bar(x - width / 2, old_means, width=width, label="Old experts (0-3)", color="#1f77b4")
    axes[0].bar(x + width / 2, new_means, width=width, label="New experts (4-6)", color="#cc6b00")
    axes[0].set_title("Mean routing logits by expert group")
    axes[0].set_ylabel("Mean logit")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x - width / 2, old_p95, width=width, label="Old experts (0-3)", color="#1f77b4")
    axes[1].bar(x + width / 2, new_p95, width=width, label="New experts (4-6)", color="#cc6b00")
    axes[1].set_title("Approximate p95 routing logits by expert group")
    axes[1].set_ylabel("Approximate p95 logit")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot router logit statistics for continual models.")
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

    render_heatmap_suite(payloads, "mean", "Router logit mean heatmaps", output_dir / "router_logit_mean_heatmaps.svg")
    render_heatmap_suite(payloads, "p95_approx", "Router logit p95 heatmaps", output_dir / "router_logit_p95_heatmaps.svg")
    render_group_histograms(payloads, output_dir / "router_logit_group_histograms.svg")
    render_group_summary(payloads, output_dir / "router_logit_group_summary.svg")

    summary = {key: payloads[key]["overall"] for key, _ in MODEL_ORDER}
    (output_dir / "router_logit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
