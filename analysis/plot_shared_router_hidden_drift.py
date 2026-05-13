#!/usr/bin/env python3
"""Plot hidden-state drift across shared-router continual-learning checkpoints."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Plot shared-router hidden drift.")
    parser.add_argument("--input", action="append", required=True, help="Repeated: label=path.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-label", default="wiki_1800")
    parser.add_argument("--pca-layers", default="2,5,9")
    parser.add_argument("--pca-token-count", type=int, default=512)
    return parser.parse_args()


def parse_input_specs(specs):
    items = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --input {spec!r}; expected label=path.pt")
        label, path = spec.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Invalid empty label in --input {spec!r}")
        items.append((label, Path(path)))
    return items


def step_from_label(label):
    if label.startswith("code_"):
        try:
            return int(label.split("_", 1)[1])
        except ValueError:
            return 0
    if label.startswith("wiki"):
        return 0
    return 0


def load_payloads(items):
    payloads = {}
    for label, path in items:
        payloads[label] = torch.load(path, map_location="cpu")
    return payloads


def common_layers(payloads):
    layer_sets = [
        set(payload["hidden_by_layer"].keys())
        for payload in payloads.values()
    ]
    layers = sorted(set.intersection(*layer_sets))
    if not layers:
        raise RuntimeError("No common captured layers found across checkpoints.")
    return layers


def compute_metrics(payloads, baseline_label, layers):
    baseline = payloads[baseline_label]["hidden_by_layer"]
    rows = []
    for label, payload in payloads.items():
        hidden_by_layer = payload["hidden_by_layer"]
        step = step_from_label(label)
        for layer in layers:
            ref = baseline[layer].float()
            cur = hidden_by_layer[layer].float()
            count = min(ref.shape[0], cur.shape[0])
            ref = ref[:count]
            cur = cur[:count]
            cosine = torch.nn.functional.cosine_similarity(ref, cur, dim=-1)
            diff = cur - ref
            rows.append(
                {
                    "label": label,
                    "step": step,
                    "layer": layer,
                    "token_count": count,
                    "cosine_similarity_mean": float(cosine.mean().item()),
                    "cosine_distance_mean": float((1.0 - cosine).mean().item()),
                    "rms_l2_drift": float(torch.sqrt((diff * diff).sum(dim=-1) / ref.shape[-1]).mean().item()),
                    "norm_ratio_mean": float(
                        (cur.norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-12)).mean().item()
                    ),
                }
            )
    rows.sort(key=lambda item: (item["step"], item["layer"]))
    return rows


def write_metrics(rows, output_dir):
    path = output_dir / "hidden_drift_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def matrix_from_rows(rows, layers, metric_name):
    steps = sorted({row["step"] for row in rows})
    layer_to_idx = {layer: i for i, layer in enumerate(layers)}
    step_to_idx = {step: i for i, step in enumerate(steps)}
    matrix = np.full((len(layers), len(steps)), np.nan, dtype=np.float32)
    for row in rows:
        matrix[layer_to_idx[row["layer"]], step_to_idx[row["step"]]] = row[metric_name]
    return steps, matrix


def plot_heatmap(rows, layers, output_dir, metric_name, title, cmap="magma"):
    steps, matrix = matrix_from_rows(rows, layers, metric_name)
    fig_width = max(8.0, 1.2 * len(steps))
    fig_height = max(5.0, 0.45 * len(layers))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(steps)))
    ax.set_xticklabels([str(step) for step in steps])
    ax.set_yticks(np.arange(len(layers)))
    ax.set_yticklabels(layers)
    ax.set_xlabel("Code training step checkpoint")
    ax.set_ylabel("Shared-router layer")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_name)
    fig.tight_layout()
    path = output_dir / f"{metric_name}_heatmap.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def pca_2d(tensor):
    x = tensor.float().numpy()
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    return x @ vh[:2].T


def plot_pca(payloads, baseline_label, layers, output_dir, token_count):
    selected = []
    requested = {f"layer_{int(item):02d}" for item in layers if item.strip()}
    available = common_layers(payloads)
    for layer in available:
        if layer in requested:
            selected.append(layer)
    if not selected:
        return []

    paths = []
    labels = sorted(payloads.keys(), key=step_from_label)
    for layer in selected:
        chunks = []
        counts = []
        for label in labels:
            hidden = payloads[label]["hidden_by_layer"][layer].float()[:token_count]
            chunks.append(hidden)
            counts.append(hidden.shape[0])
        combined = torch.cat(chunks, dim=0)
        coords = pca_2d(combined)

        fig, ax = plt.subplots(figsize=(7.5, 6.0))
        offset = 0
        cmap = plt.get_cmap("viridis")
        for i, (label, count) in enumerate(zip(labels, counts)):
            color = cmap(i / max(len(labels) - 1, 1))
            part = coords[offset: offset + count]
            ax.scatter(part[:, 0], part[:, 1], s=7, alpha=0.35, label=label, color=color)
            centroid = part.mean(axis=0)
            ax.scatter(centroid[0], centroid[1], s=70, marker="x", color=color)
            offset += count
        ax.set_title(f"Fixed Wiki hidden cloud PCA: {layer}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(markerscale=2.0, fontsize=8, loc="best")
        fig.tight_layout()
        path = output_dir / f"{layer}_pca_hidden_cloud.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    items = parse_input_specs(args.input)
    payloads = load_payloads(items)
    if args.baseline_label not in payloads:
        raise RuntimeError(f"Missing baseline label {args.baseline_label!r}")

    layers = common_layers(payloads)
    rows = compute_metrics(payloads, args.baseline_label, layers)
    csv_path = write_metrics(rows, output_dir)
    heatmaps = [
        plot_heatmap(
            rows,
            layers,
            output_dir,
            "cosine_distance_mean",
            "Fixed Wiki hidden drift from Wiki checkpoint (1 - cosine)",
            cmap="magma",
        ),
        plot_heatmap(
            rows,
            layers,
            output_dir,
            "rms_l2_drift",
            "Fixed Wiki hidden drift from Wiki checkpoint (RMS L2)",
            cmap="viridis",
        ),
        plot_heatmap(
            rows,
            layers,
            output_dir,
            "norm_ratio_mean",
            "Fixed Wiki hidden norm ratio vs Wiki checkpoint",
            cmap="coolwarm",
        ),
    ]
    pca_paths = plot_pca(
        payloads,
        args.baseline_label,
        args.pca_layers.split(","),
        output_dir,
        args.pca_token_count,
    )
    summary = {
        "baseline_label": args.baseline_label,
        "labels": [label for label, _path in items],
        "layers": layers,
        "metrics_csv": str(csv_path),
        "heatmaps": [str(path) for path in heatmaps],
        "pca_plots": [str(path) for path in pca_paths],
        "interpretation": (
            "All checkpoints are evaluated on the same fixed Wiki tokens. "
            "Non-zero drift means the hidden states entering the shared routers "
            "changed during Code continual training."
        ),
    }
    (output_dir / "hidden_drift_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
