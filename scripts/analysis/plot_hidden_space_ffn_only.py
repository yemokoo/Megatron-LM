#!/usr/bin/env python3
"""Plot hidden-space movement across FFN-only continual-learning checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STAGE_COLORS = {
    "wiki_only": "#2563eb",
    "code_trained": "#f97316",
    "router_retuned": "#16a34a",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-only", required=True, help="NPZ hidden dump after wiki-only training.")
    parser.add_argument("--code-trained", required=True, help="NPZ hidden dump after code training.")
    parser.add_argument("--router-retuned", required=True, help="NPZ hidden dump after router retuning.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--method", choices=["pca", "umap"], default="pca")
    parser.add_argument("--max-points-per-stage", type=int, default=1200)
    parser.add_argument("--max-vectors", type=int, default=350)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def load_dump(path: str, fallback_label: str):
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata"]))
    label = metadata.get("label") or fallback_label
    return {
        "path": path,
        "label": label,
        "hidden": data["hidden_layers"].astype(np.float32),
        "layers": data["layer_numbers"].astype(np.int64),
        "token_ids": data["token_ids"].astype(np.int64),
        "metadata": metadata,
    }


def pca2(x: np.ndarray):
    x = x.astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def reduce2(x: np.ndarray, method: str, seed: int):
    if method == "umap":
        try:
            import umap

            return umap.UMAP(
                n_components=2,
                random_state=seed,
                n_neighbors=30,
                min_dist=0.08,
                metric="cosine",
            ).fit_transform(x)
        except Exception as exc:
            print(f"[WARN] UMAP unavailable or failed ({exc}); falling back to PCA.")
    return pca2(x)


def subsample_indices(n: int, max_points: int, rng: np.random.Generator):
    if n <= max_points:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def common_subsample_indices(dumps, max_points: int, rng: np.random.Generator):
    n = min(dump["hidden"].shape[1] for dump in dumps)
    return subsample_indices(n, max_points, rng)


def split_stage_coords(coords: np.ndarray, n_points: int):
    return [coords[i * n_points : (i + 1) * n_points] for i in range(3)]


def draw_vectors(ax, coords_by_stage, vector_indices, alpha=0.24, linewidth=0.55, scale=1.0):
    wiki, code, retune = coords_by_stage
    for idx in vector_indices:
        ax.annotate(
            "",
            xy=code[idx],
            xytext=wiki[idx],
            arrowprops=dict(
                arrowstyle="->",
                color="#f97316",
                alpha=alpha,
                lw=linewidth,
                shrinkA=0,
                shrinkB=0,
            ),
        )
        ax.annotate(
            "",
            xy=retune[idx],
            xytext=code[idx],
            arrowprops=dict(
                arrowstyle="->",
                color="#16a34a",
                alpha=alpha,
                lw=linewidth,
                shrinkA=0,
                shrinkB=0,
            ),
        )


def expand_limits(ax, coords, pad_fraction=0.08):
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    xpad = max((xmax - xmin) * pad_fraction, 1e-5)
    ypad = max((ymax - ymin) * pad_fraction, 1e-5)
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)


def paired_cosine(a: np.ndarray, b: np.ndarray):
    n = min(a.shape[0], b.shape[0])
    a = a[:n]
    b = b[:n]
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom = np.maximum(denom, 1e-8)
    return float(np.mean(np.sum(a * b, axis=1) / denom))


def centroid_distance(a: np.ndarray, b: np.ndarray):
    return float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))


def collect_metrics(dumps):
    names = [dump["label"] for dump in dumps]
    metrics = {"layers": []}
    for layer_idx, layer_number in enumerate(dumps[0]["layers"].tolist()):
        row = {"layer": int(layer_number)}
        for i in range(len(dumps)):
            for j in range(i + 1, len(dumps)):
                key = f"{names[i]}__vs__{names[j]}"
                a = dumps[i]["hidden"][layer_idx]
                b = dumps[j]["hidden"][layer_idx]
                row[f"{key}/paired_cosine"] = paired_cosine(a, b)
                row[f"{key}/centroid_l2"] = centroid_distance(a, b)
        metrics["layers"].append(row)

    avg = []
    for dump in dumps:
        avg.append(dump["hidden"].mean(axis=0))
    metrics["layer_average"] = {}
    for i in range(len(dumps)):
        for j in range(i + 1, len(dumps)):
            key = f"{names[i]}__vs__{names[j]}"
            metrics["layer_average"][f"{key}/paired_cosine"] = paired_cosine(avg[i], avg[j])
            metrics["layer_average"][f"{key}/centroid_l2"] = centroid_distance(avg[i], avg[j])
    return metrics


def plot_layer_grid(dumps, out_path: Path, method: str, max_points: int, max_vectors: int, seed: int):
    rng = np.random.default_rng(seed)
    layers = dumps[0]["layers"]
    n_layers = len(layers)
    ncols = min(3, n_layers)
    nrows = math.ceil(n_layers / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows), squeeze=False)

    for layer_idx, layer_number in enumerate(layers):
        ax = axes[layer_idx // ncols][layer_idx % ncols]
        idx = common_subsample_indices(dumps, max_points, rng)
        sampled = [dump["hidden"][layer_idx][idx] for dump in dumps]
        coords = reduce2(np.concatenate(sampled, axis=0), method, seed + int(layer_number))
        coords_by_stage = split_stage_coords(coords, len(idx))
        vector_idx = subsample_indices(len(idx), min(max_vectors, len(idx)), rng)

        draw_vectors(ax, coords_by_stage, vector_idx, alpha=0.20, linewidth=0.45)

        for dump, part in zip(dumps, coords_by_stage):
            color = STAGE_COLORS.get(dump["label"], None)
            ax.scatter(
                part[:, 0],
                part[:, 1],
                s=8,
                alpha=0.40,
                label=dump["label"],
                c=color,
                edgecolors="none",
            )
        centroids = np.stack([part.mean(axis=0) for part in coords_by_stage])
        ax.plot(centroids[:, 0], centroids[:, 1], color="#111827", lw=1.25, alpha=0.75)
        ax.scatter(centroids[:, 0], centroids[:, 1], color="#111827", s=18, alpha=0.9)
        expand_limits(ax, coords)
        ax.set_title(f"Layer {int(layer_number)}", fontsize=12, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.16)

    for idx in range(n_layers, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(f"FFN-only Wiki Probe Hidden Space by Layer ({method.upper()})", y=0.995, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_layer_average(dumps, out_path: Path, method: str, max_points: int, max_vectors: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = common_subsample_indices(dumps, max_points, rng)
    sampled = [dump["hidden"].mean(axis=0)[idx] for dump in dumps]
    coords = reduce2(np.concatenate(sampled, axis=0), method, seed + 999)
    coords_by_stage = split_stage_coords(coords, len(idx))
    vector_idx = subsample_indices(len(idx), min(max_vectors, len(idx)), rng)

    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    draw_vectors(ax, coords_by_stage, vector_idx, alpha=0.22, linewidth=0.55)
    for dump, part in zip(dumps, coords_by_stage):
        color = STAGE_COLORS.get(dump["label"], None)
        ax.scatter(
            part[:, 0],
            part[:, 1],
            s=10,
            alpha=0.42,
            label=dump["label"],
            c=color,
            edgecolors="none",
        )
    centroids = np.stack([part.mean(axis=0) for part in coords_by_stage])
    ax.plot(centroids[:, 0], centroids[:, 1], color="#111827", lw=1.6, alpha=0.78)
    ax.scatter(centroids[:, 0], centroids[:, 1], color="#111827", s=26, alpha=0.92)
    expand_limits(ax, coords)
    ax.set_title(f"FFN-only Wiki Probe Hidden Space, Layer-Average ({method.upper()})", weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dumps = [
        load_dump(args.wiki_only, "wiki_only"),
        load_dump(args.code_trained, "code_trained"),
        load_dump(args.router_retuned, "router_retuned"),
    ]
    layer_ref = dumps[0]["layers"].tolist()
    for dump in dumps[1:]:
        if dump["layers"].tolist() != layer_ref:
            raise SystemExit("Layer numbers do not match across hidden dumps.")

    plot_layer_grid(
        dumps,
        out_dir / f"ffn_only_hidden_layers_{args.method}.png",
        args.method,
        args.max_points_per_stage,
        args.max_vectors,
        args.seed,
    )
    plot_layer_average(
        dumps,
        out_dir / f"ffn_only_hidden_layer_average_{args.method}.png",
        args.method,
        args.max_points_per_stage,
        args.max_vectors,
        args.seed,
    )

    metrics = collect_metrics(dumps)
    (out_dir / "ffn_only_hidden_space_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[DONE] wrote plots and metrics to {out_dir}")


if __name__ == "__main__":
    main()
