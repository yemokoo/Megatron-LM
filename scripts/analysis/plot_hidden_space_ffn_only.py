#!/usr/bin/env python3
"""Plot hidden-space movement across FFN-only continual-learning checkpoints."""

from __future__ import annotations

import argparse
import csv
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
    "router_retuned": "#9333ea",
}

STAGE_DISPLAY_NAMES = {
    "wiki_only": "Wiki-only",
    "code_trained": "Code-trained",
    "router_retuned": "Router-retuned",
}

MODEL_LABEL = "FFN-only"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-only", required=True, help="NPZ hidden dump after wiki-only training.")
    parser.add_argument("--code-trained", required=True, help="NPZ hidden dump after code training.")
    parser.add_argument("--router-retuned", required=True, help="NPZ hidden dump after router retuning.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--model-label",
        default="FFN-only",
        help="Model label used in plot titles. Defaults to the original FFN-only label.",
    )
    parser.add_argument("--method", choices=["pca", "umap"], default="pca")
    parser.add_argument("--max-points-per-stage", type=int, default=1200)
    parser.add_argument("--max-vectors", type=int, default=350)
    parser.add_argument("--trim-percentile", type=float, default=99.0)
    parser.add_argument("--density-bins", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--probe-task",
        default="wiki",
        help="Probe task used for the hidden dumps. Controls pairwise reference stage: wiki -> wiki_only, code -> code_trained.",
    )
    parser.add_argument(
        "--only-pairwise-density",
        action="store_true",
        help="Only redraw pairwise density plots for the selected probe task/base stage.",
    )
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
    n_samples, n_features = x.shape
    if n_samples >= n_features:
        # Hidden dumps usually have many more sampled tokens than hidden dims.
        # Eigendecomposing the feature covariance is much faster than full SVD.
        cov = (x.T @ x) / max(n_samples - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        components = eigvecs[:, np.argsort(eigvals)[-2:][::-1]]
        return x @ components
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
                color="#9333ea",
                alpha=alpha,
                lw=linewidth,
                shrinkA=0,
                shrinkB=0,
            ),
        )


def expand_limits(ax, coords, pad_fraction=0.08, trim_percentile=None):
    if trim_percentile is not None and 0 < trim_percentile < 100:
        low = (100.0 - trim_percentile) / 2.0
        high = 100.0 - low
        xmin, ymin = np.percentile(coords, low, axis=0)
        xmax, ymax = np.percentile(coords, high, axis=0)
    else:
        xmin, ymin = coords.min(axis=0)
        xmax, ymax = coords.max(axis=0)
    xpad = max((xmax - xmin) * pad_fraction, 1e-5)
    ypad = max((ymax - ymin) * pad_fraction, 1e-5)
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)


def smooth_histogram(hist: np.ndarray, passes: int = 2):
    kernel = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=np.float32)
    kernel /= kernel.sum()
    smoothed = hist.astype(np.float32, copy=True)
    for _ in range(passes):
        padded = np.pad(smoothed, 1, mode="edge")
        out = np.zeros_like(smoothed)
        for i in range(3):
            for j in range(3):
                out += kernel[i, j] * padded[i : i + smoothed.shape[0], j : j + smoothed.shape[1]]
        smoothed = out
    return smoothed


def percentile_bounds(points: np.ndarray, trim_percentile: float, pad_fraction: float = 0.08):
    if trim_percentile is not None and 0 < trim_percentile < 100:
        low = (100.0 - trim_percentile) / 2.0
        high = 100.0 - low
        xmin, ymin = np.percentile(points, low, axis=0)
        xmax, ymax = np.percentile(points, high, axis=0)
    else:
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
    xpad = max((xmax - xmin) * pad_fraction, 1e-5)
    ypad = max((ymax - ymin) * pad_fraction, 1e-5)
    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad


def density_color_steps(color: str, num_steps: int, max_alpha: float):
    rgb = matplotlib.colors.to_rgb(color)
    min_alpha = max(0.06, max_alpha * 0.22)
    alphas = np.linspace(min_alpha, max_alpha, num_steps)
    return [(rgb[0], rgb[1], rgb[2], float(alpha)) for alpha in alphas]


def draw_density_cloud(
    ax,
    points: np.ndarray,
    color: str,
    label: str,
    bins: int,
    trim_percentile: float,
    alpha: float = 0.34,
):
    xmin, xmax, ymin, ymax = percentile_bounds(points, trim_percentile, pad_fraction=0.12)
    inside = (
        (points[:, 0] >= xmin)
        & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] <= ymax)
    )
    clipped = points[inside]
    if len(clipped) < 8:
        ax.scatter(points[:, 0], points[:, 1], s=10, alpha=0.35, c=color, label=label, edgecolors="none")
        return

    hist, xedges, yedges = np.histogram2d(
        clipped[:, 0],
        clipped[:, 1],
        bins=bins,
        range=[[xmin, xmax], [ymin, ymax]],
    )
    hist = smooth_histogram(hist, passes=2)
    nonzero = hist[hist > 0]
    if len(nonzero) == 0:
        return

    levels = np.unique(
        np.concatenate(
            [
                [float(nonzero.min())],
                np.percentile(nonzero, [35, 50, 65, 78, 88, 95]),
                [float(nonzero.max()) + 1e-6],
            ]
        )
    )
    levels = levels[levels > 0]
    if len(levels) < 3:
        ax.scatter(clipped[:, 0], clipped[:, 1], s=10, alpha=0.35, c=color, label=label, edgecolors="none")
        return

    xcenters = (xedges[:-1] + xedges[1:]) / 2.0
    ycenters = (yedges[:-1] + yedges[1:]) / 2.0
    fill_colors = density_color_steps(color, len(levels) - 1, alpha)
    line_levels = levels[1:-1]
    line_widths = np.linspace(0.55, 1.45, len(line_levels)) if len(line_levels) else 0.8
    ax.contourf(xcenters, ycenters, hist.T, levels=levels, colors=fill_colors, antialiased=True)
    ax.contour(
        xcenters,
        ycenters,
        hist.T,
        levels=line_levels,
        colors=[color],
        alpha=0.80,
        linewidths=line_widths,
    )
    # A faint point layer keeps sparse tails visible without turning the plot back into a dot cloud.
    ax.scatter(clipped[:, 0], clipped[:, 1], s=5, alpha=0.08, c=color, edgecolors="none")
    ax.scatter([], [], c=color, alpha=0.75, label=label)


def normalized_density_grid(points: np.ndarray, bins: int, limits):
    xmin, xmax, ymin, ymax = limits
    inside = (
        (points[:, 0] >= xmin)
        & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] <= ymax)
    )
    clipped = points[inside]
    if len(clipped) < 2:
        clipped = points

    hist, _, _ = np.histogram2d(
        clipped[:, 0],
        clipped[:, 1],
        bins=bins,
        range=[[xmin, xmax], [ymin, ymax]],
    )
    hist = smooth_histogram(hist, passes=2)
    total = float(hist.sum())
    if total <= 0:
        return None
    return hist / total


def pca_kde_overlap(base_xy: np.ndarray, other_xy: np.ndarray, bins: int, limits):
    base_density = normalized_density_grid(base_xy, bins, limits)
    other_density = normalized_density_grid(other_xy, bins, limits)
    if base_density is None or other_density is None:
        return float("nan")
    return float(np.minimum(base_density, other_density).sum())


def kde_overlap_metrics_path(out_path: Path):
    return out_path.with_name(f"{out_path.stem}_kde_overlap.csv")


def write_kde_overlap_metrics(out_path: Path, rows):
    if not rows:
        return
    metrics_path = kde_overlap_metrics_path(out_path)
    fieldnames = [
        "plot",
        "model_label",
        "probe_task",
        "scope",
        "layer",
        "base",
        "other",
        "method",
        "kde_overlap",
    ]
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def draw_origin_marker(ax, zero_xy: np.ndarray):
    origin = zero_xy.mean(axis=0)
    ax.scatter(
        [origin[0]],
        [origin[1]],
        s=170,
        facecolors="white",
        edgecolors="#2563eb",
        linewidths=2.5,
        label="wiki_origin",
        zorder=8,
    )
    ax.scatter([origin[0]], [origin[1]], s=34, c="#2563eb", zorder=9)
    return origin


def display_stage_name(label: str):
    return STAGE_DISPLAY_NAMES.get(label, label.replace("_", " "))


def normalize_probe_task(probe_task: str):
    normalized = probe_task.strip().lower().replace("-", "_")
    if normalized in {"wiki", "wiki_probe"}:
        return "wiki"
    if normalized in {"code", "code_probe"}:
        return "code"
    return normalized


def probe_display_name(probe_task: str):
    normalized = normalize_probe_task(probe_task)
    if normalized == "wiki":
        return "Wiki Probe"
    if normalized == "code":
        return "Code Probe"
    return normalized.replace("_", " ").title()


def pairwise_base_label_for_probe(probe_task: str):
    return "code_trained" if normalize_probe_task(probe_task) == "code" else "wiki_only"


def pairwise_compare_labels(base_label: str):
    if base_label == "wiki_only":
        return ["code_trained", "router_retuned"]
    if base_label == "code_trained":
        return ["wiki_only", "router_retuned"]
    return [label for label in ["wiki_only", "code_trained", "router_retuned"] if label != base_label]


def stage_index_by_label(dumps):
    return {dump["label"]: idx for idx, dump in enumerate(dumps)}


def draw_pairwise_density_panel(
    ax,
    base_xy: np.ndarray,
    other_xy: np.ndarray,
    base_label: str,
    other_label: str,
    density_bins: int,
    trim_percentile: float,
    limits=None,
    show_legend: bool = True,
):
    if limits is None:
        limits = percentile_bounds(
            np.concatenate([base_xy, other_xy], axis=0),
            trim_percentile,
            pad_fraction=0.12,
        )
    overlap = pca_kde_overlap(base_xy, other_xy, density_bins, limits)
    base_color = STAGE_COLORS.get(base_label, "#2563eb")
    other_color = STAGE_COLORS.get(other_label, "#111827")
    draw_density_cloud(
        ax,
        base_xy,
        base_color,
        display_stage_name(base_label),
        density_bins,
        trim_percentile,
        alpha=0.32,
    )
    draw_density_cloud(
        ax,
        other_xy,
        other_color,
        display_stage_name(other_label),
        density_bins,
        trim_percentile,
        alpha=0.32,
    )

    base_centroid = base_xy.mean(axis=0)
    other_centroid = other_xy.mean(axis=0)
    ax.annotate(
        "",
        xy=other_centroid,
        xytext=base_centroid,
        arrowprops=dict(arrowstyle="->", color="#111827", lw=1.8, alpha=0.80),
    )
    ax.scatter(
        [base_centroid[0]],
        [base_centroid[1]],
        s=42,
        c=base_color,
        edgecolors="#111827",
        linewidths=0.75,
        zorder=8,
    )
    ax.scatter(
        [other_centroid[0]],
        [other_centroid[1]],
        s=42,
        c=other_color,
        edgecolors="#111827",
        linewidths=0.75,
        zorder=8,
    )
    xmin, xmax, ymin, ymax = limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if np.isfinite(overlap):
        ax.text(
            0.985,
            0.965,
            f"KDE overlap={overlap:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            weight="bold",
            color="#111827",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cbd5e1", alpha=0.82),
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.16)
    if show_legend:
        ax.legend(frameon=False, loc="upper left", fontsize=8)
    return overlap


def plot_delta_density_layer_grid(
    dumps,
    out_path: Path,
    method: str,
    max_points: int,
    seed: int,
    trim_percentile: float,
    density_bins: int,
):
    rng = np.random.default_rng(seed)
    layers = dumps[0]["layers"]
    n_layers = len(layers)
    ncols = min(3, n_layers)
    nrows = math.ceil(n_layers / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows), squeeze=False)

    for layer_idx, layer_number in enumerate(layers):
        ax = axes[layer_idx // ncols][layer_idx % ncols]
        idx = common_subsample_indices(dumps, max_points, rng)
        wiki = dumps[0]["hidden"][layer_idx][idx]
        code_delta = dumps[1]["hidden"][layer_idx][idx] - wiki
        retune_delta = dumps[2]["hidden"][layer_idx][idx] - wiki
        zero = np.zeros_like(code_delta)
        coords = reduce2(np.concatenate([zero, code_delta, retune_delta], axis=0), method, seed + 5000 + int(layer_number))
        zero_xy, code_xy, retune_xy = split_stage_coords(coords, len(idx))

        draw_density_cloud(ax, code_xy, "#f97316", "code_delta", density_bins, trim_percentile, alpha=0.30)
        draw_density_cloud(ax, retune_xy, "#9333ea", "retune_delta", density_bins, trim_percentile, alpha=0.38)
        origin = draw_origin_marker(ax, zero_xy)

        code_centroid = code_xy.mean(axis=0)
        retune_centroid = retune_xy.mean(axis=0)
        ax.annotate("", xy=code_centroid, xytext=origin, arrowprops=dict(arrowstyle="->", color="#f97316", lw=1.5, alpha=0.85))
        ax.annotate("", xy=retune_centroid, xytext=origin, arrowprops=dict(arrowstyle="->", color="#9333ea", lw=1.5, alpha=0.85))
        ax.scatter([code_centroid[0]], [code_centroid[1]], s=30, c="#f97316", edgecolors="#7c2d12", linewidths=0.7)
        ax.scatter([retune_centroid[0]], [retune_centroid[1]], s=30, c="#9333ea", edgecolors="#581c87", linewidths=0.7)

        limit_points = np.concatenate([code_xy, retune_xy, origin[None, :]], axis=0)
        expand_limits(ax, limit_points, trim_percentile=trim_percentile)
        ax.set_title(f"Layer {int(layer_number)} density delta", fontsize=11, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.16)

    for idx in range(n_layers, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(f"{MODEL_LABEL} Hidden Delta Density from Wiki-Only by Layer ({method.upper()})", y=0.995, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=230)
    plt.close(fig)


def plot_delta_density_layer_average(
    dumps,
    out_path: Path,
    zoom_out_path: Path,
    method: str,
    max_points: int,
    seed: int,
    trim_percentile: float,
    density_bins: int,
):
    rng = np.random.default_rng(seed)
    idx = common_subsample_indices(dumps, max_points, rng)
    wiki = dumps[0]["hidden"].mean(axis=0)[idx]
    code_delta = dumps[1]["hidden"].mean(axis=0)[idx] - wiki
    retune_delta = dumps[2]["hidden"].mean(axis=0)[idx] - wiki
    zero = np.zeros_like(code_delta)
    coords = reduce2(np.concatenate([zero, code_delta, retune_delta], axis=0), method, seed + 5999)
    zero_xy, code_xy, retune_xy = split_stage_coords(coords, len(idx))

    def render(path: Path, zoom_retune: bool):
        fig, ax = plt.subplots(figsize=(8.4, 6.8))
        draw_density_cloud(ax, code_xy, "#f97316", "code_delta", density_bins, trim_percentile, alpha=0.30)
        draw_density_cloud(ax, retune_xy, "#9333ea", "retune_delta", density_bins, trim_percentile, alpha=0.42)
        origin = draw_origin_marker(ax, zero_xy)
        code_centroid = code_xy.mean(axis=0)
        retune_centroid = retune_xy.mean(axis=0)
        ax.annotate("", xy=code_centroid, xytext=origin, arrowprops=dict(arrowstyle="->", color="#f97316", lw=1.8, alpha=0.88))
        ax.annotate("", xy=retune_centroid, xytext=origin, arrowprops=dict(arrowstyle="->", color="#9333ea", lw=1.8, alpha=0.88))
        ax.scatter([code_centroid[0]], [code_centroid[1]], s=42, c="#f97316", edgecolors="#7c2d12", linewidths=0.8)
        ax.scatter([retune_centroid[0]], [retune_centroid[1]], s=42, c="#9333ea", edgecolors="#581c87", linewidths=0.8)

        if zoom_retune:
            zoom_points = np.concatenate([retune_xy, origin[None, :]], axis=0)
            xmin, xmax, ymin, ymax = percentile_bounds(zoom_points, 99.4, pad_fraction=0.45)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            subtitle = "Retune/Origin Zoom"
        else:
            limit_points = np.concatenate([code_xy, retune_xy, origin[None, :]], axis=0)
            expand_limits(ax, limit_points, trim_percentile=trim_percentile)
            subtitle = "Full Delta Field"

        ax.set_title(
            f"{MODEL_LABEL} Hidden Delta Density from Wiki-Only, Layer-Average ({method.upper()})\n{subtitle}",
            weight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        fig.savefig(path, dpi=250)
        plt.close(fig)

    render(out_path, zoom_retune=False)
    render(zoom_out_path, zoom_retune=True)


def plot_hidden_density_layer_grid(
    dumps,
    out_path: Path,
    method: str,
    max_points: int,
    seed: int,
    trim_percentile: float,
    density_bins: int,
):
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
        coords = reduce2(np.concatenate(sampled, axis=0), method, seed + 7000 + int(layer_number))
        coords_by_stage = split_stage_coords(coords, len(idx))

        for dump, part in zip(dumps, coords_by_stage):
            color = STAGE_COLORS.get(dump["label"], None)
            draw_density_cloud(ax, part, color, dump["label"], density_bins, trim_percentile, alpha=0.28)

        centroids = np.stack([part.mean(axis=0) for part in coords_by_stage])
        ax.plot(centroids[:, 0], centroids[:, 1], color="#111827", lw=1.25, alpha=0.78)
        ax.scatter(centroids[:, 0], centroids[:, 1], color="#111827", s=22, alpha=0.92, zorder=8)
        expand_limits(ax, coords, trim_percentile=trim_percentile)
        ax.set_title(f"Layer {int(layer_number)} hidden density", fontsize=11, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.16)

    for idx in range(n_layers, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(f"{MODEL_LABEL} Wiki Probe Hidden Density by Layer ({method.upper()})", y=0.995, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=230)
    plt.close(fig)


def plot_hidden_density_layer_average(
    dumps,
    out_path: Path,
    method: str,
    max_points: int,
    seed: int,
    trim_percentile: float,
    density_bins: int,
):
    rng = np.random.default_rng(seed)
    idx = common_subsample_indices(dumps, max_points, rng)
    sampled = [dump["hidden"].mean(axis=0)[idx] for dump in dumps]
    coords = reduce2(np.concatenate(sampled, axis=0), method, seed + 7999)
    coords_by_stage = split_stage_coords(coords, len(idx))

    fig, ax = plt.subplots(figsize=(8.4, 6.8))
    for dump, part in zip(dumps, coords_by_stage):
        color = STAGE_COLORS.get(dump["label"], None)
        draw_density_cloud(ax, part, color, dump["label"], density_bins, trim_percentile, alpha=0.30)

    centroids = np.stack([part.mean(axis=0) for part in coords_by_stage])
    ax.plot(centroids[:, 0], centroids[:, 1], color="#111827", lw=1.8, alpha=0.80)
    ax.scatter(centroids[:, 0], centroids[:, 1], color="#111827", s=36, alpha=0.94, zorder=8)
    for dump, centroid in zip(dumps, centroids):
        ax.text(
            centroid[0],
            centroid[1],
            f" {dump['label']}",
            fontsize=9,
            color="#111827",
            weight="bold",
            alpha=0.82,
        )

    expand_limits(ax, coords, trim_percentile=trim_percentile)
    ax.set_title(f"{MODEL_LABEL} Wiki Probe Hidden Density, Layer-Average ({method.upper()})", weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_hidden_pairwise_density_layer_average(
    dumps,
    out_path: Path,
    method: str,
    max_points: int,
    seed: int,
    trim_percentile: float,
    density_bins: int,
):
    rng = np.random.default_rng(seed)
    idx = common_subsample_indices(dumps, max_points, rng)
    sampled = [dump["hidden"].mean(axis=0)[idx] for dump in dumps]
    coords = reduce2(np.concatenate(sampled, axis=0), method, seed + 8999)
    coords_by_stage = split_stage_coords(coords, len(idx))
    wiki_xy, code_xy, retune_xy = coords_by_stage

    pairs = [
        ("Wiki-only vs Code-trained", dumps[1]["label"], code_xy, "#f97316"),
        ("Wiki-only vs Router-retuned", dumps[2]["label"], retune_xy, "#9333ea"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), sharex=True, sharey=True)
    all_points = np.concatenate([wiki_xy, code_xy, retune_xy], axis=0)
    xmin, xmax, ymin, ymax = percentile_bounds(all_points, trim_percentile, pad_fraction=0.10)

    for ax, (title, other_label, other_xy, other_color) in zip(axes, pairs):
        draw_density_cloud(ax, wiki_xy, "#2563eb", dumps[0]["label"], density_bins, trim_percentile, alpha=0.24)
        draw_density_cloud(ax, other_xy, other_color, other_label, density_bins, trim_percentile, alpha=0.34)
        wiki_centroid = wiki_xy.mean(axis=0)
        other_centroid = other_xy.mean(axis=0)
        ax.annotate(
            "",
            xy=other_centroid,
            xytext=wiki_centroid,
            arrowprops=dict(arrowstyle="->", color="#111827", lw=2.0, alpha=0.82),
        )
        ax.scatter([wiki_centroid[0]], [wiki_centroid[1]], s=48, c="#2563eb", edgecolors="#1e3a8a", linewidths=0.8, zorder=8)
        ax.scatter([other_centroid[0]], [other_centroid[1]], s=48, c=other_color, edgecolors="#111827", linewidths=0.8, zorder=8)
        ax.set_title(title, weight="bold")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.16)
        ax.legend(frameon=False, loc="upper left")

    fig.suptitle(f"{MODEL_LABEL} Wiki Probe Hidden Density, Pairwise Layer-Average ({method.upper()})", weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_hidden_pairwise_density_layer_average_by_base(
    dumps,
    out_path: Path,
    method: str,
    max_points: int,
    seed: int,
    trim_percentile: float,
    density_bins: int,
    probe_task: str,
    base_label: str,
):
    label_to_idx = stage_index_by_label(dumps)
    compare_labels = pairwise_compare_labels(base_label)
    missing = [label for label in [base_label, *compare_labels] if label not in label_to_idx]
    if missing:
        raise SystemExit(f"Missing hidden dump label(s) for pairwise plot: {missing}")

    rng = np.random.default_rng(seed)
    idx = common_subsample_indices(dumps, max_points, rng)
    sampled = [dump["hidden"].mean(axis=0)[idx] for dump in dumps]
    coords = reduce2(np.concatenate(sampled, axis=0), method, seed + 9199)
    coords_by_stage = split_stage_coords(coords, len(idx))

    base_xy = coords_by_stage[label_to_idx[base_label]]
    all_points = np.concatenate(
        [coords_by_stage[label_to_idx[label]] for label in [base_label, *compare_labels]],
        axis=0,
    )
    limits = percentile_bounds(all_points, trim_percentile, pad_fraction=0.10)

    fig, axes = plt.subplots(1, len(compare_labels), figsize=(6.6 * len(compare_labels), 5.8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    metric_rows = []
    for ax, other_label in zip(axes, compare_labels):
        other_xy = coords_by_stage[label_to_idx[other_label]]
        overlap = draw_pairwise_density_panel(
            ax,
            base_xy,
            other_xy,
            base_label,
            other_label,
            density_bins,
            trim_percentile,
            limits=limits,
            show_legend=True,
        )
        metric_rows.append(
            {
                "plot": str(out_path),
                "model_label": MODEL_LABEL,
                "probe_task": normalize_probe_task(probe_task),
                "scope": "layer_average",
                "layer": "average",
                "base": base_label,
                "other": other_label,
                "method": method,
                "kde_overlap": f"{overlap:.6f}" if np.isfinite(overlap) else "nan",
            }
        )
        ax.set_title(
            f"{display_stage_name(base_label)} vs {display_stage_name(other_label)}",
            weight="bold",
        )

    fig.suptitle(
        f"{MODEL_LABEL} {probe_display_name(probe_task)} Hidden Density, Pairwise Layer-Average ({method.upper()})",
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)
    write_kde_overlap_metrics(out_path, metric_rows)


def plot_hidden_pairwise_density_layers_by_base(
    dumps,
    out_path: Path,
    method: str,
    max_points: int,
    seed: int,
    trim_percentile: float,
    density_bins: int,
    probe_task: str,
    base_label: str,
):
    label_to_idx = stage_index_by_label(dumps)
    compare_labels = pairwise_compare_labels(base_label)
    missing = [label for label in [base_label, *compare_labels] if label not in label_to_idx]
    if missing:
        raise SystemExit(f"Missing hidden dump label(s) for pairwise plot: {missing}")

    rng = np.random.default_rng(seed)
    layers = dumps[0]["layers"]
    visible_layers = [(idx, int(layer)) for idx, layer in enumerate(layers) if int(layer) != 1]
    midpoint = math.ceil(len(visible_layers) / 2)
    layer_groups = [visible_layers[:midpoint], visible_layers[midpoint:]]
    nrows = max(len(group) for group in layer_groups)
    ncols = len(compare_labels) * len(layer_groups)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.55 * ncols, 3.10 * nrows),
        squeeze=False,
    )
    metric_rows = []

    for row_idx in range(nrows):
        for group_idx, group in enumerate(layer_groups):
            if row_idx >= len(group):
                for col_idx in range(len(compare_labels)):
                    axes[row_idx][group_idx * len(compare_labels) + col_idx].axis("off")
                continue

            layer_idx, layer_number = group[row_idx]
            idx = common_subsample_indices(dumps, max_points, rng)
            sampled = [dump["hidden"][layer_idx][idx] for dump in dumps]
            coords = reduce2(np.concatenate(sampled, axis=0), method, seed + 9300 + int(layer_number))
            coords_by_stage = split_stage_coords(coords, len(idx))

            base_xy = coords_by_stage[label_to_idx[base_label]]
            all_points = np.concatenate(
                [coords_by_stage[label_to_idx[label]] for label in [base_label, *compare_labels]],
                axis=0,
            )
            limits = percentile_bounds(all_points, trim_percentile, pad_fraction=0.12)

            for col_idx, other_label in enumerate(compare_labels):
                ax = axes[row_idx][group_idx * len(compare_labels) + col_idx]
                other_xy = coords_by_stage[label_to_idx[other_label]]
                overlap = draw_pairwise_density_panel(
                    ax,
                    base_xy,
                    other_xy,
                    base_label,
                    other_label,
                    density_bins,
                    trim_percentile,
                    limits=limits,
                    show_legend=False,
                )
                metric_rows.append(
                    {
                        "plot": str(out_path),
                        "model_label": MODEL_LABEL,
                        "probe_task": normalize_probe_task(probe_task),
                        "scope": "layer",
                        "layer": layer_number,
                        "base": base_label,
                        "other": other_label,
                        "method": method,
                        "kde_overlap": f"{overlap:.6f}" if np.isfinite(overlap) else "nan",
                    }
                )
                ax.set_title("")
                if col_idx == 0:
                    ax.text(
                        -0.075,
                        0.5,
                        f"Layer {layer_number}",
                        transform=ax.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=14,
                        weight="bold",
                    )

    bottom_labels = [
        f"{display_stage_name(base_label)} vs {display_stage_name(other_label)}"
        for _group in layer_groups
        for other_label in compare_labels
    ]
    for ax, label in zip(axes[-1], bottom_labels):
        ax.set_xlabel(label, fontsize=10, weight="bold", labelpad=8)

    fig.suptitle(
        f"{MODEL_LABEL} {probe_display_name(probe_task)} Hidden Density by Layer, Pairwise ({method.upper()})",
        y=0.995,
        fontsize=16,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0.018, 1, 0.975))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    write_kde_overlap_metrics(out_path, metric_rows)


def plot_hidden_small_multiples_layer_average(
    dumps,
    out_path: Path,
    method: str,
    max_points: int,
    seed: int,
    trim_percentile: float,
    density_bins: int,
):
    rng = np.random.default_rng(seed)
    idx = common_subsample_indices(dumps, max_points, rng)
    sampled = [dump["hidden"].mean(axis=0)[idx] for dump in dumps]
    coords = reduce2(np.concatenate(sampled, axis=0), method, seed + 9999)
    coords_by_stage = split_stage_coords(coords, len(idx))

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2), sharex=True, sharey=True)
    all_points = np.concatenate(coords_by_stage, axis=0)
    xmin, xmax, ymin, ymax = percentile_bounds(all_points, trim_percentile, pad_fraction=0.10)
    wiki_centroid = coords_by_stage[0].mean(axis=0)

    for ax, dump, part in zip(axes, dumps, coords_by_stage):
        color = STAGE_COLORS.get(dump["label"], None)
        # Gray wiki reference makes the displacement readable without overplotting all three clouds.
        if dump["label"] != "wiki_only":
            draw_density_cloud(ax, coords_by_stage[0], "#94a3b8", "wiki_ref", density_bins, trim_percentile, alpha=0.18)
        draw_density_cloud(ax, part, color, dump["label"], density_bins, trim_percentile, alpha=0.40)
        centroid = part.mean(axis=0)
        if dump["label"] != "wiki_only":
            ax.annotate(
                "",
                xy=centroid,
                xytext=wiki_centroid,
                arrowprops=dict(arrowstyle="->", color="#111827", lw=1.8, alpha=0.78),
            )
        ax.scatter([centroid[0]], [centroid[1]], s=54, c=color, edgecolors="#111827", linewidths=0.8, zorder=8)
        ax.set_title(dump["label"], weight="bold", color=color)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.16)
        ax.legend(frameon=False, loc="upper left")

    fig.suptitle(f"{MODEL_LABEL} Wiki Probe Hidden Density, Small Multiples Layer-Average ({method.upper()})", weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_delta_layer_grid(
    dumps,
    out_path: Path,
    method: str,
    max_points: int,
    max_vectors: int,
    seed: int,
    trim_percentile: float,
):
    rng = np.random.default_rng(seed)
    layers = dumps[0]["layers"]
    n_layers = len(layers)
    ncols = min(3, n_layers)
    nrows = math.ceil(n_layers / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows), squeeze=False)

    for layer_idx, layer_number in enumerate(layers):
        ax = axes[layer_idx // ncols][layer_idx % ncols]
        idx = common_subsample_indices(dumps, max_points, rng)
        wiki = dumps[0]["hidden"][layer_idx][idx]
        code_delta = dumps[1]["hidden"][layer_idx][idx] - wiki
        retune_delta = dumps[2]["hidden"][layer_idx][idx] - wiki
        zero = np.zeros_like(code_delta)
        coords = reduce2(np.concatenate([zero, code_delta, retune_delta], axis=0), method, seed + 3000 + int(layer_number))
        zero_xy, code_xy, retune_xy = split_stage_coords(coords, len(idx))
        vector_idx = subsample_indices(len(idx), min(max_vectors, len(idx)), rng)

        for i in vector_idx:
            ax.plot(
                [zero_xy[i, 0], code_xy[i, 0]],
                [zero_xy[i, 1], code_xy[i, 1]],
                color="#f97316",
                alpha=0.16,
                lw=0.45,
            )
            ax.plot(
                [zero_xy[i, 0], retune_xy[i, 0]],
                [zero_xy[i, 1], retune_xy[i, 1]],
                color="#9333ea",
                alpha=0.16,
                lw=0.45,
            )

        ax.scatter(zero_xy[:, 0], zero_xy[:, 1], s=5, alpha=0.22, c="#2563eb", label="wiki_origin")
        ax.scatter(code_xy[:, 0], code_xy[:, 1], s=9, alpha=0.46, c="#f97316", label="code_delta")
        ax.scatter(retune_xy[:, 0], retune_xy[:, 1], s=9, alpha=0.46, c="#9333ea", label="retune_delta")
        centroids = np.stack([zero_xy.mean(axis=0), code_xy.mean(axis=0), retune_xy.mean(axis=0)])
        ax.plot(centroids[:, 0], centroids[:, 1], color="#111827", lw=1.2, alpha=0.78)
        ax.scatter(centroids[:, 0], centroids[:, 1], color="#111827", s=18, alpha=0.9)
        expand_limits(ax, coords, trim_percentile=trim_percentile)
        ax.set_title(f"Layer {int(layer_number)} delta from wiki", fontsize=11, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.16)

    for idx in range(n_layers, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(f"{MODEL_LABEL} Hidden Delta from Wiki-Only by Layer ({method.upper()})", y=0.995, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=230)
    plt.close(fig)


def plot_delta_layer_average(
    dumps,
    out_path: Path,
    method: str,
    max_points: int,
    max_vectors: int,
    seed: int,
    trim_percentile: float,
):
    rng = np.random.default_rng(seed)
    idx = common_subsample_indices(dumps, max_points, rng)
    wiki = dumps[0]["hidden"].mean(axis=0)[idx]
    code_delta = dumps[1]["hidden"].mean(axis=0)[idx] - wiki
    retune_delta = dumps[2]["hidden"].mean(axis=0)[idx] - wiki
    zero = np.zeros_like(code_delta)
    coords = reduce2(np.concatenate([zero, code_delta, retune_delta], axis=0), method, seed + 3999)
    zero_xy, code_xy, retune_xy = split_stage_coords(coords, len(idx))
    vector_idx = subsample_indices(len(idx), min(max_vectors, len(idx)), rng)

    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    for i in vector_idx:
        ax.plot(
            [zero_xy[i, 0], code_xy[i, 0]],
            [zero_xy[i, 1], code_xy[i, 1]],
            color="#f97316",
            alpha=0.17,
            lw=0.55,
        )
        ax.plot(
            [zero_xy[i, 0], retune_xy[i, 0]],
            [zero_xy[i, 1], retune_xy[i, 1]],
            color="#9333ea",
            alpha=0.17,
            lw=0.55,
        )
    ax.scatter(zero_xy[:, 0], zero_xy[:, 1], s=6, alpha=0.24, c="#2563eb", label="wiki_origin")
    ax.scatter(code_xy[:, 0], code_xy[:, 1], s=12, alpha=0.50, c="#f97316", label="code_delta")
    ax.scatter(retune_xy[:, 0], retune_xy[:, 1], s=12, alpha=0.50, c="#9333ea", label="retune_delta")
    centroids = np.stack([zero_xy.mean(axis=0), code_xy.mean(axis=0), retune_xy.mean(axis=0)])
    ax.plot(centroids[:, 0], centroids[:, 1], color="#111827", lw=1.5, alpha=0.80)
    ax.scatter(centroids[:, 0], centroids[:, 1], color="#111827", s=28, alpha=0.94)
    expand_limits(ax, coords, trim_percentile=trim_percentile)
    ax.set_title(f"{MODEL_LABEL} Hidden Delta from Wiki-Only, Layer-Average ({method.upper()})", weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


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
    fig.suptitle(f"{MODEL_LABEL} Wiki Probe Hidden Space by Layer ({method.upper()})", y=0.995, fontsize=16)
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
    ax.set_title(f"{MODEL_LABEL} Wiki Probe Hidden Space, Layer-Average ({method.upper()})", weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def main():
    global MODEL_LABEL
    args = parse_args()
    MODEL_LABEL = args.model_label
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_task = normalize_probe_task(args.probe_task)
    pairwise_base_label = pairwise_base_label_for_probe(probe_task)

    dumps = [
        load_dump(args.wiki_only, "wiki_only"),
        load_dump(args.code_trained, "code_trained"),
        load_dump(args.router_retuned, "router_retuned"),
    ]
    layer_ref = dumps[0]["layers"].tolist()
    for dump in dumps[1:]:
        if dump["layers"].tolist() != layer_ref:
            raise SystemExit("Layer numbers do not match across hidden dumps.")

    if args.only_pairwise_density:
        plot_hidden_pairwise_density_layer_average_by_base(
            dumps,
            out_dir / f"ffn_only_hidden_pairwise_density_{probe_task}_probe_layer_average_{args.method}.png",
            args.method,
            args.max_points_per_stage,
            args.seed,
            args.trim_percentile,
            args.density_bins,
            probe_task,
            pairwise_base_label,
        )
        plot_hidden_pairwise_density_layers_by_base(
            dumps,
            out_dir / f"ffn_only_hidden_pairwise_density_{probe_task}_probe_layers_{args.method}.png",
            args.method,
            args.max_points_per_stage,
            args.seed,
            args.trim_percentile,
            args.density_bins,
            probe_task,
            pairwise_base_label,
        )
        return

    plot_layer_grid(
        dumps,
        out_dir / f"ffn_only_hidden_layers_{args.method}.png",
        args.method,
        args.max_points_per_stage,
        args.max_vectors,
        args.seed,
    )
    plot_hidden_density_layer_grid(
        dumps,
        out_dir / f"ffn_only_hidden_density_layers_{args.method}.png",
        args.method,
        args.max_points_per_stage,
        args.seed,
        args.trim_percentile,
        args.density_bins,
    )
    plot_delta_layer_grid(
        dumps,
        out_dir / f"ffn_only_hidden_delta_layers_{args.method}.png",
        args.method,
        args.max_points_per_stage,
        args.max_vectors,
        args.seed,
        args.trim_percentile,
    )
    plot_delta_density_layer_grid(
        dumps,
        out_dir / f"ffn_only_hidden_delta_density_layers_{args.method}.png",
        args.method,
        args.max_points_per_stage,
        args.seed,
        args.trim_percentile,
        args.density_bins,
    )
    plot_delta_layer_average(
        dumps,
        out_dir / f"ffn_only_hidden_delta_layer_average_{args.method}.png",
        args.method,
        args.max_points_per_stage,
        args.max_vectors,
        args.seed,
        args.trim_percentile,
    )
    plot_delta_density_layer_average(
        dumps,
        out_dir / f"ffn_only_hidden_delta_density_layer_average_{args.method}.png",
        out_dir / f"ffn_only_hidden_delta_density_layer_average_zoom_{args.method}.png",
        args.method,
        args.max_points_per_stage,
        args.seed,
        args.trim_percentile,
        args.density_bins,
    )
    plot_layer_average(
        dumps,
        out_dir / f"ffn_only_hidden_layer_average_{args.method}.png",
        args.method,
        args.max_points_per_stage,
        args.max_vectors,
        args.seed,
    )
    plot_hidden_density_layer_average(
        dumps,
        out_dir / f"ffn_only_hidden_density_layer_average_{args.method}.png",
        args.method,
        args.max_points_per_stage,
        args.seed,
        args.trim_percentile,
        args.density_bins,
    )
    plot_hidden_pairwise_density_layer_average(
        dumps,
        out_dir / f"ffn_only_hidden_pairwise_density_layer_average_{args.method}.png",
        args.method,
        args.max_points_per_stage,
        args.seed,
        args.trim_percentile,
        args.density_bins,
    )
    plot_hidden_pairwise_density_layer_average_by_base(
        dumps,
        out_dir / f"ffn_only_hidden_pairwise_density_{probe_task}_probe_layer_average_{args.method}.png",
        args.method,
        args.max_points_per_stage,
        args.seed,
        args.trim_percentile,
        args.density_bins,
        probe_task,
        pairwise_base_label,
    )
    plot_hidden_pairwise_density_layers_by_base(
        dumps,
        out_dir / f"ffn_only_hidden_pairwise_density_{probe_task}_probe_layers_{args.method}.png",
        args.method,
        args.max_points_per_stage,
        args.seed,
        args.trim_percentile,
        args.density_bins,
        probe_task,
        pairwise_base_label,
    )
    plot_hidden_small_multiples_layer_average(
        dumps,
        out_dir / f"ffn_only_hidden_small_multiples_layer_average_{args.method}.png",
        args.method,
        args.max_points_per_stage,
        args.seed,
        args.trim_percentile,
        args.density_bins,
    )

    metrics = collect_metrics(dumps)
    (out_dir / "ffn_only_hidden_space_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[DONE] wrote plots and metrics to {out_dir}")


if __name__ == "__main__":
    main()
