#!/usr/bin/env python3
"""Plot FFN MoE router row trajectories across wiki/code/retuned checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


STAGES = [
    ("wiki_only", "#2563eb", "o"),
    ("code_trained", "#f97316", "s"),
    ("router_retuned", "#16a34a", "^"),
]


def ensure_megatron_on_path():
    repo_root = Path(__file__).resolve().parents[2]
    megatron_path = repo_root / "Megatron-LM"
    if megatron_path.exists():
        path = str(megatron_path)
        if path not in sys.path:
            sys.path.insert(0, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-only", required=True, help="Wiki-only checkpoint root or iter_* directory.")
    parser.add_argument("--code-trained", required=True, help="Code-trained checkpoint root or iter_* directory.")
    parser.add_argument("--router-retuned", required=True, help="Router-retuned checkpoint root or iter_* directory.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--router-key-regex",
        default=r"(^|\.)mlp\.router\.weight$",
        help="Regex for router tensor keys. Default targets FFN MoE router weights only.",
    )
    parser.add_argument("--old-experts", type=int, default=8)
    parser.add_argument("--display-layer-offset", type=int, default=1)
    parser.add_argument("--normalize-rows-for-pca", action="store_true")
    parser.add_argument("--plot-pca-trajectory", action="store_true", help="Also write the dense PCA trajectory diagnostic plot.")
    parser.add_argument("--list-keys", action="store_true", help="Print matched router keys before plotting.")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def is_dcp_dir(path: Path):
    return (path / ".metadata").exists() or any(path.glob("*.distcp"))


def read_latest_iteration(root: Path):
    tracker = root / "latest_checkpointed_iteration.txt"
    if not tracker.exists():
        return None
    text = tracker.read_text(encoding="utf-8").strip()
    if text == "release":
        return "release"
    return int(text)


def resolve_checkpoint_dir(path_like: str):
    path = Path(path_like)
    if is_dcp_dir(path):
        return path

    iteration = read_latest_iteration(path)
    if iteration == "release":
        candidate = path / "release"
        if is_dcp_dir(candidate):
            return candidate
    elif isinstance(iteration, int):
        candidate = path / f"iter_{iteration:07d}"
        if is_dcp_dir(candidate):
            return candidate

    candidates = [p for p in path.glob("iter_*") if p.is_dir() and is_dcp_dir(p)]
    if candidates:
        return sorted(candidates)[-1]

    raise FileNotFoundError(f"Could not resolve a torch_dist checkpoint directory from {path}")


def tensor_shape_from_metadata(meta):
    for attr in ("size", "shape"):
        value = getattr(meta, attr, None)
        if value is not None:
            return tuple(value)
    raise ValueError(f"Could not infer tensor shape from metadata object: {meta}")


def tensor_dtype_from_metadata(meta, torch):
    props = getattr(meta, "properties", None)
    dtype = getattr(props, "dtype", None) if props is not None else None
    return dtype or torch.float32


def load_dcp_tensors(ckpt_dir: Path, keys):
    ensure_megatron_on_path()
    import torch
    from torch.distributed.checkpoint import FileSystemReader

    reader = FileSystemReader(str(ckpt_dir))
    metadata = reader.read_metadata()
    state = {}
    for key in keys:
        meta = metadata.state_dict_metadata[key]
        state[key] = torch.empty(tensor_shape_from_metadata(meta), dtype=tensor_dtype_from_metadata(meta, torch))

    try:
        from torch.distributed.checkpoint import load_state_dict

        load_state_dict(state, storage_reader=reader, no_dist=True)
    except TypeError:
        import torch.distributed.checkpoint as dcp

        dcp.load(state, checkpoint_id=str(ckpt_dir))

    return {key: value.detach().cpu().float().numpy() for key, value in state.items()}


def matched_router_keys(ckpt_dir: Path, key_regex: str):
    ensure_megatron_on_path()
    from torch.distributed.checkpoint import FileSystemReader

    reader = FileSystemReader(str(ckpt_dir))
    metadata = reader.read_metadata()
    pattern = re.compile(key_regex)
    keys = []
    for key, meta in metadata.state_dict_metadata.items():
        if not hasattr(meta, "size") and not hasattr(meta, "shape"):
            continue
        if pattern.search(key):
            keys.append(key)
    return sorted(keys)


def layer_id_from_key(key: str):
    match = re.search(r"(?:decoder|encoder)\.layers\.(\d+)", key)
    if match:
        return int(match.group(1))
    match = re.search(r"layers\.(\d+)", key)
    if match:
        return int(match.group(1))
    return None


def load_router_weights(label: str, path_like: str, key_regex: str, list_keys: bool):
    ckpt_dir = resolve_checkpoint_dir(path_like)
    keys = matched_router_keys(ckpt_dir, key_regex)
    if not keys:
        raise RuntimeError(f"No router keys matched {key_regex!r} in {ckpt_dir}")
    if list_keys:
        print(f"\n[{label}] {ckpt_dir}")
        for key in keys:
            print(f"  {key}")

    tensors = load_dcp_tensors(ckpt_dir, keys)
    by_layer = {}
    for fallback_idx, key in enumerate(keys):
        layer_id = layer_id_from_key(key)
        if layer_id is None:
            layer_id = fallback_idx
        value = tensors[key]
        if value.ndim != 2:
            print(f"[WARN] skipping non-2D router tensor: {key} shape={value.shape}")
            continue
        by_layer[layer_id] = {
            "key": key,
            "weight": value,
        }
    return {
        "label": label,
        "checkpoint_dir": str(ckpt_dir),
        "by_layer": by_layer,
    }


def pca2(x: np.ndarray):
    x = x.astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def normalize_rows(x: np.ndarray):
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, 1e-8)


def split_coords(coords: np.ndarray, counts):
    out = []
    start = 0
    for count in counts:
        out.append(coords[start : start + count])
        start += count
    return out


def row_cosine(a: np.ndarray, b: np.ndarray):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / max(denom, 1e-8))


def collect_metrics(stages, layers, old_experts: int):
    metrics = {"layers": []}
    for layer in layers:
        weights = {stage["label"]: stage["by_layer"][layer]["weight"] for stage in stages}
        row = {"layer": int(layer)}
        wiki = weights["wiki_only"]
        code = weights["code_trained"]
        retune = weights["router_retuned"]
        n_old = min(old_experts, wiki.shape[0], code.shape[0], retune.shape[0])

        def summarize_pair(prefix, a, b, n_rows):
            cosines = [row_cosine(a[i], b[i]) for i in range(n_rows)]
            l2s = [float(np.linalg.norm(a[i] - b[i])) for i in range(n_rows)]
            row[f"{prefix}/mean_row_cosine"] = float(np.mean(cosines))
            row[f"{prefix}/mean_row_l2"] = float(np.mean(l2s))
            row[f"{prefix}/centroid_l2"] = float(np.linalg.norm(a[:n_rows].mean(axis=0) - b[:n_rows].mean(axis=0)))

        summarize_pair("old/wiki_vs_code", wiki, code, n_old)
        summarize_pair("old/wiki_vs_retune", wiki, retune, n_old)
        summarize_pair("old/code_vs_retune", code, retune, n_old)

        n_new = max(min(code.shape[0], retune.shape[0]) - old_experts, 0)
        if n_new > 0:
            summarize_pair("new/code_vs_retune", code[old_experts:], retune[old_experts:], n_new)

        metrics["layers"].append(row)
    return metrics


def transition_matrices(stages, layers, old_experts: int):
    weights = {stage["label"]: stage["by_layer"] for stage in stages}
    max_rows = max(
        weights[label][layer]["weight"].shape[0]
        for label in weights
        for layer in layers
    )
    n_old = min(
        old_experts,
        min(weights[label][layer]["weight"].shape[0] for label in weights for layer in layers),
    )
    n_new = max(
        [0]
        + [
            min(
                weights["code_trained"][layer]["weight"].shape[0],
                weights["router_retuned"][layer]["weight"].shape[0],
            )
            - old_experts
            for layer in layers
        ]
    )

    def empty(rows):
        return np.full((rows, len(layers)), np.nan, dtype=np.float32)

    mats = {
        "old_l2/wiki_to_code": empty(n_old),
        "old_l2/wiki_to_retune": empty(n_old),
        "old_l2/code_to_retune": empty(n_old),
        "old_cosine/wiki_to_code": empty(n_old),
        "old_cosine/wiki_to_retune": empty(n_old),
        "old_cosine/code_to_retune": empty(n_old),
        "new_l2/code_to_retune": empty(n_new),
        "new_cosine/code_to_retune": empty(n_new),
    }

    for layer_idx, layer in enumerate(layers):
        wiki = weights["wiki_only"][layer]["weight"]
        code = weights["code_trained"][layer]["weight"]
        retune = weights["router_retuned"][layer]["weight"]
        old_rows = min(n_old, wiki.shape[0], code.shape[0], retune.shape[0])
        for expert_id in range(old_rows):
            pairs = {
                "wiki_to_code": (wiki[expert_id], code[expert_id]),
                "wiki_to_retune": (wiki[expert_id], retune[expert_id]),
                "code_to_retune": (code[expert_id], retune[expert_id]),
            }
            for name, (a, b) in pairs.items():
                mats[f"old_l2/{name}"][expert_id, layer_idx] = float(np.linalg.norm(a - b))
                mats[f"old_cosine/{name}"][expert_id, layer_idx] = row_cosine(a, b)

        new_rows = min(n_new, code.shape[0] - old_experts, retune.shape[0] - old_experts)
        for local_id in range(max(new_rows, 0)):
            expert_id = old_experts + local_id
            a, b = code[expert_id], retune[expert_id]
            mats["new_l2/code_to_retune"][local_id, layer_idx] = float(np.linalg.norm(a - b))
            mats["new_cosine/code_to_retune"][local_id, layer_idx] = row_cosine(a, b)

    mats["metadata/max_rows"] = max_rows
    return mats


def geometry_matrices(stages, layers, old_experts: int):
    weights = {stage["label"]: stage["by_layer"] for stage in stages}
    n_layers = len(layers)
    out = {
        "old_norm/wiki_only": np.full((old_experts, n_layers), np.nan, dtype=np.float32),
        "old_norm/code_trained": np.full((old_experts, n_layers), np.nan, dtype=np.float32),
        "old_norm/router_retuned": np.full((old_experts, n_layers), np.nan, dtype=np.float32),
        "new_norm/code_trained": None,
        "new_norm/router_retuned": None,
        "old_new_centroid_l2/code_trained": np.full(n_layers, np.nan, dtype=np.float32),
        "old_new_centroid_l2/router_retuned": np.full(n_layers, np.nan, dtype=np.float32),
        "old_new_mean_cosine/code_trained": np.full(n_layers, np.nan, dtype=np.float32),
        "old_new_mean_cosine/router_retuned": np.full(n_layers, np.nan, dtype=np.float32),
        "old_new_max_cosine/code_trained": np.full(n_layers, np.nan, dtype=np.float32),
        "old_new_max_cosine/router_retuned": np.full(n_layers, np.nan, dtype=np.float32),
        "norm_ratio_new_over_old/code_trained": np.full(n_layers, np.nan, dtype=np.float32),
        "norm_ratio_new_over_old/router_retuned": np.full(n_layers, np.nan, dtype=np.float32),
        "new_mean_norm_delta/retune_minus_code": np.full(n_layers, np.nan, dtype=np.float32),
        "old_mean_norm_delta/retune_minus_code": np.full(n_layers, np.nan, dtype=np.float32),
    }

    max_new_rows = max(
        [0]
        + [
            weights[label][layer]["weight"].shape[0] - old_experts
            for label in ("code_trained", "router_retuned")
            for layer in layers
        ]
    )
    out["new_norm/code_trained"] = np.full((max_new_rows, n_layers), np.nan, dtype=np.float32)
    out["new_norm/router_retuned"] = np.full((max_new_rows, n_layers), np.nan, dtype=np.float32)

    for layer_idx, layer in enumerate(layers):
        for label in ("wiki_only", "code_trained", "router_retuned"):
            w = weights[label][layer]["weight"]
            old = w[: min(old_experts, w.shape[0])]
            if old.size:
                out[f"old_norm/{label}"][: old.shape[0], layer_idx] = np.linalg.norm(old, axis=1)

            if label in ("code_trained", "router_retuned") and w.shape[0] > old_experts:
                new = w[old_experts:]
                out[f"new_norm/{label}"][: new.shape[0], layer_idx] = np.linalg.norm(new, axis=1)

                old_centroid = old.mean(axis=0)
                new_centroid = new.mean(axis=0)
                out[f"old_new_centroid_l2/{label}"][layer_idx] = float(np.linalg.norm(old_centroid - new_centroid))

                old_unit = normalize_rows(old)
                new_unit = normalize_rows(new)
                pair_cos = old_unit @ new_unit.T
                out[f"old_new_mean_cosine/{label}"][layer_idx] = float(np.mean(pair_cos))
                out[f"old_new_max_cosine/{label}"][layer_idx] = float(np.max(pair_cos))
                out[f"norm_ratio_new_over_old/{label}"][layer_idx] = float(
                    np.mean(np.linalg.norm(new, axis=1)) / max(np.mean(np.linalg.norm(old, axis=1)), 1e-8)
                )

        code_old_norm = np.nanmean(out["old_norm/code_trained"][:, layer_idx])
        retune_old_norm = np.nanmean(out["old_norm/router_retuned"][:, layer_idx])
        out["old_mean_norm_delta/retune_minus_code"][layer_idx] = retune_old_norm - code_old_norm

        code_new_norm = np.nanmean(out["new_norm/code_trained"][:, layer_idx])
        retune_new_norm = np.nanmean(out["new_norm/router_retuned"][:, layer_idx])
        out["new_mean_norm_delta/retune_minus_code"][layer_idx] = retune_new_norm - code_new_norm

    return out


def imshow_heatmap(ax, matrix, title, layers, expert_offset, cmap, vmin=None, vmax=None, cbar_label=""):
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels([str(layer + 1) for layer in layers], fontsize=8)
    ax.set_xlabel("Layer")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([str(expert_offset + idx) for idx in range(matrix.shape[0])], fontsize=8)
    ax.set_ylabel("Expert row")
    ax.grid(False)
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=9)
    return image


def plot_router_geometry_dashboard(stages, layers, out_dir: Path, old_experts: int):
    geo = geometry_matrices(stages, layers, old_experts)
    x = np.arange(len(layers))
    layer_labels = [str(layer + 1) for layer in layers]
    width = 0.35

    old_code_norm = np.nanmean(geo["old_norm/code_trained"], axis=0)
    old_retune_norm = np.nanmean(geo["old_norm/router_retuned"], axis=0)
    new_code_norm = np.nanmean(geo["new_norm/code_trained"], axis=0)
    new_retune_norm = np.nanmean(geo["new_norm/router_retuned"], axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.4))

    axes[0][0].bar(x - width / 2, old_code_norm, width, color="#64748b", alpha=0.88, label="old rows after code")
    axes[0][0].bar(x + width / 2, new_code_norm, width, color="#f97316", alpha=0.88, label="new rows after code")
    axes[0][0].set_title("Before retune: router row norm balance", weight="bold")
    axes[0][0].set_ylabel("Mean row norm")
    axes[0][0].set_xticks(x)
    axes[0][0].set_xticklabels(layer_labels)
    axes[0][0].grid(axis="y", alpha=0.24)
    axes[0][0].legend(frameon=False)

    axes[0][1].bar(x - width / 2, old_retune_norm, width, color="#64748b", alpha=0.88, label="old rows after retune")
    axes[0][1].bar(x + width / 2, new_retune_norm, width, color="#16a34a", alpha=0.88, label="new rows after retune")
    axes[0][1].set_title("After retune: router row norm balance", weight="bold")
    axes[0][1].set_ylabel("Mean row norm")
    axes[0][1].set_xticks(x)
    axes[0][1].set_xticklabels(layer_labels)
    axes[0][1].grid(axis="y", alpha=0.24)
    axes[0][1].legend(frameon=False)

    axes[1][0].plot(
        x,
        geo["norm_ratio_new_over_old/code_trained"],
        marker="o",
        lw=2.2,
        color="#f97316",
        label="after code",
    )
    axes[1][0].plot(
        x,
        geo["norm_ratio_new_over_old/router_retuned"],
        marker="o",
        lw=2.2,
        color="#16a34a",
        label="after retune",
    )
    axes[1][0].axhline(1.0, color="#94a3b8", lw=1.1, linestyle="--")
    axes[1][0].set_title("New/old norm ratio", weight="bold")
    axes[1][0].set_ylabel("mean ||new row|| / mean ||old row||")
    axes[1][0].set_xticks(x)
    axes[1][0].set_xticklabels(layer_labels)
    axes[1][0].grid(alpha=0.24)
    axes[1][0].legend(frameon=False)

    axes[1][1].plot(
        x,
        geo["old_new_centroid_l2/code_trained"],
        marker="o",
        lw=2.2,
        color="#f97316",
        label="after code",
    )
    axes[1][1].plot(
        x,
        geo["old_new_centroid_l2/router_retuned"],
        marker="o",
        lw=2.2,
        color="#16a34a",
        label="after retune",
    )
    axes[1][1].set_title("Old/new router row separation", weight="bold")
    axes[1][1].set_ylabel("centroid L2 distance")
    axes[1][1].set_xticks(x)
    axes[1][1].set_xticklabels(layer_labels)
    axes[1][1].grid(alpha=0.24)
    axes[1][1].legend(frameon=False)

    fig.suptitle(
        "FFN Router Weight Geometry\n"
        "What router weights can show in the freeze-old-rows setting",
        fontsize=17,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out_dir / "ffn_only_router_weight_geometry_dashboard.png", dpi=230)
    plt.close(fig)

    return geo


def plot_router_norm_heatmaps(geo, layers, out_dir: Path, old_experts: int):
    old_values = np.concatenate(
        [
            geo["old_norm/wiki_only"].reshape(-1),
            geo["old_norm/code_trained"].reshape(-1),
            geo["old_norm/router_retuned"].reshape(-1),
        ]
    )
    new_values = np.concatenate(
        [geo["new_norm/code_trained"].reshape(-1), geo["new_norm/router_retuned"].reshape(-1)]
    )
    all_values = np.concatenate([old_values, new_values])
    all_values = all_values[np.isfinite(all_values)]
    vmax = float(np.percentile(all_values, 95)) if len(all_values) else None

    fig, axes = plt.subplots(2, 3, figsize=(15.8, 7.6), squeeze=False)
    imshow_heatmap(axes[0][0], geo["old_norm/wiki_only"], "Old rows: wiki-only", layers, 0, "Blues", vmin=0.0, vmax=vmax, cbar_label="row norm")
    imshow_heatmap(axes[0][1], geo["old_norm/code_trained"], "Old rows: after code", layers, 0, "Blues", vmin=0.0, vmax=vmax, cbar_label="row norm")
    imshow_heatmap(axes[0][2], geo["old_norm/router_retuned"], "Old rows: after retune", layers, 0, "Blues", vmin=0.0, vmax=vmax, cbar_label="row norm")
    imshow_heatmap(axes[1][0], geo["new_norm/code_trained"], "New rows: after code", layers, old_experts, "Oranges", vmin=0.0, vmax=vmax, cbar_label="row norm")
    imshow_heatmap(axes[1][1], geo["new_norm/router_retuned"], "New rows: after retune", layers, old_experts, "Greens", vmin=0.0, vmax=vmax, cbar_label="row norm")

    delta = geo["new_norm/router_retuned"] - geo["new_norm/code_trained"]
    finite_delta = delta[np.isfinite(delta)]
    dmax = float(max(abs(np.percentile(finite_delta, 5)), abs(np.percentile(finite_delta, 95)))) if len(finite_delta) else 1.0
    imshow_heatmap(
        axes[1][2],
        delta,
        "New rows: retune - code",
        layers,
        old_experts,
        "coolwarm",
        vmin=-dmax,
        vmax=dmax,
        cbar_label="norm delta",
    )

    fig.suptitle("FFN Router Row Norm Heatmaps", fontsize=17, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "ffn_only_router_weight_row_norm_heatmaps.png", dpi=230)
    plt.close(fig)


def plot_old_new_similarity_dashboard(geo, layers, out_dir: Path):
    x = np.arange(len(layers))
    layer_labels = [str(layer + 1) for layer in layers]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    axes[0].plot(x, geo["old_new_mean_cosine/code_trained"], marker="o", lw=2.2, color="#f97316", label="after code")
    axes[0].plot(x, geo["old_new_mean_cosine/router_retuned"], marker="o", lw=2.2, color="#16a34a", label="after retune")
    axes[0].axhline(0.0, color="#94a3b8", lw=1.0, linestyle="--")
    axes[0].set_title("Mean old-new row cosine", weight="bold")
    axes[0].set_ylabel("mean pairwise cosine")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(layer_labels)
    axes[0].grid(alpha=0.24)
    axes[0].legend(frameon=False)

    axes[1].plot(x, geo["old_new_max_cosine/code_trained"], marker="o", lw=2.2, color="#f97316", label="after code")
    axes[1].plot(x, geo["old_new_max_cosine/router_retuned"], marker="o", lw=2.2, color="#16a34a", label="after retune")
    axes[1].set_title("Most similar old-new row pair", weight="bold")
    axes[1].set_ylabel("max pairwise cosine")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(layer_labels)
    axes[1].grid(alpha=0.24)
    axes[1].legend(frameon=False)

    fig.suptitle("Old/New Router Row Competition Geometry", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_dir / "ffn_only_router_weight_old_new_similarity.png", dpi=230)
    plt.close(fig)


def plot_transition_heatmaps(stages, layers, out_dir: Path, old_experts: int):
    mats = transition_matrices(stages, layers, old_experts)

    old_l2_keys = [
        ("old_l2/wiki_to_code", "Old rows: Wiki -> Code"),
        ("old_l2/wiki_to_retune", "Old rows: Wiki -> Retuned"),
        ("old_l2/code_to_retune", "Old rows: Code -> Retuned"),
    ]
    old_l2_values = np.concatenate([mats[key].reshape(-1) for key, _ in old_l2_keys])
    old_l2_values = old_l2_values[np.isfinite(old_l2_values)]
    l2_vmax = float(np.percentile(old_l2_values, 95)) if len(old_l2_values) else None

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.6), squeeze=False)
    for ax, (key, title) in zip(axes[0], old_l2_keys):
        imshow_heatmap(ax, mats[key], title, layers, 0, "YlOrRd", vmin=0.0, vmax=l2_vmax, cbar_label="L2 distance")
    fig.suptitle("FFN Router Weight Movement Heatmap (Old Expert Rows)", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = out_dir / "ffn_only_router_weight_old_row_l2_heatmaps.png"
    fig.savefig(path, dpi=230)
    plt.close(fig)

    old_cos_keys = [
        ("old_cosine/wiki_to_code", "Old rows: Wiki vs Code"),
        ("old_cosine/wiki_to_retune", "Old rows: Wiki vs Retuned"),
        ("old_cosine/code_to_retune", "Old rows: Code vs Retuned"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.6), squeeze=False)
    for ax, (key, title) in zip(axes[0], old_cos_keys):
        imshow_heatmap(ax, mats[key], title, layers, 0, "viridis", vmin=0.0, vmax=1.0, cbar_label="row cosine")
    fig.suptitle("FFN Router Weight Similarity Heatmap (Old Expert Rows)", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path_cos = out_dir / "ffn_only_router_weight_old_row_cosine_heatmaps.png"
    fig.savefig(path_cos, dpi=230)
    plt.close(fig)

    if mats["new_l2/code_to_retune"].shape[0] > 0:
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), squeeze=False)
        imshow_heatmap(
            axes[0][0],
            mats["new_l2/code_to_retune"],
            "New rows: Code -> Retuned",
            layers,
            old_experts,
            "YlOrRd",
            vmin=0.0,
            cbar_label="L2 distance",
        )
        imshow_heatmap(
            axes[0][1],
            mats["new_cosine/code_to_retune"],
            "New rows: Code vs Retuned",
            layers,
            old_experts,
            "viridis",
            vmin=0.0,
            vmax=1.0,
            cbar_label="row cosine",
        )
        fig.suptitle("FFN Router Weight Movement Heatmap (New Expert Rows)", fontsize=16, weight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        path_new = out_dir / "ffn_only_router_weight_new_row_heatmaps.png"
        fig.savefig(path_new, dpi=230)
        plt.close(fig)

    return mats


def plot_layer_summary(mats, layers, out_dir: Path):
    x = np.arange(len(layers))
    layer_labels = [str(layer + 1) for layer in layers]

    def colmean(key):
        return np.nanmean(mats[key], axis=0)

    wiki_to_code = colmean("old_l2/wiki_to_code")
    wiki_to_retune = colmean("old_l2/wiki_to_retune")
    code_to_retune = colmean("old_l2/code_to_retune")
    return_ratio = wiki_to_retune / np.maximum(wiki_to_code, 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.8))
    axes[0].plot(x, wiki_to_code, marker="o", lw=2.2, color="#f97316", label="Wiki -> Code")
    axes[0].plot(x, wiki_to_retune, marker="o", lw=2.2, color="#16a34a", label="Wiki -> Retuned")
    axes[0].plot(x, code_to_retune, marker="o", lw=2.2, color="#64748b", label="Code -> Retuned")
    axes[0].set_title("Mean row movement by layer", weight="bold")
    axes[0].set_ylabel("Mean L2 distance")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(layer_labels)
    axes[0].set_xlabel("Layer")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].axhline(1.0, color="#94a3b8", lw=1.2, linestyle="--", label="No recovery")
    axes[1].plot(x, return_ratio, marker="o", lw=2.4, color="#2563eb", label="Retuned distance / Code distance")
    axes[1].fill_between(x, 0, np.minimum(return_ratio, 1.0), color="#2563eb", alpha=0.10)
    axes[1].set_title("Wiki-space recovery ratio", weight="bold")
    axes[1].set_ylabel("Lower means closer to wiki-only")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(layer_labels)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle("FFN Router Weight Change Summary", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = out_dir / "ffn_only_router_weight_layer_summary.png"
    fig.savefig(path, dpi=230)
    plt.close(fig)


def plot_weight_story_dashboard(mats, layers, out_dir: Path, old_experts: int):
    layer_x = np.arange(len(layers))
    layer_labels = [str(layer + 1) for layer in layers]

    old_code_from_wiki = np.nanmean(mats["old_l2/wiki_to_code"], axis=0)
    old_retune_from_wiki = np.nanmean(mats["old_l2/wiki_to_retune"], axis=0)
    old_code_to_retune = np.nanmean(mats["old_l2/code_to_retune"], axis=0)
    old_retune_cos_wiki = np.nanmean(mats["old_cosine/wiki_to_retune"], axis=0)

    has_new = mats["new_l2/code_to_retune"].shape[0] > 0
    new_code_to_retune = np.nanmean(mats["new_l2/code_to_retune"], axis=0) if has_new else None
    new_cos = np.nanmean(mats["new_cosine/code_to_retune"], axis=0) if has_new else None

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2))
    width = 0.34

    axes[0][0].bar(layer_x - width / 2, old_code_from_wiki, width, color="#f97316", alpha=0.85, label="after code training")
    axes[0][0].bar(layer_x + width / 2, old_retune_from_wiki, width, color="#16a34a", alpha=0.85, label="after router retune")
    axes[0][0].set_title("Old router rows: distance from wiki-only", weight="bold")
    axes[0][0].set_ylabel("Mean L2 distance")
    axes[0][0].set_xticks(layer_x)
    axes[0][0].set_xticklabels(layer_labels)
    axes[0][0].grid(axis="y", alpha=0.24)
    axes[0][0].legend(frameon=False)
    axes[0][0].text(
        0.02,
        0.96,
        "If orange is near zero,\nold rows were effectively unchanged\nby code training.",
        transform=axes[0][0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="#cbd5e1", alpha=0.90, boxstyle="round,pad=0.35"),
    )

    axes[0][1].bar(layer_x, old_code_to_retune, color="#64748b", alpha=0.88)
    axes[0][1].set_title("Old router rows: movement caused by retune", weight="bold")
    axes[0][1].set_ylabel("Mean L2 distance, code -> retuned")
    axes[0][1].set_xticks(layer_x)
    axes[0][1].set_xticklabels(layer_labels)
    axes[0][1].grid(axis="y", alpha=0.24)

    if has_new:
        axes[1][0].bar(layer_x, new_code_to_retune, color="#0ea5e9", alpha=0.88)
        axes[1][0].set_title(f"New router rows {old_experts}+ : movement caused by retune", weight="bold")
        axes[1][0].set_ylabel("Mean L2 distance, code -> retuned")
    else:
        axes[1][0].text(0.5, 0.5, "No new rows found", transform=axes[1][0].transAxes, ha="center", va="center", fontsize=13)
        axes[1][0].set_title("New router rows", weight="bold")
    axes[1][0].set_xticks(layer_x)
    axes[1][0].set_xticklabels(layer_labels)
    axes[1][0].grid(axis="y", alpha=0.24)

    axes[1][1].plot(layer_x, old_retune_cos_wiki, marker="o", lw=2.2, color="#16a34a", label="old rows: wiki vs retuned")
    if has_new:
        axes[1][1].plot(layer_x, new_cos, marker="o", lw=2.2, color="#0ea5e9", label="new rows: code vs retuned")
    axes[1][1].set_title("Cosine similarity after retune", weight="bold")
    axes[1][1].set_ylabel("Mean row cosine")
    axes[1][1].set_ylim(0.0, 1.02)
    axes[1][1].set_xticks(layer_x)
    axes[1][1].set_xticklabels(layer_labels)
    axes[1][1].grid(alpha=0.24)
    axes[1][1].legend(frameon=False)

    fig.suptitle(
        "FFN Router Weight Story\n"
        "Code training barely changes old rows; router retune moves old/new rows",
        fontsize=17,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out_dir / "ffn_only_router_weight_story_dashboard.png", dpi=230)
    plt.close(fig)


def plot_recovery_ratio_heatmap(mats, layers, out_dir: Path):
    wiki_to_code = mats["old_l2/wiki_to_code"]
    wiki_to_retune = mats["old_l2/wiki_to_retune"]
    ratio = wiki_to_retune / np.maximum(wiki_to_code, 1e-8)
    finite = ratio[np.isfinite(ratio)]
    vmax = max(1.5, float(np.percentile(finite, 95))) if len(finite) else 1.5
    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    image = ax.imshow(np.ma.masked_invalid(ratio), aspect="auto", cmap="RdYlGn_r", norm=norm)
    ax.set_title(
        "Router Weight Ratio by Expert Row\n"
        "diagnostic only: unstable when Wiki->Code distance is near zero",
        fontsize=15,
        weight="bold",
    )
    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels([str(layer + 1) for layer in layers])
    ax.set_xlabel("Layer")
    ax.set_yticks(np.arange(ratio.shape[0]))
    ax.set_yticklabels([str(idx) for idx in range(ratio.shape[0])])
    ax.set_ylabel("Old expert row")

    for row in range(ratio.shape[0]):
        for col in range(ratio.shape[1]):
            value = ratio[row, col]
            if not np.isfinite(value):
                continue
            color = "white" if value > 1.15 or value < 0.45 else "#111827"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7.5, color=color)

    cbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label("||retuned - wiki|| / ||code - wiki||", fontsize=9)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_dir / "ffn_only_router_weight_recovery_ratio_heatmap.png", dpi=230)
    plt.close(fig)


def plot_recovery_scatter(mats, layers, out_dir: Path):
    wiki_to_code = mats["old_l2/wiki_to_code"]
    wiki_to_retune = mats["old_l2/wiki_to_retune"]
    code_to_retune = mats["old_l2/code_to_retune"]

    xs, ys, cs, sizes = [], [], [], []
    for layer_idx, layer in enumerate(layers):
        for expert_id in range(wiki_to_code.shape[0]):
            x = wiki_to_code[expert_id, layer_idx]
            y = wiki_to_retune[expert_id, layer_idx]
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            xs.append(float(x))
            ys.append(float(y))
            cs.append(layer + 1)
            sizes.append(42 + 18 * min(max(float(code_to_retune[expert_id, layer_idx]), 0.0), 2.5))

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    cs = np.asarray(cs)
    sizes = np.asarray(sizes)
    max_axis = float(max(np.max(xs), np.max(ys))) * 1.08 if len(xs) else 1.0
    recovered = int(np.sum(ys < xs)) if len(xs) else 0
    total = int(len(xs))
    median_ratio = float(np.median(ys / np.maximum(xs, 1e-8))) if len(xs) else float("nan")

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    ax.plot([0, max_axis], [0, max_axis], color="#64748b", lw=1.4, linestyle="--", label="no recovery line")
    scatter = ax.scatter(xs, ys, c=cs, s=sizes, cmap="viridis", alpha=0.82, edgecolors="white", linewidths=0.55)
    ax.fill_between([0, max_axis], [0, max_axis], [0, 0], color="#16a34a", alpha=0.08, label="closer to wiki after retune")
    ax.set_xlim(0, max_axis)
    ax.set_ylim(0, max_axis)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Distance moved by code training: ||code - wiki||")
    ax.set_ylabel("Distance after retune: ||retuned - wiki||")
    ax.set_title("Router Weight Distance Scatter\nuse carefully when x-axis is near zero", fontsize=14, weight="bold")
    ax.grid(alpha=0.24)
    ax.legend(frameon=False, loc="upper left")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Layer", fontsize=9)
    ax.text(
        0.98,
        0.04,
        f"recovered rows: {recovered}/{total}\nmedian ratio: {median_ratio:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="#cbd5e1", alpha=0.88, boxstyle="round,pad=0.35"),
    )
    fig.tight_layout()
    fig.savefig(out_dir / "ffn_only_router_weight_recovery_scatter.png", dpi=230)
    plt.close(fig)


def plot_layer_recovery_bars(mats, layers, out_dir: Path):
    wiki_to_code = np.nanmean(mats["old_l2/wiki_to_code"], axis=0)
    wiki_to_retune = np.nanmean(mats["old_l2/wiki_to_retune"], axis=0)
    ratio = wiki_to_retune / np.maximum(wiki_to_code, 1e-8)
    x = np.arange(len(layers))
    layer_labels = [str(layer + 1) for layer in layers]

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.25, 1.0]})
    width = 0.36
    axes[0].bar(x - width / 2, wiki_to_code, width=width, color="#f97316", alpha=0.86, label="after code training")
    axes[0].bar(x + width / 2, wiki_to_retune, width=width, color="#16a34a", alpha=0.86, label="after router retune")
    axes[0].set_ylabel("Mean distance from wiki-only")
    axes[0].set_title("Layer-wise router drift from wiki-only", weight="bold")
    axes[0].grid(axis="y", alpha=0.24)
    axes[0].legend(frameon=False)

    colors = np.where(ratio < 1.0, "#16a34a", "#ef4444")
    axes[1].axhline(1.0, color="#64748b", lw=1.2, linestyle="--")
    axes[1].bar(x, ratio, color=colors, alpha=0.86)
    axes[1].set_ylabel("Recovery ratio")
    axes[1].set_xlabel("Layer")
    axes[1].set_title("Retuned distance / Code-trained distance", weight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(layer_labels)
    axes[1].grid(axis="y", alpha=0.24)
    for idx, value in enumerate(ratio):
        axes[1].text(idx, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("FFN Router Weight Recovery Overview", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "ffn_only_router_weight_recovery_overview.png", dpi=230)
    plt.close(fig)


def plot_layer_grid(stages, layers, out_path: Path, old_experts: int, display_layer_offset: int, normalize_for_pca: bool):
    n_layers = len(layers)
    ncols = min(3, n_layers)
    nrows = math.ceil(n_layers / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.8 * nrows), squeeze=False)

    for panel_idx, layer in enumerate(layers):
        ax = axes[panel_idx // ncols][panel_idx % ncols]
        raw_parts = [stage["by_layer"][layer]["weight"] for stage in stages]
        pca_parts = [normalize_rows(part) if normalize_for_pca else part for part in raw_parts]
        coords = pca2(np.concatenate(pca_parts, axis=0))
        parts = split_coords(coords, [part.shape[0] for part in raw_parts])

        wiki_xy, code_xy, retune_xy = parts
        old_count = min(old_experts, len(wiki_xy), len(code_xy), len(retune_xy))
        for expert_id in range(old_count):
            ax.plot(
                [wiki_xy[expert_id, 0], code_xy[expert_id, 0], retune_xy[expert_id, 0]],
                [wiki_xy[expert_id, 1], code_xy[expert_id, 1], retune_xy[expert_id, 1]],
                color="#111827",
                alpha=0.24,
                lw=0.85,
            )
            ax.annotate(
                "",
                xy=retune_xy[expert_id],
                xytext=code_xy[expert_id],
                arrowprops=dict(arrowstyle="->", color="#111827", alpha=0.28, lw=0.85, shrinkA=0, shrinkB=0),
            )

        new_count = max(min(len(code_xy), len(retune_xy)) - old_experts, 0)
        for local_id in range(new_count):
            expert_id = old_experts + local_id
            ax.plot(
                [code_xy[expert_id, 0], retune_xy[expert_id, 0]],
                [code_xy[expert_id, 1], retune_xy[expert_id, 1]],
                color="#6b7280",
                alpha=0.20,
                lw=0.75,
                linestyle="--",
            )

        for (label, color, marker), part in zip(STAGES, parts):
            old_part = part[: min(old_experts, len(part))]
            if len(old_part):
                ax.scatter(old_part[:, 0], old_part[:, 1], s=34, c=color, marker=marker, alpha=0.86, label=f"{label} old")
            if len(part) > old_experts:
                new_part = part[old_experts:]
                ax.scatter(
                    new_part[:, 0],
                    new_part[:, 1],
                    s=44,
                    c=color,
                    marker="x",
                    alpha=0.90,
                    linewidths=1.4,
                    label=f"{label} new",
                )

        centroids = []
        for part in parts:
            centroids.append(part[: min(old_experts, len(part))].mean(axis=0))
        centroids = np.stack(centroids)
        ax.plot(centroids[:, 0], centroids[:, 1], color="#111827", lw=2.0, alpha=0.72)
        ax.scatter(centroids[:, 0], centroids[:, 1], color="#111827", s=38, alpha=0.90, zorder=8)

        xmin, ymin = coords.min(axis=0)
        xmax, ymax = coords.max(axis=0)
        xpad = max((xmax - xmin) * 0.10, 1e-5)
        ypad = max((ymax - ymin) * 0.10, 1e-5)
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)
        ax.set_title(f"Layer {layer + display_layer_offset}", fontsize=11, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.16)

    for panel_idx in range(n_layers, nrows * ncols):
        axes[panel_idx // ncols][panel_idx % ncols].axis("off")

    handles, labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
    dedup = dict(zip(labels, handles))
    fig.legend(dedup.values(), dedup.keys(), loc="lower center", ncol=3, frameon=False)
    suffix = "row-normalized " if normalize_for_pca else ""
    fig.suptitle(f"Diagnostic PCA: FFN Router Weight Row Trajectory by Layer ({suffix}PCA)", y=0.992, fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0.045, 1, 0.955))
    fig.savefig(out_path, dpi=230)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stages = [
        load_router_weights("wiki_only", args.wiki_only, args.router_key_regex, args.list_keys),
        load_router_weights("code_trained", args.code_trained, args.router_key_regex, args.list_keys),
        load_router_weights("router_retuned", args.router_retuned, args.router_key_regex, args.list_keys),
    ]
    common_layers = sorted(set.intersection(*(set(stage["by_layer"].keys()) for stage in stages)))
    if not common_layers:
        raise SystemExit("No common router layers found across the three checkpoints.")

    if args.plot_pca_trajectory:
        plot_path = out_dir / "ffn_only_router_weight_row_trajectory_layers_pca.png"
        plot_layer_grid(
            stages,
            common_layers,
            plot_path,
            args.old_experts,
            args.display_layer_offset,
            args.normalize_rows_for_pca,
        )
    heatmap_mats = plot_transition_heatmaps(stages, common_layers, out_dir, args.old_experts)
    geometry_mats = plot_router_geometry_dashboard(stages, common_layers, out_dir, args.old_experts)
    plot_router_norm_heatmaps(geometry_mats, common_layers, out_dir, args.old_experts)
    plot_old_new_similarity_dashboard(geometry_mats, common_layers, out_dir)
    plot_weight_story_dashboard(heatmap_mats, common_layers, out_dir, args.old_experts)

    metrics = collect_metrics(stages, common_layers, args.old_experts)
    metrics["checkpoints"] = {stage["label"]: stage["checkpoint_dir"] for stage in stages}
    metrics["router_key_regex"] = args.router_key_regex
    metrics["normalize_rows_for_pca"] = args.normalize_rows_for_pca
    (out_dir / "ffn_only_router_weight_trajectory_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.plot_pca_trajectory:
        print(f"[DONE] wrote {plot_path}")
    print(f"[DONE] wrote {out_dir / 'ffn_only_router_weight_trajectory_metrics.json'}")


if __name__ == "__main__":
    main()
