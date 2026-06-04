#!/usr/bin/env python3
"""Plot FFN MoE router row trajectories across wiki/code/retuned checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STAGES = [
    ("wiki_only", "#2563eb", "o"),
    ("code_trained", "#f97316", "s"),
    ("router_retuned", "#16a34a", "^"),
]


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

    handles, labels = axes[0][0].get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    fig.legend(dedup.values(), dedup.keys(), loc="upper center", ncol=4, frameon=False)
    suffix = "row-normalized " if normalize_for_pca else ""
    fig.suptitle(f"FFN-only Router Weight Row Trajectory by Layer ({suffix}PCA)", y=0.995, fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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

    plot_path = out_dir / "ffn_only_router_weight_row_trajectory_layers_pca.png"
    plot_layer_grid(
        stages,
        common_layers,
        plot_path,
        args.old_experts,
        args.display_layer_offset,
        args.normalize_rows_for_pca,
    )

    metrics = collect_metrics(stages, common_layers, args.old_experts)
    metrics["checkpoints"] = {stage["label"]: stage["checkpoint_dir"] for stage in stages}
    metrics["router_key_regex"] = args.router_key_regex
    metrics["normalize_rows_for_pca"] = args.normalize_rows_for_pca
    (out_dir / "ffn_only_router_weight_trajectory_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[DONE] wrote {plot_path}")
    print(f"[DONE] wrote {out_dir / 'ffn_only_router_weight_trajectory_metrics.json'}")


if __name__ == "__main__":
    main()
