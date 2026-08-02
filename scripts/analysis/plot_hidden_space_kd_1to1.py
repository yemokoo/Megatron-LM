#!/usr/bin/env python3
"""Plot joint-PCA hidden distributions for Wiki-only -> KD-init -> Code/Wiki 1:1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_hidden_space_ffn_only import (
    collect_metrics,
    draw_density_cloud,
    expand_limits,
    reduce2,
    subsample_indices,
)


STAGES = (
    ("wiki_only", "Wiki-only", "#2563eb"),
    ("kd_init", "KD initialization", "#f97316"),
    ("code_wiki_1to1", "Code:Wiki 1:1", "#9333ea"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-only", required=True)
    parser.add_argument("--kd-init", required=True)
    parser.add_argument("--code-wiki-1to1", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--probe-task", choices=["wiki", "code"], required=True)
    parser.add_argument("--model-label", default="G2 FFN-only")
    parser.add_argument("--method", choices=["pca", "umap"], default="pca")
    parser.add_argument("--max-points-per-stage", type=int, default=1200)
    parser.add_argument("--trim-percentile", type=float, default=99.0)
    parser.add_argument("--density-bins", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--stage1-log",
        default=None,
        help="Optional Code phase-1 log with probe metrics at local steps 0 and 1800.",
    )
    parser.add_argument("--code-phase-steps", type=int, default=1800)
    return parser.parse_args()


def load_dump(path: str, expected_label: str):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        label = metadata.get("label") or expected_label
        if label != expected_label:
            raise SystemExit(
                f"Unexpected dump label for {path}: expected={expected_label} actual={label}"
            )
        return {
            "path": path,
            "label": label,
            "hidden": data["hidden_layers"].astype(np.float32),
            "layers": data["layer_numbers"].astype(np.int64),
            "token_ids": data["token_ids"].astype(np.int64),
            "positions": data["positions"].astype(np.int64),
            "sample_indices": data["sample_indices"].astype(np.int64),
            "metadata": metadata,
        }


def validate_same_tokens(dumps):
    reference = dumps[0]
    identity_fields = ("token_ids", "positions", "sample_indices")
    for dump in dumps[1:]:
        if not np.array_equal(dump["layers"], reference["layers"]):
            raise SystemExit(
                f"Layer mismatch: {reference['label']}={reference['layers'].tolist()} "
                f"{dump['label']}={dump['layers'].tolist()}"
            )
        if dump["hidden"].shape != reference["hidden"].shape:
            raise SystemExit(
                f"Hidden shape mismatch: {reference['label']}={reference['hidden'].shape} "
                f"{dump['label']}={dump['hidden'].shape}"
            )
        for field in identity_fields:
            if not np.array_equal(dump[field], reference[field]):
                raise SystemExit(
                    f"Token identity mismatch in {field}: "
                    f"{reference['label']} vs {dump['label']}"
                )

    digest = hashlib.sha256()
    for field in identity_fields:
        digest.update(reference[field].tobytes())
    return digest.hexdigest()


def joint_coords(dumps, layer_idx, indices, method, seed):
    sampled = [dump["hidden"][layer_idx][indices] for dump in dumps]
    coords = reduce2(np.concatenate(sampled, axis=0), method, seed)
    return coords, np.split(coords, len(dumps))


def draw_centroid_path(ax, parts, stages=STAGES):
    centroids = np.stack([part.mean(axis=0) for part in parts])
    ax.plot(centroids[:, 0], centroids[:, 1], color="#111827", lw=1.5, alpha=0.82)
    for centroid, (_label, _display, color) in zip(centroids, stages):
        ax.scatter(
            [centroid[0]],
            [centroid[1]],
            s=35,
            c=color,
            edgecolors="#111827",
            linewidths=0.7,
            zorder=8,
        )


def plot_layer_grid(dumps, out_path, args, indices):
    layers = dumps[0]["layers"]
    ncols = min(3, len(layers))
    nrows = math.ceil(len(layers) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.5 * nrows), squeeze=False)

    for layer_idx, layer_number in enumerate(layers):
        ax = axes[layer_idx // ncols][layer_idx % ncols]
        coords, parts = joint_coords(
            dumps,
            layer_idx,
            indices,
            args.method,
            args.seed + int(layer_number),
        )
        for part, (_label, display, color) in zip(parts, STAGES):
            draw_density_cloud(
                ax,
                part,
                color,
                display,
                args.density_bins,
                args.trim_percentile,
                alpha=0.30,
            )
        draw_centroid_path(ax, parts)
        expand_limits(ax, coords, trim_percentile=args.trim_percentile)
        ax.set_title(f"Layer {int(layer_number)}", fontsize=11, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.16)

    for idx in range(len(layers), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        f"{args.model_label} {args.probe_task.title()} Probe Hidden Density by Layer "
        f"({args.method.upper()}, joint fit)",
        y=0.995,
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.965))
    fig.savefig(out_path, dpi=230)
    plt.close(fig)


def plot_layer_average(dumps, out_path, args, indices):
    sampled = [dump["hidden"].mean(axis=0)[indices] for dump in dumps]
    coords = reduce2(np.concatenate(sampled, axis=0), args.method, args.seed + 10000)
    parts = np.split(coords, len(dumps))

    fig, ax = plt.subplots(figsize=(10.3, 6.8))
    for part, (_label, display, color) in zip(parts, STAGES):
        draw_density_cloud(
            ax,
            part,
            color,
            display,
            args.density_bins,
            args.trim_percentile,
            alpha=0.32,
        )
    draw_centroid_path(ax, parts)
    expand_limits(ax, coords, trim_percentile=args.trim_percentile)
    ax.set_title(
        f"{args.model_label} {args.probe_task.title()} Probe Hidden Density, Layer-Average\n"
        f"{args.method.upper()} jointly fit across all three checkpoints",
        weight="bold",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.18)
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.015, 1.0),
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_kd_vs_code1phase_layer_grid(dumps, out_path, args, indices):
    pair_dumps = dumps[1:]
    pair_stages = STAGES[1:]
    layers = pair_dumps[0]["layers"]
    ncols = min(3, len(layers))
    nrows = math.ceil(len(layers) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.5 * nrows), squeeze=False)

    for layer_idx, layer_number in enumerate(layers):
        ax = axes[layer_idx // ncols][layer_idx % ncols]
        sampled = [dump["hidden"][layer_idx][indices] for dump in pair_dumps]
        coords = reduce2(
            np.concatenate(sampled, axis=0),
            args.method,
            args.seed + 20000 + int(layer_number),
        )
        parts = np.split(coords, len(pair_dumps))
        for part, (_label, display, color) in zip(parts, pair_stages):
            draw_density_cloud(
                ax,
                part,
                color,
                display,
                args.density_bins,
                args.trim_percentile,
                alpha=0.32,
            )
        draw_centroid_path(ax, parts, stages=pair_stages)
        expand_limits(ax, coords, trim_percentile=args.trim_percentile)
        ax.set_title(f"Layer {int(layer_number)}", fontsize=11, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.16)

    for idx in range(len(layers), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        f"{args.model_label} {args.probe_task.title()} Probe: "
        f"KD Initialization vs Code Phase 1\n"
        f"Hidden Density by Layer ({args.method.upper()}, pairwise joint fit)",
        y=0.997,
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    fig.savefig(out_path, dpi=230)
    plt.close(fig)


def plot_kd_vs_code1phase_layer_average(dumps, out_path, args, indices):
    pair_dumps = dumps[1:]
    pair_stages = STAGES[1:]
    sampled = [dump["hidden"].mean(axis=0)[indices] for dump in pair_dumps]
    coords = reduce2(np.concatenate(sampled, axis=0), args.method, args.seed + 30000)
    parts = np.split(coords, len(pair_dumps))

    fig, ax = plt.subplots(figsize=(10.3, 6.8))
    for part, (_label, display, color) in zip(parts, pair_stages):
        draw_density_cloud(
            ax,
            part,
            color,
            display,
            args.density_bins,
            args.trim_percentile,
            alpha=0.34,
        )
    draw_centroid_path(ax, parts, stages=pair_stages)
    expand_limits(ax, coords, trim_percentile=args.trim_percentile)
    ax.set_title(
        f"{args.model_label} {args.probe_task.title()} Probe: "
        f"KD Initialization vs Code Phase 1\n"
        f"Layer-Average Hidden Density ({args.method.upper()}, pairwise joint fit)",
        weight="bold",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.18)
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.015, 1.0),
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def read_phase1_probe_performance(log_path: str, probe_task: str, final_step: int):
    pattern = re.compile(
        rf"probe {re.escape(probe_task)}_probe at iteration\s+(?P<iteration>\d+)\s+\|\s+"
        rf"local_iteration:\s+(?P<local>\d+)\s+\|\s+next_token_acc:\s+"
        rf"(?P<acc>[0-9.Ee+-]+)\s+\|\s+ppl:\s+(?P<ppl>[0-9.Ee+-]+)"
    )
    records = {}
    for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        local_step = int(match.group("local"))
        if local_step in {0, final_step}:
            records[local_step] = {
                "global_iteration": int(match.group("iteration")),
                "local_iteration": local_step,
                "next_token_acc": float(match.group("acc")),
                "ppl": float(match.group("ppl")),
            }
    missing = sorted({0, final_step} - set(records))
    if missing:
        raise SystemExit(f"Missing {probe_task} probe performance at local step(s) {missing}: {log_path}")
    return {
        "source_log": str(log_path),
        "kd_init": records[0],
        "code_phase1": records[final_step],
    }


def plot_probe_performance(performance, out_path, args):
    stages = ("kd_init", "code_phase1")
    displays = ("KD initialization", "Code phase 1")
    colors = (STAGES[1][2], STAGES[2][2])
    accuracy = [performance[stage]["next_token_acc"] for stage in stages]
    ppl = [performance[stage]["ppl"] for stage in stages]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for ax, values, title, ylabel, fmt in (
        (axes[0], accuracy, "Next-token accuracy", "Accuracy", ".4f"),
        (axes[1], ppl, "Perplexity", "PPL (lower is better)", ".3f"),
    ):
        bars = ax.bar(displays, values, color=colors, width=0.58)
        ax.bar_label(bars, labels=[format(value, fmt) for value in values], padding=4, fontsize=10)
        ax.set_title(title, weight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.24)
        ax.set_axisbelow(True)
        ax.margins(y=0.16)
    fig.suptitle(
        f"{args.model_label} {args.probe_task.title()} Probe Performance: "
        "KD Initialization vs Code Phase 1",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=230)
    plt.close(fig)


def token_preservation_stats(a: np.ndarray, b: np.ndarray):
    dot = np.sum(a * b, axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    cosine = dot / np.maximum(a_norm * b_norm, 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    normalized_l2 = 2.0 * np.linalg.norm(a - b, axis=1) / np.maximum(a_norm + b_norm, 1e-8)
    return {
        "paired_cosine_mean": float(cosine.mean()),
        "paired_cosine_median": float(np.median(cosine)),
        "paired_cosine_p05": float(np.percentile(cosine, 5)),
        "paired_cosine_p25": float(np.percentile(cosine, 25)),
        "fraction_cosine_ge_0_99": float(np.mean(cosine >= 0.99)),
        "fraction_cosine_ge_0_95": float(np.mean(cosine >= 0.95)),
        "fraction_cosine_ge_0_90": float(np.mean(cosine >= 0.90)),
        "normalized_l2_mean": float(normalized_l2.mean()),
        "normalized_l2_median": float(np.median(normalized_l2)),
        "normalized_l2_p75": float(np.percentile(normalized_l2, 75)),
        "normalized_l2_p95": float(np.percentile(normalized_l2, 95)),
    }


def collect_preservation_metrics(dumps):
    pairs = ((0, 1), (0, 2), (1, 2))
    result = {"layers": []}
    for layer_idx, layer_number in enumerate(dumps[0]["layers"]):
        row = {"layer": int(layer_number), "pairs": {}}
        for left_idx, right_idx in pairs:
            left = dumps[left_idx]
            right = dumps[right_idx]
            pair_name = f"{left['label']}__vs__{right['label']}"
            row["pairs"][pair_name] = token_preservation_stats(
                left["hidden"][layer_idx],
                right["hidden"][layer_idx],
            )
        result["layers"].append(row)

    result["layer_average"] = {}
    averaged = [dump["hidden"].mean(axis=0) for dump in dumps]
    for left_idx, right_idx in pairs:
        pair_name = f"{dumps[left_idx]['label']}__vs__{dumps[right_idx]['label']}"
        result["layer_average"][pair_name] = token_preservation_stats(
            averaged[left_idx],
            averaged[right_idx],
        )
    return result


def plot_preservation_by_layer(preservation, out_path, args):
    pair_specs = (
        ("wiki_only__vs__kd_init", "Wiki-only → KD initialization", "#f97316"),
        ("wiki_only__vs__code_wiki_1to1", "Wiki-only → Code:Wiki 1:1", "#9333ea"),
        ("kd_init__vs__code_wiki_1to1", "KD initialization → Code:Wiki 1:1", "#059669"),
    )
    layers = [row["layer"] for row in preservation["layers"]]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    for pair_name, display, color in pair_specs:
        cosine = [
            row["pairs"][pair_name]["paired_cosine_mean"]
            for row in preservation["layers"]
        ]
        normalized_l2 = [
            row["pairs"][pair_name]["normalized_l2_median"]
            for row in preservation["layers"]
        ]
        axes[0].plot(layers, cosine, marker="o", color=color, label=display)
        axes[1].plot(layers, normalized_l2, marker="o", color=color, label=display)

    axes[0].set_title("Same-token hidden cosine (higher = more preserved)", weight="bold")
    axes[0].set_ylabel("Mean paired cosine")
    axes[1].set_title("Same-token hidden displacement (lower = more preserved)", weight="bold")
    axes[1].set_ylabel("Median symmetric normalized L2")
    for ax in axes:
        ax.set_xlabel("Transformer layer")
        ax.set_xticks(layers)
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        f"{args.model_label} {args.probe_task.title()} Probe Hidden Preservation",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    fig.savefig(out_path, dpi=230)
    plt.close(fig)


def main():
    args = parse_args()
    if args.max_points_per_stage <= 0:
        raise SystemExit("--max-points-per-stage must be positive")

    dumps = [
        load_dump(args.wiki_only, "wiki_only"),
        load_dump(args.kd_init, "kd_init"),
        load_dump(args.code_wiki_1to1, "code_wiki_1to1"),
    ]
    token_identity_sha256 = validate_same_tokens(dumps)
    rng = np.random.default_rng(args.seed)
    indices = subsample_indices(
        dumps[0]["hidden"].shape[1],
        args.max_points_per_stage,
        rng,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"kd_1to1_hidden_{args.probe_task}_probe"
    plot_layer_grid(dumps, out_dir / f"{prefix}_layers_{args.method}.png", args, indices)
    plot_layer_average(
        dumps,
        out_dir / f"{prefix}_layer_average_{args.method}.png",
        args,
        indices,
    )
    plot_kd_vs_code1phase_layer_grid(
        dumps,
        out_dir / f"{prefix}_kd_init_vs_code_phase1_layers_{args.method}.png",
        args,
        indices,
    )
    plot_kd_vs_code1phase_layer_average(
        dumps,
        out_dir / f"{prefix}_kd_init_vs_code_phase1_layer_average_{args.method}.png",
        args,
        indices,
    )
    preservation = collect_preservation_metrics(dumps)
    plot_preservation_by_layer(
        preservation,
        out_dir / f"{prefix}_preservation_by_layer.png",
        args,
    )

    metrics = collect_metrics(dumps)
    metrics["probe_task"] = args.probe_task
    metrics["method"] = args.method
    metrics["joint_fit"] = True
    metrics["captured_tokens"] = int(dumps[0]["hidden"].shape[1])
    metrics["plotted_tokens_per_stage"] = int(len(indices))
    metrics["token_identity_sha256"] = token_identity_sha256
    metrics["inputs"] = {dump["label"]: dump["path"] for dump in dumps}
    metrics["same_token_preservation"] = preservation
    if args.stage1_log:
        performance = read_phase1_probe_performance(
            args.stage1_log,
            args.probe_task,
            args.code_phase_steps,
        )
        plot_probe_performance(
            performance,
            out_dir / f"{prefix}_kd_init_vs_code_phase1_performance.png",
            args,
        )
        metrics["kd_init_vs_code_phase1_probe_performance"] = performance
    (out_dir / f"{prefix}_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[DONE] {args.probe_task} plots={out_dir} "
        f"tokens={dumps[0]['hidden'].shape[1]} token_sha256={token_identity_sha256}"
    )


if __name__ == "__main__":
    main()
