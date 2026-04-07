#!/usr/bin/env python3
import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def make_pair_labels(num_experts: int):
    return [tuple(pair) for pair in itertools.combinations(range(num_experts), 2)]


def make_lookup(num_experts: int, combo_labels):
    lookup = torch.full((num_experts * num_experts,), -1, dtype=torch.long)
    for idx, (a, b) in enumerate(combo_labels):
        lookup[a * num_experts + b] = idx
    return lookup


def encode_pairs(pairs: torch.Tensor, num_experts: int, lookup: torch.Tensor):
    codes = pairs[:, 0].long() * num_experts + pairs[:, 1].long()
    indices = lookup[codes]
    if (indices < 0).any():
        raise ValueError("Encountered routing pair outside expected combination set.")
    return indices


def transition_summary(new_pairs: torch.Tensor, old_pairs: torch.Tensor, source_num_experts: int):
    exact_same = (new_pairs == old_pairs).all(dim=-1)
    old_old = (new_pairs < source_num_experts).all(dim=-1)
    new_new = (new_pairs >= source_num_experts).all(dim=-1)
    old_new = ~(old_old | new_new)
    token_count = max(int(new_pairs.shape[0]), 1)
    return {
        "fraction_exact_same_pair": float(exact_same.float().mean().item()),
        "fraction_old_old_pair": float(old_old.float().mean().item()),
        "fraction_old_new_pair": float(old_new.float().mean().item()),
        "fraction_new_new_pair": float(new_new.float().mean().item()),
        "token_count": token_count,
    }


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_heatmaps(layer_fraction_maps, old_labels, new_labels, output_path: Path, title: str):
    layer_names = list(layer_fraction_maps.keys())
    n_layers = len(layer_names)
    ncols = 2
    nrows = int(np.ceil(n_layers / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 3.8 * nrows), squeeze=False)
    vmax = max(float(matrix.max()) for matrix in layer_fraction_maps.values())
    vmax = max(vmax, 1e-8)

    for ax in axes.flatten():
        ax.axis("off")

    for ax, layer_name in zip(axes.flatten(), layer_names):
        ax.axis("on")
        matrix = layer_fraction_maps[layer_name]
        im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0.0, vmax=vmax)
        ax.set_title(layer_name, fontsize=11, pad=8)
        ax.set_xticks(np.arange(len(new_labels)))
        ax.set_xticklabels(new_labels, rotation=90, fontsize=7)
        ax.set_yticks(np.arange(len(old_labels)))
        ax.set_yticklabels(old_labels, fontsize=8)
        ax.set_xlabel("Expanded 7-expert top-2 pair", fontsize=9)
        ax.set_ylabel("Original 4-expert top-2 pair", fontsize=9)

    fig.suptitle(title, fontsize=14, y=0.995)
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.08, right=0.93, hspace=0.55, wspace=0.30)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Row-normalized transition fraction", fontsize=10)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_aggregate_heatmap(matrix, old_labels, new_labels, output_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0.0, vmax=max(float(matrix.max()), 1e-8))
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xticks(np.arange(len(new_labels)))
    ax.set_xticklabels(new_labels, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(old_labels)))
    ax.set_yticklabels(old_labels, fontsize=9)
    ax.set_xlabel("Expanded 7-expert top-2 pair", fontsize=10)
    ax.set_ylabel("Original 4-expert top-2 pair", fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.96, 1])
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Row-normalized transition fraction", fontsize=10)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare shared-router top-2 pair transitions.")
    parser.add_argument("--wiki-routing-pt", type=str, required=True)
    parser.add_argument("--code-routing-pt", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--compare-label", type=str, required=True)
    parser.add_argument("--source-num-experts", type=int, default=4)
    parser.add_argument("--total-num-experts", type=int, default=7)
    args = parser.parse_args()

    wiki_payload = torch.load(args.wiki_routing_pt, map_location="cpu")
    code_payload = torch.load(args.code_routing_pt, map_location="cpu")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    old_pair_labels = make_pair_labels(args.source_num_experts)
    new_pair_labels = make_pair_labels(args.total_num_experts)
    old_lookup = make_lookup(args.total_num_experts, old_pair_labels)
    new_lookup = make_lookup(args.total_num_experts, new_pair_labels)
    old_label_strings = [f"e{a}+e{b}" for a, b in old_pair_labels]
    new_label_strings = [f"e{a}+e{b}" for a, b in new_pair_labels]

    shared_layers = sorted(
        set(wiki_payload["pairs_by_layer"].keys()) & set(code_payload["pairs_by_layer"].keys())
    )
    layer_counts = {}
    layer_row_fractions = {}
    layer_summaries = {}
    aggregate_counts = torch.zeros((len(old_pair_labels), len(new_pair_labels)), dtype=torch.long)

    for layer_name in shared_layers:
        wiki_pairs = wiki_payload["pairs_by_layer"][layer_name].to(torch.long)
        code_pairs = code_payload["pairs_by_layer"][layer_name].to(torch.long)
        if wiki_pairs.shape != code_pairs.shape:
            raise ValueError(f"Shape mismatch at {layer_name}: {wiki_pairs.shape} vs {code_pairs.shape}")
        wiki_idx = encode_pairs(wiki_pairs, args.total_num_experts, old_lookup)
        code_idx = encode_pairs(code_pairs, args.total_num_experts, new_lookup)
        flat_transition_idx = wiki_idx * len(new_pair_labels) + code_idx
        counts = torch.bincount(
            flat_transition_idx, minlength=len(old_pair_labels) * len(new_pair_labels)
        ).reshape(len(old_pair_labels), len(new_pair_labels))
        fractions = counts.float() / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
        aggregate_counts += counts
        layer_counts[layer_name] = counts.tolist()
        layer_row_fractions[layer_name] = fractions.numpy()
        layer_summaries[layer_name] = transition_summary(
            new_pairs=code_pairs, old_pairs=wiki_pairs, source_num_experts=args.source_num_experts
        )

    aggregate_row_fraction = (
        aggregate_counts.float() / aggregate_counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    ).numpy()

    manifest = {
        "compare_label": args.compare_label,
        "wiki_routing_pt": args.wiki_routing_pt,
        "code_routing_pt": args.code_routing_pt,
        "source_num_experts": args.source_num_experts,
        "total_num_experts": args.total_num_experts,
        "old_pair_labels": old_label_strings,
        "new_pair_labels": new_label_strings,
        "layer_summaries": layer_summaries,
    }
    save_json(output_root / "manifest.json", manifest)
    save_json(output_root / "layer_transition_counts.json", layer_counts)
    save_json(
        output_root / "layer_transition_row_fraction.json",
        {layer: matrix.tolist() for layer, matrix in layer_row_fractions.items()},
    )
    save_json(output_root / "aggregate_transition_row_fraction.json", aggregate_row_fraction.tolist())

    plot_aggregate_heatmap(
        aggregate_row_fraction,
        old_label_strings,
        new_label_strings,
        output_root / "aggregate_transition_heatmap.png",
        f"{args.compare_label}: aggregate top-2 routing transition",
    )
    plot_heatmaps(
        layer_row_fractions,
        old_label_strings,
        new_label_strings,
        output_root / "layer_transition_heatmaps.png",
        f"{args.compare_label}: layer-wise top-2 routing transition",
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
