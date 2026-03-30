#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sorted_layers(summary):
    return sorted(summary.keys())


def plot_single_dataset(summary, output_dir: Path, stem: str, label: str):
    layers = sorted_layers(summary)
    all_old = [summary[layer]["fraction_all_old_group"] for layer in layers]
    mixed = [summary[layer]["fraction_mixed_old_new_group"] for layer in layers]
    all_new = [summary[layer]["fraction_all_new_group"] for layer in layers]
    new_usage = [summary[layer]["mean_new_group_slot_fraction"] for layer in layers]
    mean_selected = [summary[layer]["mean_selected_router_score"] for layer in layers]
    assignments = np.array(
        [summary[layer]["expert_assignment_fractions"] for layer in layers],
        dtype=np.float32,
    )

    plt.figure(figsize=(10, 5))
    plt.bar(layers, all_old, label="old only")
    plt.bar(layers, mixed, bottom=all_old, label="mixed")
    stacked_bottom = [old + mix for old, mix in zip(all_old, mixed)]
    plt.bar(layers, all_new, bottom=stacked_bottom, label="new only")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Fraction of tokens in routing")
    plt.title(f"Routing by expert group: {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_expert_group_usage.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(layers, new_usage, color="#59a14f")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Mean new-group slot fraction")
    plt.title(f"New-group slot usage: {label}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_new_group_slot_fraction.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(layers, mean_selected, color="#f28e2b")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean selected router score")
    plt.title(f"Mean selected router score: {label}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_selected_router_score.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 6))
    im = plt.imshow(assignments, aspect="auto", cmap="viridis")
    plt.yticks(np.arange(len(layers)), layers)
    plt.xticks(np.arange(assignments.shape[1]), np.arange(assignments.shape[1]))
    plt.xlabel("Expert index")
    plt.ylabel("Layer")
    plt.title(f"Expert assignment fractions: {label}")
    plt.colorbar(im, label="Assignment fraction")
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_expert_assignment_heatmap.png", dpi=220)
    plt.close()


def plot_dataset_comparison(wiki_summary, code_summary, output_dir: Path, title: str):
    wiki_layers = sorted_layers(wiki_summary)
    code_layers = sorted_layers(code_summary)
    if wiki_layers != code_layers:
        raise RuntimeError("Layer mismatch between wiki and code routing summaries.")

    layers = wiki_layers
    x = np.arange(len(layers))
    width = 0.36

    wiki_old = np.array([wiki_summary[layer]["fraction_all_old_group"] for layer in layers])
    wiki_mix = np.array([wiki_summary[layer]["fraction_mixed_old_new_group"] for layer in layers])
    wiki_new = np.array([wiki_summary[layer]["fraction_all_new_group"] for layer in layers])

    code_old = np.array([code_summary[layer]["fraction_all_old_group"] for layer in layers])
    code_mix = np.array([code_summary[layer]["fraction_mixed_old_new_group"] for layer in layers])
    code_new = np.array([code_summary[layer]["fraction_all_new_group"] for layer in layers])

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, wiki_old, width=width, label="wiki old only", color="#4e79a7")
    plt.bar(x - width / 2, wiki_mix, width=width, bottom=wiki_old, label="wiki mixed", color="#f28e2b")
    plt.bar(x - width / 2, wiki_new, width=width, bottom=wiki_old + wiki_mix, label="wiki new only", color="#59a14f")
    plt.bar(x + width / 2, code_old, width=width, label="code old only", color="#9c755f")
    plt.bar(x + width / 2, code_mix, width=width, bottom=code_old, label="code mixed", color="#ffbe7d")
    plt.bar(x + width / 2, code_new, width=width, bottom=code_old + code_mix, label="code new only", color="#8cd17d")
    plt.xticks(x, layers, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Fraction of tokens in routing")
    plt.title(f"Routing by dataset: {title}")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "dataset_comparison_expert_group_usage.png", dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot routing profile for a single experiment on wiki/code datasets.")
    parser.add_argument("--wiki-routing-json", required=True)
    parser.add_argument("--code-routing-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    wiki_summary = load_json(Path(args.wiki_routing_json))
    code_summary = load_json(Path(args.code_routing_json))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_single_dataset(wiki_summary, output_dir, "wiki", f"{args.title} | wiki test")
    plot_single_dataset(code_summary, output_dir, "code", f"{args.title} | code test")
    plot_dataset_comparison(wiki_summary, code_summary, output_dir, args.title)

    (output_dir / "routing_profile_summary.json").write_text(
        json.dumps({"wiki": wiki_summary, "code": code_summary}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
