#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sorted_layers(summary):
    return sorted(summary["routing_summary"].keys())


def plot_single_dataset(summary, output_dir: Path, stem: str):
    layers = sorted_layers(summary)
    all_old = [summary["routing_summary"][layer]["fraction_all_old_group"] for layer in layers]
    mixed = [summary["routing_summary"][layer]["fraction_mixed_old_new_group"] for layer in layers]
    all_new = [summary["routing_summary"][layer]["fraction_all_new_group"] for layer in layers]
    new_usage = [summary["routing_summary"][layer]["fraction_using_new_expert"] for layer in layers]
    router_logits = np.array(
        [summary["routing_summary"][layer]["mean_router_logits_per_expert"] for layer in layers],
        dtype=np.float32,
    )

    plt.figure(figsize=(10, 5))
    plt.bar(layers, all_old, label="old4 only")
    plt.bar(layers, mixed, bottom=all_old, label="mixed old4+new3")
    stacked_bottom = [old + mix for old, mix in zip(all_old, mixed)]
    plt.bar(layers, all_new, bottom=stacked_bottom, label="new3 only")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Fraction of tokens in routing")
    plt.title(f"Routing by expert group: {summary['label']}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_expert_group_usage.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(layers, new_usage)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Fraction using new expert")
    plt.title(f"New expert usage: {summary['label']}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_new_expert_usage.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    im = plt.imshow(router_logits, aspect="auto", cmap="viridis")
    plt.yticks(np.arange(len(layers)), layers)
    plt.xticks(np.arange(router_logits.shape[1]), np.arange(router_logits.shape[1]))
    plt.xlabel("Expert index")
    plt.ylabel("Layer")
    plt.title(f"Mean router logits: {summary['label']}")
    plt.colorbar(im, label="Mean router logit")
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_router_logit_heatmap.png", dpi=200)
    plt.close()


def plot_dataset_comparison(wiki_summary, code_summary, output_dir: Path):
    wiki_layers = sorted_layers(wiki_summary)
    code_layers = sorted_layers(code_summary)
    if wiki_layers != code_layers:
        raise RuntimeError("Layer mismatch between wiki and code routing summaries.")

    layers = wiki_layers
    x = np.arange(len(layers))
    width = 0.36

    wiki_old = np.array([wiki_summary["routing_summary"][layer]["fraction_all_old_group"] for layer in layers])
    wiki_mix = np.array([wiki_summary["routing_summary"][layer]["fraction_mixed_old_new_group"] for layer in layers])
    wiki_new = np.array([wiki_summary["routing_summary"][layer]["fraction_all_new_group"] for layer in layers])

    code_old = np.array([code_summary["routing_summary"][layer]["fraction_all_old_group"] for layer in layers])
    code_mix = np.array([code_summary["routing_summary"][layer]["fraction_mixed_old_new_group"] for layer in layers])
    code_new = np.array([code_summary["routing_summary"][layer]["fraction_all_new_group"] for layer in layers])

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, wiki_old, width=width, label="wiki old4 only", color="#4e79a7")
    plt.bar(x - width / 2, wiki_mix, width=width, bottom=wiki_old, label="wiki mixed", color="#f28e2b")
    plt.bar(x - width / 2, wiki_new, width=width, bottom=wiki_old + wiki_mix, label="wiki new3 only", color="#59a14f")
    plt.bar(x + width / 2, code_old, width=width, label="code old4 only", color="#9c755f")
    plt.bar(x + width / 2, code_mix, width=width, bottom=code_old, label="code mixed", color="#ffbe7d")
    plt.bar(x + width / 2, code_new, width=width, bottom=code_old + code_mix, label="code new3 only", color="#8cd17d")
    plt.xticks(x, layers, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Fraction of tokens in routing")
    plt.title("QV LoRA routing by dataset")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "expert_group_usage_comparison.png", dpi=200)
    plt.close()

    wiki_new_usage = [wiki_summary["routing_summary"][layer]["fraction_using_new_expert"] for layer in layers]
    code_new_usage = [code_summary["routing_summary"][layer]["fraction_using_new_expert"] for layer in layers]

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, wiki_new_usage, width=width, label="wiki")
    plt.bar(x + width / 2, code_new_usage, width=width, label="code")
    plt.xticks(x, layers, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Fraction using new expert")
    plt.title("New expert usage by dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "new_expert_usage_comparison.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-summary", required=True)
    parser.add_argument("--code-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    wiki_summary = load_summary(Path(args.wiki_summary))
    code_summary = load_summary(Path(args.code_summary))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_single_dataset(wiki_summary, output_dir, "wiki")
    plot_single_dataset(code_summary, output_dir, "code")
    plot_dataset_comparison(wiki_summary, code_summary, output_dir)

    combined = {
        "wiki": wiki_summary,
        "code": code_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
