#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def add_value_labels(ax, bars, fmt):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=10,
        )


def main():
    parser = argparse.ArgumentParser(description="Plot shared-vs-split router probe results.")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--title", type=str, default="Shared vs Split Router Clone")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    shared = load_json(input_dir / "shared_router.json")
    split = load_json(input_dir / "split_router_clone.json")

    labels = ["Shared Router", "Split Router Clone"]
    colors = ["#4C78A8", "#F58518"]
    acc_values = [shared["next_token_acc"], split["next_token_acc"]]
    ppl_values = [shared["ppl"], split["ppl"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8))
    fig.suptitle(args.title, fontsize=14, y=0.98)

    acc_bars = axes[0].bar(labels, acc_values, color=colors, width=0.62)
    axes[0].set_title("Next-Token Accuracy", fontsize=12, pad=10)
    axes[0].set_ylabel("Accuracy")
    axes[0].tick_params(axis="x", rotation=10)
    add_value_labels(axes[0], acc_bars, "{:.4f}")

    ppl_bars = axes[1].bar(labels, ppl_values, color=colors, width=0.62)
    axes[1].set_title("Perplexity", fontsize=12, pad=10)
    axes[1].set_ylabel("PPL")
    axes[1].tick_params(axis="x", rotation=10)
    add_value_labels(axes[1], ppl_bars, "{:.3f}")

    legend_handles = [Patch(facecolor=color, label=label) for color, label in zip(colors, labels)]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.92),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.84])

    output_path = Path(args.output) if args.output else input_dir / "summary_bars.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
