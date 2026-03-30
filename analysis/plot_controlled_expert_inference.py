#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(input_dir: Path):
    rows = []
    for path in sorted(input_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "path": path,
                "label": data["label"],
                "acc": data["next_token_acc"],
                "ppl": data["ppl"],
            }
        )
    return rows


def parse_top1_vs_top2(rows):
    parsed = {}
    for row in rows:
        parts = row["label"].split("_")
        dataset = parts[-2]
        mode = parts[-1]
        parsed.setdefault(dataset, {})[mode] = row
    ordered_datasets = [d for d in ("wiki", "code") if d in parsed]
    ordered_modes = ["top2", "top1"]
    return parsed, ordered_datasets, ordered_modes


def parse_group_masks(rows):
    parsed = {}
    for row in rows:
        parts = row["label"].split("_")
        dataset = parts[1]
        mode = "_".join(parts[2:])
        parsed.setdefault(dataset, {})[mode] = row
    ordered_datasets = [d for d in ("wiki", "code") if d in parsed]
    ordered_modes = ["unrestricted", "wiki_only", "code_only"]
    return parsed, ordered_datasets, ordered_modes


def plot_grouped_bars(parsed, datasets, modes, metric_key, ylabel, title, output_path: Path):
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), squeeze=False)
    axes = axes[0]
    for ax, dataset in zip(axes, datasets):
        values = []
        labels = []
        for mode in modes:
            row = parsed.get(dataset, {}).get(mode)
            if row is None:
                continue
            values.append(row[metric_key])
            labels.append(mode)
        ax.bar(labels, values)
        ax.set_title(dataset)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--mode", required=True, choices=["top1_vs_top2", "group_masks"])
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    rows = load_results(input_dir)
    if not rows:
        raise SystemExit(f"No JSON results found under {input_dir}")

    if args.mode == "top1_vs_top2":
        parsed, datasets, modes = parse_top1_vs_top2(rows)
        title_prefix = "Top2 vs Top1"
    else:
        parsed, datasets, modes = parse_group_masks(rows)
        title_prefix = "Expert Group Masks"

    plot_grouped_bars(
        parsed,
        datasets,
        modes,
        "acc",
        "Next-token accuracy",
        f"{title_prefix}: accuracy",
        input_dir / "summary_accuracy.png",
    )
    plot_grouped_bars(
        parsed,
        datasets,
        modes,
        "ppl",
        "Perplexity",
        f"{title_prefix}: perplexity",
        input_dir / "summary_ppl.png",
    )


if __name__ == "__main__":
    main()
