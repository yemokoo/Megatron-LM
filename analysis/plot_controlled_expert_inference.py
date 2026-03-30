#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def infer_label_from_filename(path: Path):
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 3:
        return "_".join(parts[-3:]) if parts[-2] in ("wiki", "code") else "_".join(parts[-2:])
    return stem


def load_results(input_dir: Path):
    rows = []
    for path in sorted(input_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        label = data.get("label", infer_label_from_filename(path))
        acc = data.get("next_token_acc", data.get("acc"))
        ppl = data.get("ppl")
        if acc is None or ppl is None:
            continue
        rows.append(
            {
                "path": path,
                "label": label,
                "acc": acc,
                "ppl": ppl,
            }
        )
    return rows


def parse_top1_vs_top2(rows):
    parsed = {}
    for row in rows:
        parts = row["label"].split("_")
        dataset = next((part for part in parts if part in ("wiki", "code")), None)
        mode = next((part for part in parts if part in ("top1", "top2")), None)
        if dataset is None or mode is None:
            continue
        parsed.setdefault(dataset, {})[mode] = row
    ordered_datasets = [d for d in ("wiki", "code") if d in parsed]
    ordered_modes = ["top2", "top1"]
    return parsed, ordered_datasets, ordered_modes


def parse_group_masks(rows):
    parsed = {}
    for row in rows:
        parts = row["label"].split("_")
        dataset = next((part for part in parts if part in ("wiki", "code")), None)
        mode = next((candidate for candidate in ("unrestricted", "wiki_only", "code_only") if candidate in row["label"]), None)
        if dataset is None or mode is None:
            continue
        parsed.setdefault(dataset, {})[mode] = row
    ordered_datasets = [d for d in ("wiki", "code") if d in parsed]
    ordered_modes = ["unrestricted", "wiki_only", "code_only"]
    return parsed, ordered_datasets, ordered_modes


def plot_grouped_bars(parsed, datasets, modes, metric_key, ylabel, title, output_path: Path):
    color_map = {
        "top2": "#4C78A8",
        "top1": "#F58518",
        "unrestricted": "#4C78A8",
        "wiki_only": "#54A24B",
        "code_only": "#E45756",
    }
    legend_map = {
        "top2": "Top-2 routing",
        "top1": "Top-1 routing",
        "unrestricted": "Unrestricted",
        "wiki_only": "Wiki experts only",
        "code_only": "Code experts only",
    }
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), squeeze=False)
    axes = axes[0]
    legend_handles = []
    for ax, dataset in zip(axes, datasets):
        values = []
        labels = []
        colors = []
        for mode in modes:
            row = parsed.get(dataset, {}).get(mode)
            if row is None:
                continue
            values.append(row[metric_key])
            labels.append(mode)
            colors.append(color_map.get(mode, "#888888"))
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(f"{dataset} dataset")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        for bar, value in zip(bars, values):
            bar_height = bar.get_height()
            y_min, y_max = ax.get_ylim()
            axis_span = max(y_max - y_min, 1e-8)
            inside_margin = axis_span * 0.02
            if bar_height > axis_span * 0.08:
                y_pos = max(bar_height - inside_margin, y_min + inside_margin)
                va = "top"
                color = "white"
            else:
                y_pos = bar_height + inside_margin
                va = "bottom"
                color = "black"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f"{value:.4f}",
                ha="center",
                va=va,
                fontsize=10,
                color=color,
                fontweight="bold",
            )
        for bar, mode in zip(bars, labels):
            if all(existing.get_label() != legend_map.get(mode, mode) for existing in legend_handles):
                bar.set_label(legend_map.get(mode, mode))
                legend_handles.append(bar)
    fig.suptitle(title)
    show_legend = not (set(modes) == {"top1", "top2"} or set(modes) == {"top2", "top1"})
    if legend_handles and show_legend:
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=min(len(legend_handles), 3),
            frameon=False,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.92])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.95])
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
