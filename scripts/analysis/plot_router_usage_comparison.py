#!/usr/bin/env python3
"""Plot shared-router expert usage from probe TensorBoard logs."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalar_tags(event_dir):
    event_dir = Path(event_dir)
    accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    if tags:
        return accumulator, tags

    event_files = sorted(event_dir.rglob("events.out.tfevents*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under {event_dir}")
    accumulator = EventAccumulator(str(event_files[-1]), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    if not tags:
        raise RuntimeError(f"No scalar tags found in {event_dir}")
    return accumulator, tags


def latest_scalar(accumulator, tag):
    events = accumulator.Scalars(tag)
    if not events:
        raise KeyError(tag)
    return float(events[-1].value)


def infer_layer_numbers(tags, probe_name):
    prefix = f"{probe_name}/router/layer_"
    layers = set()
    for tag in tags:
        if not tag.startswith(prefix):
            continue
        rest = tag[len(prefix):]
        layer = rest.split("/", 1)[0]
        if layer.isdigit():
            layers.add(int(layer))
    return sorted(layers)


def load_router_usage(event_dir, probe_name, num_experts):
    accumulator, tags = load_scalar_tags(event_dir)
    tag_set = set(tags)

    usage = {}
    for name in ("old_expert_fraction", "new_expert_fraction", "new_expert_prob_mass"):
        tag = f"{probe_name}/router/{name}"
        if tag in tag_set:
            usage[name] = latest_scalar(accumulator, tag)

    expert_usage = []
    direct_expert_tags = [
        f"{probe_name}/router/expert_{expert_idx}_usage" for expert_idx in range(num_experts)
    ]
    if all(tag in tag_set for tag in direct_expert_tags):
        expert_usage = [latest_scalar(accumulator, tag) for tag in direct_expert_tags]
    else:
        layers = infer_layer_numbers(tags, probe_name)
        if not layers:
            raise RuntimeError(
                f"No expert usage tags found for probe '{probe_name}' in {event_dir}"
            )
        per_layer = []
        for layer in layers:
            values = []
            for expert_idx in range(num_experts):
                tag = f"{probe_name}/router/layer_{layer}/expert_{expert_idx}_usage"
                if tag in tag_set:
                    values.append(latest_scalar(accumulator, tag))
                else:
                    values.append(0.0)
            per_layer.append(values)
        expert_usage = np.asarray(per_layer, dtype=np.float64).mean(axis=0).tolist()

    layer_new_fraction = {}
    for layer in infer_layer_numbers(tags, probe_name):
        tag = f"{probe_name}/router/layer_{layer}/new_expert_fraction"
        if tag in tag_set:
            layer_new_fraction[layer] = latest_scalar(accumulator, tag)

    return {
        "summary": usage,
        "expert_usage": np.asarray(expert_usage, dtype=np.float64),
        "layer_new_fraction": layer_new_fraction,
    }


def setup_axis(ax, title, ylabel):
    ax.set_title(title, fontsize=14, fontweight="bold", pad=16)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_bar_labels(ax, bars, fmt="{:.2f}", padding=0.01):
    ymax = ax.get_ylim()[1]
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + ymax * padding,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_expert_usage(args, baseline, kd, output_path):
    experts = np.arange(args.num_experts)
    width = 0.38
    colors = ["#3F7CAC", "#C26D3A"]

    fig, ax = plt.subplots(figsize=(13.5, 6.8), dpi=args.dpi)
    ax.axvspan(-0.5, args.num_existing_experts - 0.5, color="#EAF3FA", alpha=0.8, zorder=0)
    ax.axvspan(args.num_existing_experts - 0.5, args.num_experts - 0.5, color="#FFF2E8", alpha=0.8, zorder=0)

    bars_a = ax.bar(
        experts - width / 2,
        baseline["expert_usage"],
        width,
        label=args.baseline_label,
        color=colors[0],
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    bars_b = ax.bar(
        experts + width / 2,
        kd["expert_usage"],
        width,
        label=args.kd_label,
        color=colors[1],
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )

    ax.axvline(args.num_existing_experts - 0.5, color="#555555", linestyle="--", linewidth=1.2)
    ax.text((args.num_existing_experts - 1) / 2, ax.get_ylim()[1] * 0.96, "old/wiki experts", ha="center", va="top", fontsize=10, color="#4B6475")
    ax.text((args.num_existing_experts + args.num_experts - 1) / 2, ax.get_ylim()[1] * 0.96, "new/code experts", ha="center", va="top", fontsize=10, color="#7A5134")
    ax.set_xticks(experts)
    ax.set_xticklabels([str(idx) for idx in experts])
    ax.set_xlabel("Expert index", fontsize=11)
    ax.set_xlim(-0.8, args.num_experts - 0.2)
    setup_axis(ax, args.title, "Top-k selection share")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False, fontsize=11)
    fig.subplots_adjust(top=0.82, left=0.08, right=0.98, bottom=0.12)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_summary(args, baseline, kd, output_path):
    metric_keys = ["old_expert_fraction", "new_expert_fraction", "new_expert_prob_mass"]
    metric_labels = ["Old top-k fraction", "New top-k fraction", "New prob mass"]
    x = np.arange(len(metric_keys))
    width = 0.36
    colors = ["#3F7CAC", "#C26D3A"]

    baseline_values = [baseline["summary"].get(key, 0.0) for key in metric_keys]
    kd_values = [kd["summary"].get(key, 0.0) for key in metric_keys]

    fig, ax = plt.subplots(figsize=(9.8, 6.2), dpi=args.dpi)
    bars_a = ax.bar(x - width / 2, baseline_values, width, label=args.baseline_label, color=colors[0], edgecolor="white", linewidth=0.8)
    bars_b = ax.bar(x + width / 2, kd_values, width, label=args.kd_label, color=colors[1], edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, max(1.0, max(baseline_values + kd_values) * 1.18))
    setup_axis(ax, "Router Usage Summary on Code Probe", "Fraction / probability mass")
    add_bar_labels(ax, bars_a)
    add_bar_labels(ax, bars_b)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False, fontsize=11)
    fig.subplots_adjust(top=0.80, left=0.10, right=0.97, bottom=0.16)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_layer_new_fraction(args, baseline, kd, output_path):
    layers = sorted(set(baseline["layer_new_fraction"]) | set(kd["layer_new_fraction"]))
    if not layers:
        return

    x = np.arange(len(layers))
    width = 0.36
    colors = ["#3F7CAC", "#C26D3A"]
    baseline_values = [baseline["layer_new_fraction"].get(layer, 0.0) for layer in layers]
    kd_values = [kd["layer_new_fraction"].get(layer, 0.0) for layer in layers]

    fig, ax = plt.subplots(figsize=(10.8, 6.2), dpi=args.dpi)
    ax.bar(x - width / 2, baseline_values, width, label=args.baseline_label, color=colors[0], edgecolor="white", linewidth=0.8)
    ax.bar(x + width / 2, kd_values, width, label=args.kd_label, color=colors[1], edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(layer) for layer in layers])
    ax.set_xlabel("Layer number", fontsize=11)
    ax.set_ylim(0, max(1.0, max(baseline_values + kd_values) * 1.18))
    setup_axis(ax, "New/Code Expert Top-k Fraction by Layer", "New expert top-k fraction")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False, fontsize=11)
    fig.subplots_adjust(top=0.80, left=0.10, right=0.97, bottom=0.15)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True, help="TensorBoard dir for the no-KD G2 diagnostic run.")
    parser.add_argument("--kd-dir", required=True, help="TensorBoard dir for the KD diagnostic run.")
    parser.add_argument("--output-dir", required=True, help="Directory to write PNG plots.")
    parser.add_argument("--probe-name", default="code_probe")
    parser.add_argument("--baseline-label", default="G2 no KD")
    parser.add_argument("--kd-label", default="G2 RouterKD lambda=10")
    parser.add_argument("--title", default="Code Probe Shared-Router Expert Usage")
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--num-existing-experts", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_router_usage(args.baseline_dir, args.probe_name, args.num_experts)
    kd = load_router_usage(args.kd_dir, args.probe_name, args.num_experts)

    plot_expert_usage(args, baseline, kd, output_dir / "router_usage_expert_bar.png")
    plot_summary(args, baseline, kd, output_dir / "router_usage_summary_bar.png")
    plot_layer_new_fraction(args, baseline, kd, output_dir / "router_usage_layer_new_fraction_bar.png")

    print(f"wrote {output_dir / 'router_usage_expert_bar.png'}")
    print(f"wrote {output_dir / 'router_usage_summary_bar.png'}")
    if (output_dir / "router_usage_layer_new_fraction_bar.png").exists():
        print(f"wrote {output_dir / 'router_usage_layer_new_fraction_bar.png'}")


if __name__ == "__main__":
    main()
