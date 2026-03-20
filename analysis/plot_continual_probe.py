#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


PROBE_RE = re.compile(
    r"probe\s+(?P<name>\S+)\s+at iteration\s+(?P<iteration>\d+).*?"
    r"next_token_acc:\s+(?P<accuracy>[0-9.]+)\s+\|\s+ppl:\s+(?P<ppl>[0-9.E+-]+)"
)


def load_probe_points(path: Path, probe_name: str):
    points = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = PROBE_RE.search(line)
            if match is None or match.group("name") != probe_name:
                continue
            points.append(
                (
                    int(match.group("iteration")),
                    float(match.group("accuracy")),
                    float(match.group("ppl")),
                )
            )
    if not points:
        raise ValueError(f"No probe points named '{probe_name}' found in {path}")
    return points


def plot_metric(stage_a_points, stage_b_points, metric_index: int, ylabel: str, output: Path, title: str):
    plt.figure(figsize=(9, 5))
    plt.plot(
        [point[0] for point in stage_a_points],
        [point[metric_index] for point in stage_a_points],
        marker="o",
        linewidth=2,
        label="Stage A",
    )
    plt.plot(
        [point[0] for point in stage_b_points],
        [point[metric_index] for point in stage_b_points],
        marker="o",
        linewidth=2,
        label="Stage B",
    )
    plt.axvline(stage_a_points[-1][0], color="black", linestyle="--", linewidth=1, label="Stage switch")
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Task A probe metrics across Stage A and Stage B.")
    parser.add_argument("--stage-a-log", required=True)
    parser.add_argument("--stage-b-log", required=True)
    parser.add_argument("--probe-name", default="task_a_probe")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--title-prefix", default="Task A probe during continual learning")
    args = parser.parse_args()

    stage_a_log = Path(args.stage_a_log)
    stage_b_log = Path(args.stage_b_log)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    stage_a_points = load_probe_points(stage_a_log, args.probe_name)
    stage_b_points = load_probe_points(stage_b_log, args.probe_name)

    plot_metric(
        stage_a_points,
        stage_b_points,
        1,
        "Next-token accuracy",
        output_prefix.with_name(output_prefix.name + "_accuracy.png"),
        f"{args.title_prefix}: accuracy",
    )
    plot_metric(
        stage_a_points,
        stage_b_points,
        2,
        "Perplexity",
        output_prefix.with_name(output_prefix.name + "_ppl.png"),
        f"{args.title_prefix}: perplexity",
    )


if __name__ == "__main__":
    main()
