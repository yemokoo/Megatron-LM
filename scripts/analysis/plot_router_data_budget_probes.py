#!/usr/bin/env python3
"""Plot router-retune data-budget probe curves from a phase3 log."""

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


PROBE_RE = re.compile(
    r"probe\s+(?P<probe>code_probe|wiki_probe)\s+at iteration\s+(?P<iteration>\d+)\s+\|\s+"
    r"local_iteration:\s+(?P<local_iteration>\d+)\s+\|\s+"
    r"next_token_acc:\s+(?P<acc>[0-9.]+)\s+\|\s+ppl:\s+(?P<ppl>[0-9.Ee+-]+)"
)

PROBE_LABEL = {
    "code_probe": "Code probe",
    "wiki_probe": "Wiki probe",
}

PROBE_COLOR = {
    "code_probe": "#E76F51",
    "wiki_probe": "#2A9D8F",
}


@dataclass(frozen=True)
class ProbePoint:
    probe: str
    iteration: int
    local_iteration: int
    retune_step: int
    data_percent: float
    next_token_acc: float
    ppl: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot wiki/code probe accuracy over router-retune data-budget percent."
    )
    parser.add_argument(
        "--run-dir",
        default=(
            ".local/weights/a100/mha/g2-checkpoints/code/phase3_data_budget/"
            "g2-exp2-phase3-router-only-retune-wikicode-data-budget-100pct-checkpoints-"
            "from-new-experts-all-router-no-reinit-mb72-3600"
        ),
        help="Router data-budget run directory containing logs/phase3_run.log.",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="Explicit phase3_run.log path. Overrides --run-dir.",
    )
    parser.add_argument(
        "--out-dir",
        default="analysis_outputs/router_data_budget_probes",
        help="Output directory for CSV and plots.",
    )
    parser.add_argument(
        "--source-step",
        type=int,
        default=None,
        help="Actual checkpoint iteration at retune start. Defaults to PHASE3_SOURCE.txt or 1800.",
    )
    parser.add_argument(
        "--full-retune-iters",
        type=int,
        default=3600,
        help="Retune steps corresponding to 100%% of wiki/code data.",
    )
    parser.add_argument(
        "--bar-percents",
        default="1,5,10,25,50,100",
        help="Comma-separated x-axis percent milestones for the bar plot.",
    )
    parser.add_argument("--no-png", action="store_true", help="Skip optional PNG generation.")
    return parser.parse_args()


def read_source_step(run_dir: Path, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    marker = run_dir / "PHASE3_SOURCE.txt"
    if marker.exists():
        for line in marker.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("source_step="):
                return int(line.split("=", 1)[1].strip())
    return 1800


def parse_probe_points(log_path: Path, source_step: int, full_retune_iters: int) -> List[ProbePoint]:
    if not log_path.exists():
        raise FileNotFoundError(f"missing log: {log_path}")

    points: List[ProbePoint] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = PROBE_RE.search(line)
            if match is None:
                continue
            iteration = int(match.group("iteration"))
            retune_step = iteration - source_step
            if retune_step < 0:
                continue
            points.append(
                ProbePoint(
                    probe=match.group("probe"),
                    iteration=iteration,
                    local_iteration=int(match.group("local_iteration")),
                    retune_step=retune_step,
                    data_percent=retune_step / full_retune_iters * 100.0,
                    next_token_acc=float(match.group("acc")),
                    ppl=float(match.group("ppl")),
                )
            )
    if not points:
        raise ValueError(f"no probe points found in {log_path}")
    return points


def by_probe(points: Iterable[ProbePoint]) -> Dict[str, List[ProbePoint]]:
    grouped: Dict[str, List[ProbePoint]] = {}
    for point in points:
        grouped.setdefault(point.probe, []).append(point)
    for probe_points in grouped.values():
        probe_points.sort(key=lambda point: point.retune_step)
    return grouped


def write_csv(points: Sequence[ProbePoint], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "probe",
                "iteration",
                "local_iteration",
                "retune_step",
                "data_percent",
                "next_token_acc",
                "ppl",
            ]
        )
        for point in sorted(points, key=lambda item: (item.probe, item.retune_step)):
            writer.writerow(
                [
                    point.probe,
                    point.iteration,
                    point.local_iteration,
                    point.retune_step,
                    f"{point.data_percent:.6f}",
                    f"{point.next_token_acc:.6f}",
                    f"{point.ppl:.6f}",
                ]
            )


def nearest_points(points: Sequence[ProbePoint], percents: Sequence[float]) -> List[ProbePoint]:
    selected = []
    for percent in percents:
        selected.append(min(points, key=lambda point: abs(point.data_percent - percent)))
    return selected


def write_milestone_csv(points: Sequence[ProbePoint], percents: Sequence[float], csv_path: Path) -> None:
    grouped = by_probe(points)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "target_percent",
                "probe",
                "actual_percent",
                "iteration",
                "retune_step",
                "next_token_acc",
                "ppl",
            ]
        )
        for target_percent in percents:
            for probe in ("wiki_probe", "code_probe"):
                point = min(grouped[probe], key=lambda item: abs(item.data_percent - target_percent))
                writer.writerow(
                    [
                        f"{target_percent:g}",
                        probe,
                        f"{point.data_percent:.6f}",
                        point.iteration,
                        point.retune_step,
                        f"{point.next_token_acc:.6f}",
                        f"{point.ppl:.6f}",
                    ]
                )


def maybe_import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_line(points: Sequence[ProbePoint], out_base: Path, write_png: bool) -> None:
    plt = maybe_import_matplotlib()
    grouped = by_probe(points)

    fig, ax = plt.subplots(figsize=(13, 7))
    for probe in ("wiki_probe", "code_probe"):
        probe_points = grouped.get(probe, [])
        ax.plot(
            [point.data_percent for point in probe_points],
            [point.next_token_acc for point in probe_points],
            marker="o",
            markersize=3.0,
            linewidth=2.2,
            color=PROBE_COLOR[probe],
            label=PROBE_LABEL[probe],
        )
    ax.set_title("Router-Only Finetune Data Budget: Probe Accuracy Every 1%", fontsize=18, weight="bold")
    ax.set_xlabel("Retune data budget (%)")
    ax.set_ylabel("Next-token accuracy")
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".svg"))
    if write_png:
        fig.savefig(out_base.with_suffix(".png"), dpi=180)
    plt.close(fig)


def plot_bar(points: Sequence[ProbePoint], percents: Sequence[float], out_base: Path, write_png: bool) -> None:
    plt = maybe_import_matplotlib()
    grouped = by_probe(points)
    wiki_points = nearest_points(grouped["wiki_probe"], percents)
    code_points = nearest_points(grouped["code_probe"], percents)

    x = list(range(len(percents)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(
        [value - width / 2 for value in x],
        [point.next_token_acc for point in wiki_points],
        width=width,
        color=PROBE_COLOR["wiki_probe"],
        label=PROBE_LABEL["wiki_probe"],
    )
    ax.bar(
        [value + width / 2 for value in x],
        [point.next_token_acc for point in code_points],
        width=width,
        color=PROBE_COLOR["code_probe"],
        label=PROBE_LABEL["code_probe"],
    )
    ax.set_title("Router-Only Finetune Data Budget: Selected Probe Milestones", fontsize=18, weight="bold")
    ax.set_xlabel("Retune data budget (%)")
    ax.set_ylabel("Next-token accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{percent:g}%" for percent in percents])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".svg"))
    if write_png:
        fig.savefig(out_base.with_suffix(".png"), dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    log_path = Path(args.log) if args.log else run_dir / "logs" / "phase3_run.log"
    source_step = read_source_step(run_dir, args.source_step)
    out_dir = Path(args.out_dir)
    percents = [float(value.strip()) for value in args.bar_percents.split(",") if value.strip()]

    points = parse_probe_points(log_path, source_step, args.full_retune_iters)
    write_csv(points, out_dir / "router_data_budget_probe_points.csv")
    write_milestone_csv(points, percents, out_dir / "router_data_budget_probe_milestones.csv")
    plot_line(points, out_dir / "router_data_budget_probe_accuracy_line", not args.no_png)
    plot_bar(points, percents, out_dir / "router_data_budget_probe_accuracy_bars", not args.no_png)

    print(f"[DONE] parsed {len(points)} probe points from {log_path}")
    print(f"[DONE] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
