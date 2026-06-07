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
    parser.add_argument(
        "--bar-step-percent",
        type=float,
        default=None,
        help=(
            "Generate evenly-spaced bar milestones from this percent to 100. "
            "For example, 5 produces 5,10,...,100 and overrides --bar-percents."
        ),
    )
    parser.add_argument(
        "--bar-min-y-span",
        type=float,
        default=0.006,
        help="Minimum y-axis span for bar plots so tiny noise is not visually over-amplified.",
    )
    parser.add_argument(
        "--bar-value-font-size",
        type=float,
        default=8.5,
        help="Font size for value labels on bar plots.",
    )
    parser.add_argument(
        "--bar-label-position",
        choices=("outside", "inside", "none"),
        default="outside",
        help="Where to draw per-bar value labels.",
    )
    parser.add_argument(
        "--bar-fig-width",
        type=float,
        default=None,
        help="Optional explicit bar-plot figure width in inches.",
    )
    parser.add_argument(
        "--bar-fig-height",
        type=float,
        default=7.0,
        help="Bar-plot figure height in inches.",
    )
    parser.add_argument(
        "--plateau-percent",
        type=float,
        default=25.0,
        help="Percent where the optional post-percent range band starts.",
    )
    parser.add_argument(
        "--no-plateau-band",
        action="store_true",
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--show-plateau-band",
        action="store_false",
        dest="no_plateau_band",
        help="Draw the post-percent min/max band and mean line.",
    )
    parser.add_argument(
        "--reference-percent",
        type=float,
        default=100.0,
        help="Reference percent for convergence band, usually the 100%% result.",
    )
    parser.add_argument(
        "--convergence-tolerance",
        type=float,
        default=0.0005,
        help="Accuracy tolerance around the reference value used to mark early plateau.",
    )
    parser.add_argument(
        "--no-convergence-band",
        action="store_true",
        help="Do not draw the reference +/- tolerance band or first-within-tolerance marker.",
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
            local_iteration = int(match.group("local_iteration"))
            # The displayed probe iteration may include PROBE_STEP_OFFSET for W&B
            # alignment. local_iteration is the actual checkpoint/training step.
            retune_step = local_iteration - source_step
            if retune_step < 0:
                continue
            points.append(
                ProbePoint(
                    probe=match.group("probe"),
                    iteration=iteration,
                    local_iteration=local_iteration,
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


def parse_bar_percents(args: argparse.Namespace) -> List[float]:
    if args.bar_step_percent is not None:
        if args.bar_step_percent <= 0 or args.bar_step_percent > 100:
            raise ValueError("--bar-step-percent must be in (0, 100].")
        percents = []
        current = args.bar_step_percent
        while current <= 100.0 + 1e-9:
            percents.append(round(current, 6))
            current += args.bar_step_percent
        if percents[-1] != 100.0:
            percents.append(100.0)
        return percents
    return [float(value.strip()) for value in args.bar_percents.split(",") if value.strip()]


def percent_suffix(percents: Sequence[float]) -> str:
    if len(percents) >= 2:
        diffs = [
            round(percents[idx + 1] - percents[idx], 6)
            for idx in range(len(percents) - 1)
        ]
        if max(diffs) - min(diffs) < 1e-6:
            step = diffs[0]
            return f"every_{step:g}pct".replace(".", "p")
    return "selected"


def percent_step(percents: Sequence[float]) -> float | None:
    if len(percents) < 2:
        return None
    diffs = [
        round(percents[idx + 1] - percents[idx], 6)
        for idx in range(len(percents) - 1)
    ]
    if max(diffs) - min(diffs) < 1e-6:
        return diffs[0]
    return None


def bar_title_suffix(percents: Sequence[float]) -> str:
    step = percent_step(percents)
    if step is not None:
        return f"Every {step:g}%"
    return "Selected Milestones"


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


def plot_single_probe_line(
    points: Sequence[ProbePoint], probe: str, out_base: Path, write_png: bool
) -> None:
    plt = maybe_import_matplotlib()
    probe_points = by_probe(points).get(probe, [])
    if not probe_points:
        raise ValueError(f"no points for {probe}")

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(
        [point.data_percent for point in probe_points],
        [point.next_token_acc for point in probe_points],
        marker="o",
        markersize=3.0,
        linewidth=2.4,
        color=PROBE_COLOR[probe],
        label=PROBE_LABEL[probe],
    )
    ax.set_title(
        f"Router-Only Finetune Data Budget: {PROBE_LABEL[probe]} Accuracy Every 1%",
        fontsize=18,
        weight="bold",
    )
    ax.set_xlabel("Retune data budget (%)")
    ax.set_ylabel("Next-token accuracy")
    ax.set_xlim(1, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".svg"))
    if write_png:
        fig.savefig(out_base.with_suffix(".png"), dpi=180)
    plt.close(fig)


def plot_single_probe_bar(
    points: Sequence[ProbePoint],
    probe: str,
    percents: Sequence[float],
    out_base: Path,
    write_png: bool,
    min_y_span: float,
    value_font_size: float,
    label_position: str,
    fig_width: float | None,
    fig_height: float,
    plateau_percent: float,
    show_plateau_band: bool,
    reference_percent: float,
    convergence_tolerance: float,
    show_convergence_band: bool,
) -> None:
    plt = maybe_import_matplotlib()
    all_probe_points = by_probe(points)[probe]
    probe_points = nearest_points(all_probe_points, percents)
    values = [point.next_token_acc for point in probe_points]
    reference_point = min(all_probe_points, key=lambda point: abs(point.data_percent - reference_percent))
    reference_value = reference_point.next_token_acc
    convergence_low = reference_value - convergence_tolerance
    convergence_high = reference_value + convergence_tolerance
    first_converged_index = None
    for idx, point in enumerate(probe_points):
        if abs(point.next_token_acc - reference_value) <= convergence_tolerance:
            first_converged_index = idx
            break
    plateau_values = [
        point.next_token_acc
        for point in all_probe_points
        if point.data_percent >= plateau_percent
    ]
    if not plateau_values:
        plateau_values = values
    band_values = [convergence_low, convergence_high] if show_convergence_band else []
    value_min = min(values + plateau_values + band_values)
    value_max = max(values + plateau_values + band_values)
    value_range = value_max - value_min
    y_span = max(value_range * 1.8, min_y_span)
    y_mid = (value_min + value_max) / 2
    y_floor = y_mid - y_span / 2
    y_top = y_mid + y_span / 2

    x = list(range(len(percents)))
    # Long milestone bars need horizontal space; otherwise 20 labels collapse.
    width = fig_width if fig_width is not None else max(16.0, len(percents) * 0.95)
    fig, ax = plt.subplots(figsize=(width, fig_height))

    if show_convergence_band:
        ax.axhspan(
            convergence_low,
            convergence_high,
            color="#2563eb",
            alpha=0.10,
            label=f"{reference_percent:g}% +/- {convergence_tolerance:g}",
            zorder=0,
        )
        ax.axhline(
            reference_value,
            color="#2563eb",
            linestyle="--",
            linewidth=1.6,
            alpha=0.85,
            label=f"{reference_percent:g}% reference",
            zorder=1,
        )
        if first_converged_index is not None:
            first_point = probe_points[first_converged_index]
            ax.axvline(
                first_converged_index,
                color="#111827",
                linestyle=":",
                linewidth=1.6,
                alpha=0.85,
                zorder=1,
            )
            ax.text(
                first_converged_index,
                y_top - y_span * 0.04,
                f"within {convergence_tolerance:g}\nfrom {first_point.data_percent:g}%",
                ha="center",
                va="top",
                color="#111827",
                fontsize=max(value_font_size, 8.0),
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.88, "pad": 3},
                zorder=4,
            )

    if show_plateau_band:
        band_min = min(plateau_values)
        band_max = max(plateau_values)
        band_mean = sum(plateau_values) / len(plateau_values)
        ax.axhspan(
            band_min,
            band_max,
            color=PROBE_COLOR[probe],
            alpha=0.12,
            label=f">= {plateau_percent:g}% range",
            zorder=0,
        )
        ax.axhline(
            band_mean,
            color="#111827",
            linestyle="--",
            linewidth=1.6,
            alpha=0.8,
            label=f">= {plateau_percent:g}% mean",
            zorder=1,
        )
        plateau_idx = min(range(len(percents)), key=lambda idx: abs(percents[idx] - plateau_percent))
        ax.axvline(
            plateau_idx - 0.5,
            color="#6b7280",
            linestyle=":",
            linewidth=1.4,
            alpha=0.75,
        )

    bars = ax.bar(
        x,
        [value - y_floor for value in values],
        bottom=y_floor,
        width=0.64,
        color=PROBE_COLOR[probe],
        label=PROBE_LABEL[probe],
        zorder=2,
    )
    if label_position != "none":
        for bar, value in zip(bars, values):
            if label_position == "outside":
                y = min(value + y_span * 0.025, y_top - y_span * 0.02)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y,
                    f"{value:.4f}",
                    ha="center",
                    va="bottom",
                    rotation=90 if len(percents) > 12 else 0,
                    color="#111827",
                    fontsize=value_font_size,
                    fontweight="bold",
                    clip_on=False,
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value - y_span * 0.035,
                    f"{value:.4f}",
                    ha="center",
                    va="top",
                    color="white",
                    fontsize=value_font_size,
                    fontweight="bold",
                )
    ax.set_title(
        f"Router-Only Finetune Data Budget: {PROBE_LABEL[probe]} {bar_title_suffix(percents)}",
        fontsize=18,
        weight="bold",
    )
    ax.set_xlabel("Retune data budget (%)")
    ax.set_ylabel("Next-token accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{percent:g}%" for percent in percents], rotation=0)
    ax.set_ylim(y_floor, y_top)
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
    percents = parse_bar_percents(args)
    bar_suffix = percent_suffix(percents)

    points = parse_probe_points(log_path, source_step, args.full_retune_iters)
    write_csv(points, out_dir / "router_data_budget_probe_points.csv")
    write_milestone_csv(points, percents, out_dir / "router_data_budget_probe_milestones.csv")
    plot_single_probe_line(points, "wiki_probe", out_dir / "router_data_budget_wiki_accuracy_line", not args.no_png)
    plot_single_probe_line(points, "code_probe", out_dir / "router_data_budget_code_accuracy_line", not args.no_png)
    plot_single_probe_bar(
        points,
        "wiki_probe",
        percents,
        out_dir / f"router_data_budget_wiki_accuracy_bars_{bar_suffix}",
        not args.no_png,
        args.bar_min_y_span,
        args.bar_value_font_size,
        args.bar_label_position,
        args.bar_fig_width,
        args.bar_fig_height,
        args.plateau_percent,
        not args.no_plateau_band,
        args.reference_percent,
        args.convergence_tolerance,
        not args.no_convergence_band,
    )
    plot_single_probe_bar(
        points,
        "code_probe",
        percents,
        out_dir / f"router_data_budget_code_accuracy_bars_{bar_suffix}",
        not args.no_png,
        args.bar_min_y_span,
        args.bar_value_font_size,
        args.bar_label_position,
        args.bar_fig_width,
        args.bar_fig_height,
        args.plateau_percent,
        not args.no_plateau_band,
        args.reference_percent,
        args.convergence_tolerance,
        not args.no_convergence_band,
    )

    print(f"[DONE] parsed {len(points)} probe points from {log_path}")
    print(f"[DONE] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
