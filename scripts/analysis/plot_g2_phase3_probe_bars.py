#!/usr/bin/env python3
"""Plot G2 phase1 vs phase3 probe metrics from training logs.

This script intentionally uses only the Python standard library and writes SVG
directly so it can run on KT nodes without installing matplotlib.
"""

import argparse
import csv
import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PROBE_RE = re.compile(
    r"probe\s+(?P<probe>code_probe|wiki_probe)\s+at iteration\s+(?P<iteration>\d+)\s+\|\s+"
    r"local_iteration:\s+(?P<local_iteration>\d+)\s+\|\s+"
    r"next_token_acc:\s+(?P<acc>[0-9.]+)\s+\|\s+ppl:\s+(?P<ppl>[0-9.Ee+-]+)"
)


@dataclass(frozen=True)
class RunSpec:
    experiment: str
    stage: str
    run_id: str
    log_name: str


@dataclass(frozen=True)
class ProbeResult:
    experiment: str
    stage: str
    probe: str
    iteration: int
    local_iteration: int
    next_token_acc: float
    ppl: float
    log_path: Path


RUNS = (
    RunSpec(
        experiment="exp1",
        stage="phase1_before_router_retune",
        run_id=(
            "code/phase1/"
            "g2-top4-e8to16-ffn352-r256-wiki-to-code-shared-router-qkvo-all-experts-router-"
            "mha-a100-bf16-mb72-1800"
        ),
        log_name="logs/run.log",
    ),
    RunSpec(
        experiment="exp1",
        stage="phase3_after_router_retune",
        run_id=(
            "code/phase3/"
            "g2-exp1-phase3-router-only-retune-wikicode-from-all-experts-router-no-reinit-"
            "mb72-1800"
        ),
        log_name="logs/phase3_run.log",
    ),
    RunSpec(
        experiment="exp2",
        stage="phase1_before_router_retune",
        run_id=(
            "code/phase1/"
            "g2-exp2-top4-e8to16-ffn352-r256-wiki-to-code-new-experts-all-router-"
            "mha-a100-bf16-mb72-1800"
        ),
        log_name="logs/run.log",
    ),
    RunSpec(
        experiment="exp2",
        stage="phase3_after_router_retune",
        run_id=(
            "code/phase3/"
            "g2-exp2-phase3-router-only-retune-wikicode-from-new-experts-all-router-no-reinit-"
            "mb72-1800"
        ),
        log_name="logs/phase3_run.log",
    ),
)

STAGE_LABEL = {
    "phase1_before_router_retune": "Before retune",
    "phase3_after_router_retune": "After retune",
}

PROBE_LABEL = {
    "code_probe": "Code",
    "wiki_probe": "Wiki",
}

METRIC_LABEL = {
    "next_token_acc": "Next-token accuracy",
    "ppl": "Perplexity",
}

BAR_COLORS = {
    "phase1_before_router_retune": "#415A77",
    "phase3_after_router_retune": "#D97706",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create bar charts comparing G2 phase1 and phase3 probe metrics from logs."
    )
    parser.add_argument(
        "--g2-root",
        default=".local/weights/a100/mha/g2-checkpoints",
        help="Path to the G2 checkpoint registry root.",
    )
    parser.add_argument(
        "--out-dir",
        default="analysis_outputs/g2_phase3_log_probe_bars",
        help="Output directory for CSV, markdown, and SVG files.",
    )
    return parser.parse_args()


def parse_last_probe_results(log_path: Path, experiment: str, stage: str) -> Dict[str, ProbeResult]:
    if not log_path.exists():
        raise FileNotFoundError(f"missing log: {log_path}")

    results: Dict[str, ProbeResult] = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = PROBE_RE.search(line)
            if match is None:
                continue
            probe = match.group("probe")
            results[probe] = ProbeResult(
                experiment=experiment,
                stage=stage,
                probe=probe,
                iteration=int(match.group("iteration")),
                local_iteration=int(match.group("local_iteration")),
                next_token_acc=float(match.group("acc")),
                ppl=float(match.group("ppl")),
                log_path=log_path,
            )
    missing = sorted(set(PROBE_LABEL) - set(results))
    if missing:
        raise ValueError(f"{log_path} is missing final probe(s): {', '.join(missing)}")
    return results


def collect_results(g2_root: Path) -> List[ProbeResult]:
    rows: List[ProbeResult] = []
    for spec in RUNS:
        log_path = g2_root / spec.run_id / spec.log_name
        results = parse_last_probe_results(log_path, spec.experiment, spec.stage)
        rows.extend(results[probe] for probe in ("code_probe", "wiki_probe"))
    return rows


def write_csv(rows: Iterable[ProbeResult], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "experiment",
                "stage",
                "probe",
                "iteration",
                "local_iteration",
                "next_token_acc",
                "ppl",
                "log_path",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.experiment,
                    row.stage,
                    row.probe,
                    row.iteration,
                    row.local_iteration,
                    f"{row.next_token_acc:.6f}",
                    f"{row.ppl:.6f}",
                    row.log_path,
                ]
            )


def write_markdown(rows: List[ProbeResult], md_path: Path) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# G2 Phase 3 Log-Based Probe Summary",
        "",
        "| Experiment | Stage | Probe | Iteration | Accuracy | PPL |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.experiment} | "
            f"{STAGE_LABEL[row.stage]} | "
            f"{PROBE_LABEL[row.probe]} | "
            f"{row.iteration} | "
            f"{row.next_token_acc:.6f} | "
            f"{row.ppl:.3f} |"
        )

    lines.extend(["", "## Delta: After - Before", ""])
    lines.append("| Experiment | Probe | Accuracy Delta | PPL Delta |")
    lines.append("| --- | --- | ---: | ---: |")

    keyed = {(r.experiment, r.stage, r.probe): r for r in rows}
    for experiment in ("exp1", "exp2"):
        for probe in ("code_probe", "wiki_probe"):
            before = keyed[(experiment, "phase1_before_router_retune", probe)]
            after = keyed[(experiment, "phase3_after_router_retune", probe)]
            lines.append(
                "| "
                f"{experiment} | "
                f"{PROBE_LABEL[probe]} | "
                f"{after.next_token_acc - before.next_token_acc:+.6f} | "
                f"{after.ppl - before.ppl:+.3f} |"
            )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2.0
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def write_metric_svg(rows: List[ProbeResult], metric: str, svg_path: Path) -> None:
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1120, 620
    left, right, top, bottom = 84, 34, 72, 96
    plot_left, plot_right = left, width - right
    plot_top, plot_bottom = top, height - bottom
    plot_height = plot_bottom - plot_top

    values = [getattr(row, metric) for row in rows]
    y_min = 0.0 if metric == "next_token_acc" else min(values)
    y_max = max(values)
    if metric == "ppl":
        padding = (y_max - y_min) * 0.12
        y_min = max(0.0, y_min - padding)
        y_max = y_max + padding
    else:
        y_max = min(1.0, max(0.01, y_max + 0.03))

    clusters: List[Tuple[str, str]] = [
        ("exp1", "code_probe"),
        ("exp1", "wiki_probe"),
        ("exp2", "code_probe"),
        ("exp2", "wiki_probe"),
    ]
    keyed = {(r.experiment, r.stage, r.probe): r for r in rows}
    cluster_width = (plot_right - plot_left) / len(clusters)
    bar_width = 56
    gap = 16
    axis = []

    for i in range(6):
        y_value = y_min + (y_max - y_min) * i / 5.0
        y = scale(y_value, y_min, y_max, plot_bottom, plot_top)
        axis.append(
            f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" '
            'stroke="#e8e2d6" stroke-width="1"/>'
        )
        axis.append(
            f'<text x="{plot_left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            'font-family="sans-serif" font-size="12" fill="#4a4238">'
            f"{y_value:.3g}</text>"
        )

    bars = []
    for idx, (experiment, probe) in enumerate(clusters):
        center = plot_left + cluster_width * (idx + 0.5)
        group_start = center - bar_width - gap / 2
        for stage_idx, stage in enumerate(
            ("phase1_before_router_retune", "phase3_after_router_retune")
        ):
            row = keyed[(experiment, stage, probe)]
            value = getattr(row, metric)
            x = group_start + stage_idx * (bar_width + gap)
            y = scale(value, y_min, y_max, plot_bottom, plot_top)
            h = max(1.0, plot_bottom - y)
            color = BAR_COLORS[stage]
            bars.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width}" height="{h:.2f}" '
                f'rx="6" fill="{color}"/>'
            )
            bars.append(
                f'<text x="{x + bar_width / 2:.2f}" y="{y - 7:.2f}" text-anchor="middle" '
                'font-family="sans-serif" font-size="12" fill="#211a14">'
                f"{value:.3f}</text>"
            )
        bars.append(
            f'<text x="{center:.2f}" y="{plot_bottom + 28}" text-anchor="middle" '
            'font-family="sans-serif" font-size="13" font-weight="700" fill="#211a14">'
            f"{experiment.upper()}</text>"
        )
        bars.append(
            f'<text x="{center:.2f}" y="{plot_bottom + 48}" text-anchor="middle" '
            'font-family="sans-serif" font-size="13" fill="#4a4238">'
            f"{PROBE_LABEL[probe]}</text>"
        )

    legend = []
    legend_x = width - 360
    legend_y = 28
    for offset, stage in enumerate(("phase1_before_router_retune", "phase3_after_router_retune")):
        x = legend_x + offset * 170
        legend.append(
            f'<rect x="{x}" y="{legend_y}" width="18" height="18" rx="4" '
            f'fill="{BAR_COLORS[stage]}"/>'
        )
        legend.append(
            f'<text x="{x + 26}" y="{legend_y + 14}" font-family="sans-serif" '
            f'font-size="13" fill="#211a14">{html.escape(STAGE_LABEL[stage])}</text>'
        )

    title = f"G2 Phase1 vs Phase3 Log Probe - {METRIC_LABEL[metric]}"
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbf7ef"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" '
        'font-family="sans-serif" font-size="22" font-weight="700" fill="#211a14">'
        f"{html.escape(title)}</text>",
        *legend,
        *axis,
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" '
        'stroke="#4a4238" stroke-width="1.5"/>',
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" '
        'stroke="#4a4238" stroke-width="1.5"/>',
        *bars,
        f'<text x="24" y="{(plot_top + plot_bottom) / 2:.2f}" text-anchor="middle" '
        'font-family="sans-serif" font-size="14" fill="#211a14" '
        f'transform="rotate(-90 24 {(plot_top + plot_bottom) / 2:.2f})">'
        f"{html.escape(METRIC_LABEL[metric])}</text>",
        "</svg>",
    ]
    svg_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    args = parse_args()
    g2_root = Path(args.g2_root)
    out_dir = Path(args.out_dir)
    rows = collect_results(g2_root)

    csv_path = out_dir / "g2_phase3_probe_summary.csv"
    md_path = out_dir / "g2_phase3_probe_summary.md"
    acc_svg = out_dir / "g2_phase3_probe_accuracy_bars.svg"
    ppl_svg = out_dir / "g2_phase3_probe_ppl_bars.svg"

    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    write_metric_svg(rows, "next_token_acc", acc_svg)
    write_metric_svg(rows, "ppl", ppl_svg)

    print(f"rows: {len(rows)}")
    print(f"csv: {csv_path}")
    print(f"summary: {md_path}")
    print(f"accuracy_svg: {acc_svg}")
    print(f"ppl_svg: {ppl_svg}")


if __name__ == "__main__":
    main()
