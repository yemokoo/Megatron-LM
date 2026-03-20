#!/usr/bin/env python3
import argparse
import csv
import json
import re
from html import escape
from pathlib import Path


PROBE_RE = re.compile(
    r"probe\s+(?P<name>\S+)\s+at iteration\s+(?P<iteration>\d+)\s+\|\s+"
    r"local_iteration:\s+(?P<local_iteration>\d+)\s+\|\s+"
    r"next_token_acc:\s+(?P<accuracy>[0-9.]+)\s+\|\s+ppl:\s+(?P<ppl>[0-9.E+-]+)"
)

SUITE_VARIANTS = {
    "a_to_b": (
        ("full", "Shared Unfreeze", "#1f77b4"),
        ("new_only", "Shared Freeze", "#cc6b00"),
    ),
    "b_to_a": (
        ("full", "Shared Unfreeze", "#1f77b4"),
        ("new_only", "Shared Freeze", "#cc6b00"),
    ),
}

PROBE_LABELS = {
    "task_a_probe": "wiki",
    "task_b_probe": "code",
}

METRIC_LABELS = {
    "accuracy": "Next-token accuracy",
    "ppl": "Perplexity",
}


def parse_probe_log(path: Path, variant: str, suite: str):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = PROBE_RE.search(line)
            if match is None:
                continue
            local_step = int(match.group("local_iteration"))
            if local_step > 1800:
                continue
            rows.append(
                {
                    "suite": suite,
                    "variant": variant,
                    "probe": match.group("name"),
                    "local_step": local_step,
                    "accuracy": float(match.group("accuracy")),
                    "ppl": float(match.group("ppl")),
                    "source": f"log:{path}",
                }
            )
    return rows


def parse_schedule_csv(path: Path, variant: str, suite: str, probe: str):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "suite": suite,
                    "variant": variant,
                    "probe": probe,
                    "local_step": int(float(row["local_step"])),
                    "accuracy": float(row["accuracy"]),
                    "ppl": float(row["ppl"]),
                    "source": f"schedule:{path}",
                }
            )
    return rows


def dedupe_rows(rows):
    keyed = {}
    for row in rows:
        key = (row["suite"], row["variant"], row["probe"], row["local_step"])
        keyed[key] = row
    return sorted(keyed.values(), key=lambda row: (row["suite"], row["variant"], row["probe"], row["local_step"]))


def build_series(rows, suite: str, probe: str, metric: str):
    series = []
    for variant, label, color in SUITE_VARIANTS[suite]:
        variant_rows = [row for row in rows if row["suite"] == suite and row["variant"] == variant and row["probe"] == probe]
        if not variant_rows:
            continue
        xs = [row["local_step"] for row in variant_rows]
        ys = [row[metric] for row in variant_rows]
        series.append((label, color, xs, ys))
    return series


def svg_polyline(points):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def scale_value(value, src_min, src_max, dst_min, dst_max):
    if src_max == src_min:
        return (dst_min + dst_max) / 2.0
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def write_svg(rows, suite: str, probe: str, metric: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = build_series(rows, suite, probe, metric)

    width = 900
    height = 500
    margin_left = 80
    margin_right = 30
    margin_top = 55
    margin_bottom = 60
    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    title = f"{suite.replace('_', ' ').upper()} / {PROBE_LABELS[probe]} / {METRIC_LABELS[metric]}"
    x_label = "Local continual-learning step"
    y_label = METRIC_LABELS[metric]

    all_xs = [x for _, _, xs, _ in series for x in xs]
    all_ys = [y for _, _, _, ys in series for y in ys]
    if all_xs:
        x_min = min(all_xs)
        x_max = max(all_xs)
    else:
        x_min, x_max = 0, 1800
    if all_ys:
        y_min = min(all_ys)
        y_max = max(all_ys)
    else:
        y_min, y_max = 0.0, 1.0
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    y_pad = (y_max - y_min) * 0.08
    y_min -= y_pad
    y_max += y_pad

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-size="20" font-family="sans-serif">{escape(title)}</text>',
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#333" stroke-width="1.5"/>',
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#333" stroke-width="1.5"/>',
    ]

    for i in range(5):
        y_val = y_min + (y_max - y_min) * (i / 4.0)
        y = scale_value(y_val, y_min, y_max, plot_bottom, plot_top)
        svg.append(
            f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" stroke="#dddddd" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{plot_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="sans-serif" fill="#444">{y_val:.3g}</text>'
        )

    for x_val in [0, 300, 600, 900, 1200, 1500, 1800]:
        if x_val < x_min or x_val > x_max:
            continue
        x = scale_value(x_val, x_min, x_max, plot_left, plot_right)
        svg.append(
            f'<line x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" y2="{plot_bottom}" stroke="#eeeeee" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{x:.2f}" y="{plot_bottom + 22}" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#444">{x_val}</text>'
        )

    for label, color, xs, ys in series:
        points = [
            (
                scale_value(x, x_min, x_max, plot_left, plot_right),
                scale_value(y, y_min, y_max, plot_bottom, plot_top),
            )
            for x, y in zip(xs, ys)
        ]
        svg.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{svg_polyline(points)}"/>'
        )
        for x, y in points:
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>')

    legend_x = plot_right - 180
    legend_y = plot_top + 10
    for idx, (label, color, _, _) in enumerate(series):
        y = legend_y + idx * 22
        svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<circle cx="{legend_x + 12}" cy="{y}" r="3" fill="{color}"/>')
        svg.append(
            f'<text x="{legend_x + 32}" y="{y + 4}" font-size="12" font-family="sans-serif" fill="#222">{escape(label)}</text>'
        )

    svg.append(
        f'<text x="{width / 2:.1f}" y="{height - 15}" text-anchor="middle" font-size="14" font-family="sans-serif">{escape(x_label)}</text>'
    )
    svg.append(
        f'<text x="20" y="{height / 2:.1f}" text-anchor="middle" font-size="14" font-family="sans-serif" transform="rotate(-90 20 {height / 2:.1f})">{escape(y_label)}</text>'
    )
    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Plot A->B / B->A probe comparison graphs.")
    parser.add_argument("--a-to-b-full-log", required=True)
    parser.add_argument("--a-to-b-full-pre-log", default=None)
    parser.add_argument("--a-to-b-new-only-log", required=True)
    parser.add_argument("--b-to-a-full-log", required=True)
    parser.add_argument("--b-to-a-new-only-log", required=True)
    parser.add_argument("--a-to-b-full-code-schedule", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.a_to_b_full_pre_log:
        rows.extend(parse_probe_log(Path(args.a_to_b_full_pre_log), "full", "a_to_b"))
    rows.extend(parse_probe_log(Path(args.a_to_b_full_log), "full", "a_to_b"))
    rows.extend(parse_probe_log(Path(args.a_to_b_new_only_log), "new_only", "a_to_b"))
    rows.extend(parse_probe_log(Path(args.b_to_a_full_log), "full", "b_to_a"))
    rows.extend(parse_probe_log(Path(args.b_to_a_new_only_log), "new_only", "b_to_a"))

    # Replace the missing A->B full code probe with the supplemental checkpoint+midpoint eval series.
    rows = [
        row
        for row in rows
        if not (row["suite"] == "a_to_b" and row["variant"] == "full" and row["probe"] == "task_b_probe")
    ]
    rows.extend(parse_schedule_csv(Path(args.a_to_b_full_code_schedule), "full", "a_to_b", "task_b_probe"))
    rows = dedupe_rows(rows)

    with (output_dir / "combined_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    with (output_dir / "combined_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["suite", "variant", "probe", "local_step", "accuracy", "ppl", "source"],
        )
        writer.writeheader()
        writer.writerows(rows)

    specs = [
        ("a_to_b", "task_a_probe", "ppl"),
        ("a_to_b", "task_a_probe", "accuracy"),
        ("a_to_b", "task_b_probe", "ppl"),
        ("a_to_b", "task_b_probe", "accuracy"),
        ("b_to_a", "task_a_probe", "ppl"),
        ("b_to_a", "task_a_probe", "accuracy"),
        ("b_to_a", "task_b_probe", "ppl"),
        ("b_to_a", "task_b_probe", "accuracy"),
    ]
    for suite, probe, metric in specs:
        write_svg(rows, suite, probe, metric, output_dir / f"{suite}_{PROBE_LABELS[probe]}_{metric}.svg")


if __name__ == "__main__":
    main()
