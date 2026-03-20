#!/usr/bin/env python3
import argparse
import csv
import html
import re
from pathlib import Path


LINE_RE = re.compile(
    r"iteration\s+(\d+)\s*/\s*(\d+).*?\|\s+lm loss:\s*([0-9.+\-Ee]+)"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract lm loss points from a training log and plot them."
    )
    parser.add_argument("--log", required=True, help="Path to training log file")
    parser.add_argument("--svg", required=True, help="Output SVG path")
    parser.add_argument("--csv", required=True, help="Output CSV path")
    parser.add_argument("--title", default="LM Loss from Log", help="Plot title")
    return parser.parse_args()


def extract_points(log_path: Path):
    points = []
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = LINE_RE.search(line)
            if not match:
                continue
            iteration = int(match.group(1))
            total_iterations = int(match.group(2))
            lm_loss = float(match.group(3))
            points.append((iteration, total_iterations, lm_loss))
    return points


def write_csv(points, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "total_iterations", "lm_loss"])
        writer.writerows(points)


def write_svg(points, svg_path: Path, title: str):
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    iterations = [p[0] for p in points]
    losses = [p[2] for p in points]
    width, height = 1000, 560
    left, right, top, bottom = 80, 30, 50, 60
    plot_w = width - left - right
    plot_h = height - top - bottom

    min_x, max_x = min(iterations), max(iterations)
    min_y, max_y = min(losses), max(losses)
    if max_x == min_x:
        max_x += 1
    if max_y == min_y:
        max_y += 1

    def map_x(x):
        return left + (x - min_x) / (max_x - min_x) * plot_w

    def map_y(y):
        return top + plot_h - (y - min_y) / (max_y - min_y) * plot_h

    polyline = " ".join(f"{map_x(x):.2f},{map_y(y):.2f}" for x, y in zip(iterations, losses))

    grid_lines = []
    for i in range(6):
        y = top + i * plot_h / 5
        value = max_y - i * (max_y - min_y) / 5
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" '
            f'stroke="#d9d9d9" stroke-width="1"/>'
        )
        grid_lines.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="12" fill="#444">{value:.3f}</text>'
        )

    x_ticks = []
    for i in range(6):
        x = left + i * plot_w / 5
        value = round(min_x + i * (max_x - min_x) / 5)
        x_ticks.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" '
            f'stroke="#eeeeee" stroke-width="1"/>'
        )
        x_ticks.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle" '
            f'font-size="12" fill="#444">{value}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2:.0f}" y="28" text-anchor="middle" font-size="22" font-family="sans-serif" fill="#111">{html.escape(title)}</text>
{''.join(grid_lines)}
{''.join(x_ticks)}
<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#888" stroke-width="1.2"/>
<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{polyline}"/>
<text x="{width/2:.0f}" y="{height - 18}" text-anchor="middle" font-size="14" font-family="sans-serif" fill="#222">Iteration</text>
<text x="22" y="{height/2:.0f}" text-anchor="middle" font-size="14" font-family="sans-serif" fill="#222" transform="rotate(-90 22 {height/2:.0f})">LM Loss</text>
</svg>
"""
    svg_path.write_text(svg, encoding="utf-8")


def main():
    args = parse_args()
    log_path = Path(args.log)
    svg_path = Path(args.svg)
    csv_path = Path(args.csv)

    points = extract_points(log_path)
    if not points:
        raise SystemExit(f"No lm loss points found in {log_path}")

    write_csv(points, csv_path)
    write_svg(points, svg_path, args.title)
    print(
        f"Parsed {len(points)} points from {log_path}\n"
        f"CSV: {csv_path}\n"
        f"SVG: {svg_path}"
    )


if __name__ == "__main__":
    main()
