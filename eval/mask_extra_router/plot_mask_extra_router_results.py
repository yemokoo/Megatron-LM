#!/usr/bin/env python3
import argparse
import html
import json
from pathlib import Path

LEGEND = [
    ("#4e79a7", "Task1 only"),
    ("#59a14f", "Continual"),
    ("#e15759", "Continual + masked"),
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_overview_svg(cases, output_path: Path):
    width, height = 1200, 620
    left, right, top, bottom = 90, 40, 70, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_acc = max(
        max(case["baseline"]["next_token_acc"], case["continual"]["next_token_acc"], case["masked"]["next_token_acc"])
        for case in cases
    )
    y_max = max(1.0, max_acc * 1.15)

    def map_y(v):
        return top + plot_h - (v / y_max) * plot_h

    grid = []
    for i in range(6):
        value = y_max * i / 5
        y = map_y(value)
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#dddddd" stroke-width="1"/>'
        )
        grid.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#444">{value:.3f}</text>'
        )

    bars = []
    labels = []
    group_w = plot_w / max(len(cases), 1)
    bar_w = group_w * 0.18
    for i, case in enumerate(cases):
        group_center = left + group_w * (i + 0.5)
        xs = [group_center - bar_w * 1.2, group_center, group_center + bar_w * 1.2]
        vals = [
            case["baseline"]["next_token_acc"],
            case["continual"]["next_token_acc"],
            case["masked"]["next_token_acc"],
        ]
        for x, v, (color, _) in zip(xs, vals, LEGEND):
            y = map_y(v)
            bars.append(
                f'<rect x="{x - bar_w / 2:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{top + plot_h - y:.2f}" fill="{color}"/>'
            )
            bars.append(
                f'<text x="{x:.2f}" y="{y - 8:.2f}" text-anchor="middle" font-size="11" fill="#222">{v:.4f}</text>'
            )
        labels.append(
            f'<text x="{group_center:.2f}" y="{top + plot_h + 28:.2f}" text-anchor="middle" font-size="12" fill="#222">{html.escape(case["name"])}</text>'
        )

    legend_items = []
    legend_x = left
    for idx, (color, text) in enumerate(LEGEND):
        x = legend_x + idx * 220
        legend_items.append(f'<rect x="{x}" y="24" width="18" height="18" fill="{color}"/>')
        legend_items.append(f'<text x="{x + 26}" y="38" font-size="14" fill="#222">{html.escape(text)}</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2:.0f}" y="52" text-anchor="middle" font-size="24" font-family="sans-serif" fill="#111">Task1 Accuracy After Masking Task2 Experts</text>
{''.join(legend_items)}
{''.join(grid)}
<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#888" stroke-width="1.2"/>
{''.join(bars)}
{''.join(labels)}
<text x="{width/2:.0f}" y="{height - 24}" text-anchor="middle" font-size="14" font-family="sans-serif" fill="#222">Experiment</text>
<text x="26" y="{height/2:.0f}" text-anchor="middle" font-size="14" font-family="sans-serif" fill="#222" transform="rotate(-90 26 {height/2:.0f})">Next-token accuracy</text>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def make_case_svg(case, output_path: Path):
    width, height = 760, 560
    left, right, top, bottom = 90, 40, 70, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [
        case["baseline"]["next_token_acc"],
        case["continual"]["next_token_acc"],
        case["masked"]["next_token_acc"],
    ]
    labels = ["Task1 only", "Continual", "Continual + masked"]
    y_max = max(1.0, max(values) * 1.15)

    def map_y(v):
        return top + plot_h - (v / y_max) * plot_h

    grid = []
    for i in range(6):
        value = y_max * i / 5
        y = map_y(value)
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#dddddd" stroke-width="1"/>'
        )
        grid.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#444">{value:.3f}</text>'
        )

    group_center = left + plot_w / 2
    spacing = plot_w / 4
    xs = [group_center - spacing, group_center, group_center + spacing]
    bar_w = plot_w * 0.12

    bars = []
    xlabels = []
    for x, value, label, (color, _) in zip(xs, values, labels, LEGEND):
        y = map_y(value)
        bars.append(
            f'<rect x="{x - bar_w / 2:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{top + plot_h - y:.2f}" fill="{color}"/>'
        )
        bars.append(
            f'<text x="{x:.2f}" y="{y - 8:.2f}" text-anchor="middle" font-size="12" fill="#222">{value:.4f}</text>'
        )
        xlabels.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 28:.2f}" text-anchor="middle" font-size="12" fill="#222">{html.escape(label)}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2:.0f}" y="42" text-anchor="middle" font-size="22" font-family="sans-serif" fill="#111">{html.escape(case["name"])}</text>
{''.join(grid)}
<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#888" stroke-width="1.2"/>
{''.join(bars)}
{''.join(xlabels)}
<text x="{width/2:.0f}" y="{height - 24}" text-anchor="middle" font-size="14" font-family="sans-serif" fill="#222">Setting</text>
<text x="26" y="{height/2:.0f}" text-anchor="middle" font-size="14" font-family="sans-serif" fill="#222" transform="rotate(-90 26 {height/2:.0f})">Next-token accuracy</text>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-a", required=True)
    parser.add_argument("--baseline-b", required=True)
    parser.add_argument("--a-to-b", required=True)
    parser.add_argument("--a-to-b-masked", required=True)
    parser.add_argument("--a-to-b-new-only", required=True)
    parser.add_argument("--a-to-b-new-only-masked", required=True)
    parser.add_argument("--b-to-a", required=True)
    parser.add_argument("--b-to-a-masked", required=True)
    parser.add_argument("--b-to-a-new-only", required=True)
    parser.add_argument("--b-to-a-new-only-masked", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-svg", required=True)
    args = parser.parse_args()

    baseline_a = load_json(Path(args.baseline_a))
    baseline_b = load_json(Path(args.baseline_b))
    cases = [
        {
            "name": "A -> B (Task1=A)",
            "baseline": baseline_a,
            "continual": load_json(Path(args.a_to_b)),
            "masked": load_json(Path(args.a_to_b_masked)),
        },
        {
            "name": "A -> B new-only (Task1=A)",
            "baseline": baseline_a,
            "continual": load_json(Path(args.a_to_b_new_only)),
            "masked": load_json(Path(args.a_to_b_new_only_masked)),
        },
        {
            "name": "B -> A (Task1=B)",
            "baseline": baseline_b,
            "continual": load_json(Path(args.b_to_a)),
            "masked": load_json(Path(args.b_to_a_masked)),
        },
        {
            "name": "B -> A new-only (Task1=B)",
            "baseline": baseline_b,
            "continual": load_json(Path(args.b_to_a_new_only)),
            "masked": load_json(Path(args.b_to_a_new_only_masked)),
        },
    ]

    summary = {
        "cases": cases,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_svg = Path(args.output_svg)
    make_overview_svg(cases, output_svg)

    case_outputs = {
        "a_to_b": output_svg.with_name("task1_accuracy_a_to_b.svg"),
        "a_to_b_new_only": output_svg.with_name("task1_accuracy_a_to_b_new_only.svg"),
        "b_to_a": output_svg.with_name("task1_accuracy_b_to_a.svg"),
        "b_to_a_new_only": output_svg.with_name("task1_accuracy_b_to_a_new_only.svg"),
    }
    make_case_svg(cases[0], case_outputs["a_to_b"])
    make_case_svg(cases[1], case_outputs["a_to_b_new_only"])
    make_case_svg(cases[2], case_outputs["b_to_a"])
    make_case_svg(cases[3], case_outputs["b_to_a_new_only"])

    summary["case_images"] = {key: str(path) for key, path in case_outputs.items()}
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
