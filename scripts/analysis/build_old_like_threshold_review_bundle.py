#!/usr/bin/env python3
"""Build human-review plots/tables for choosing an old-like threshold.

This reads only completed histogram summaries. It does not assign GT.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCORES = (
    "percentile_mean_l2_l9",
    "percentile_median_l2_l9",
    "percentile_min_l2_l9",
    "late_percentile_mean_l7_l9",
    "raw_cosine_mean_l2_l9",
)


def font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".inprogress")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".inprogress")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def plot_panels(
    panels: list[tuple[str, np.ndarray, np.ndarray, float, float]],
    path: Path,
    *,
    columns: int,
    log_y: bool,
) -> None:
    rows = math.ceil(len(panels) / columns)
    cell_w, cell_h = 720, 430
    image = Image.new("RGB", (columns * cell_w + 80, rows * cell_h + 130), "white")
    draw = ImageDraw.Draw(image)
    title = "Code vs Wiki score histograms (100 display bins; CSV remains exact)"
    if log_y:
        title += " (log10 probability)"
    draw.text((image.width // 2, 20), title, fill="#111111", font=font(29), anchor="ma")
    for panel_index, (name, code, wiki, x_min, x_max) in enumerate(panels):
        row, column = divmod(panel_index, columns)
        left, top = 80 + column * cell_w, 80 + row * cell_h
        right, bottom = left + 610, top + 330
        draw.rectangle((left, top, right, bottom), outline="#333333", width=2)
        draw.text(((left + right) // 2, top - 27), name, fill="#111111", font=font(18), anchor="ma")
        display_bins = min(100, code.size)
        if code.size % display_bins:
            raise RuntimeError(f"cannot evenly display-bin histogram of size {code.size}")
        code_coarse = code.reshape(display_bins, -1).sum(axis=1, dtype=np.uint64)
        wiki_coarse = wiki.reshape(display_bins, -1).sum(axis=1, dtype=np.uint64)
        code_prob = code_coarse.astype(np.float64) / code.sum(dtype=np.uint64)
        wiki_prob = wiki_coarse.astype(np.float64) / wiki.sum(dtype=np.uint64)
        plot_width = right - left
        values_all = []
        for values in (code_prob, wiki_prob):
            xpixel = np.minimum((np.arange(values.size) * plot_width) // values.size, plot_width - 1)
            reduced = np.zeros(plot_width, dtype=np.float64)
            np.maximum.at(reduced, xpixel, values)
            values_all.append(reduced)
        if log_y:
            lower_log = -10.0
            transformed = [np.maximum(value, 10**lower_log) for value in values_all]
            ys = [(np.log10(value) - lower_log) / -lower_log for value in transformed]
        else:
            maximum = max(float(value.max()) for value in values_all)
            ys = [value / max(maximum, 1e-30) for value in values_all]
        for values, color in zip(ys, ("#1768ac", "#d62728")):
            points = [
                (left + index, bottom - int(round(float(value) * (bottom - top))))
                for index, value in enumerate(values)
            ]
            draw.line(points, fill=color, width=2)
        for fraction in (0.0, 0.5, 1.0):
            x = left + int(round(fraction * (right - left)))
            value = x_min + fraction * (x_max - x_min)
            draw.line((x, bottom, x, bottom + 5), fill="#333333")
            draw.text((x, bottom + 8), f"{value:.3g}", fill="#333333", font=font(14), anchor="ma")
    draw.line((image.width - 390, image.height - 45, image.width - 340, image.height - 45), fill="#1768ac", width=4)
    draw.text((image.width - 330, image.height - 45), "Code", fill="#222222", font=font(16), anchor="lm")
    draw.line((image.width - 220, image.height - 45, image.width - 170, image.height - 45), fill="#d62728", width=4)
    draw.text((image.width - 160, image.height - 45), "Wiki", fill="#222222", font=font(16), anchor="lm")
    image.save(path, format="PNG", optimize=True)


def crop_hist(counts: np.ndarray, full_min: float, full_max: float, view_min: float, view_max: float) -> np.ndarray:
    bins = counts.size
    start = max(0, int(math.floor((view_min - full_min) / (full_max - full_min) * bins)))
    end = min(bins, int(math.ceil((view_max - full_min) / (full_max - full_min) * bins)))
    return counts[start:end]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-numeric", required=True, type=Path)
    parser.add_argument("--wiki-numeric", required=True, type=Path)
    parser.add_argument("--code-cosine", required=True, type=Path)
    parser.add_argument("--wiki-cosine", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.code_numeric, allow_pickle=False) as data:
        code_aggregate = data["aggregate_hist"].copy()
        code_patterns = data["pattern_counts"].copy()
    with np.load(args.wiki_numeric, allow_pickle=False) as data:
        wiki_aggregate = data["aggregate_hist"].copy()
        wiki_patterns = data["pattern_counts"].copy()
    with np.load(args.code_cosine, allow_pickle=False) as data:
        code_cosine = data["counts"].copy()
    with np.load(args.wiki_cosine, allow_pickle=False) as data:
        wiki_cosine = data["counts"].copy()

    global_panels = []
    global_zoom_panels = []
    for index, name in enumerate(SCORES[:4]):
        global_panels.append((name, code_aggregate[index], wiki_aggregate[index], 0.0, 1.0))
        global_zoom_panels.append(
            (name, crop_hist(code_aggregate[index], 0, 1, 0.8, 1), crop_hist(wiki_aggregate[index], 0, 1, 0.8, 1), 0.8, 1.0)
        )
    plot_panels(global_panels, output / "global_score_hist_log.png", columns=2, log_y=True)
    plot_panels(global_zoom_panels, output / "global_score_hist_zoom_0p8_1_log.png", columns=2, log_y=True)

    raw_panels = []
    raw_zoom_panels = []
    for layer in range(2, 10):
        raw_panels.append((f"Layer {layer} raw cosine", code_cosine[layer - 1], wiki_cosine[layer - 1], -1.0, 1.0))
        raw_zoom_panels.append(
            (
                f"Layer {layer} raw cosine",
                crop_hist(code_cosine[layer - 1], -1, 1, 0.9, 1),
                crop_hist(wiki_cosine[layer - 1], -1, 1, 0.9, 1),
                0.9,
                1.0,
            )
        )
    plot_panels(raw_panels, output / "layer_raw_cosine_hist_log.png", columns=2, log_y=True)
    plot_panels(raw_zoom_panels, output / "layer_raw_cosine_hist_zoom_0p9_1_log.png", columns=2, log_y=True)

    targets = (0.20, 0.10, 0.05, 0.02, 0.01, 0.005, 0.001)
    rows: list[dict] = []
    for score_index, name in enumerate(SCORES):
        code = code_aggregate[score_index]
        wiki = wiki_aggregate[score_index]
        lower, upper = ((0.0, 1.0) if score_index < 4 else (-1.0, 1.0))
        code_tail = np.cumsum(code[::-1], dtype=np.uint64)[::-1] / code.sum(dtype=np.uint64)
        wiki_tail = np.cumsum(wiki[::-1], dtype=np.uint64)[::-1] / wiki.sum(dtype=np.uint64)
        edges = np.linspace(lower, upper, code.size + 1)
        for target in targets:
            index = int(np.argmin(np.abs(code_tail - target)))
            rows.append(
                {
                    "score": name,
                    "target_code_coverage": target,
                    "threshold_ge": float(edges[index]),
                    "actual_code_coverage": float(code_tail[index]),
                    "wiki_recall": float(wiki_tail[index]),
                    "balanced_prior_precision": float(wiki_tail[index] / max(wiki_tail[index] + code_tail[index], 1e-30)),
                    "selected_as_gt": False,
                }
            )
    write_csv(output / "compact_threshold_review.csv", rows)

    patterns = np.arange(256, dtype=np.uint16)
    bit_count = np.asarray([int(value).bit_count() for value in patterns])
    stable_rows = []
    for top_index, top_percent in enumerate((1, 5, 10, 20)):
        for minimum in (2, 4, 6, 8):
            code_rate = float(code_patterns[top_index, bit_count >= minimum].sum() / code_patterns[top_index].sum())
            wiki_rate = float(wiki_patterns[top_index, bit_count >= minimum].sum() / wiki_patterns[top_index].sum())
            stable_rows.append(
                {
                    "layerwise_top_percent": top_percent,
                    "minimum_stable_layers": minimum,
                    "code_coverage": code_rate,
                    "wiki_recall": wiki_rate,
                    "wiki_over_code_enrichment": wiki_rate / max(code_rate, 1e-30),
                    "selected_as_gt": False,
                }
            )
    write_csv(output / "stable_layer_count_review.csv", stable_rows)
    atomic_json(
        output / "metadata.json",
        {
            "schema": "old_like_threshold_review_bundle_v1",
            "complete": True,
            "gt_assigned": False,
            "layers": list(range(2, 10)),
            "combination_note": "All global scores combine per-layer scalar metrics; no hidden vectors were averaged or concatenated.",
            "code_tokens": int(code_aggregate[0].sum()),
            "wiki_tokens": int(wiki_aggregate[0].sum()),
        },
    )
    print(json.dumps({"complete": True, "output": str(output), "gt_assigned": False}, indent=2))


if __name__ == "__main__":
    main()
