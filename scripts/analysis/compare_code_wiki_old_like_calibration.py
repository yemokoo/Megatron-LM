#!/usr/bin/env python3
"""Compare Wiki positive-reference and Code stability distributions.

Both cross-layer summaries must have been computed with the *Code* cosine CDF.
The script reports binned AUROC and threshold curves without assigning GT.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np


AGGREGATE_NAMES = (
    "percentile_mean",
    "percentile_median",
    "percentile_min",
    "late_l7_l9_percentile_mean",
    "raw_cosine_mean",
)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".inprogress")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    temporary = path.with_name(path.name + ".inprogress")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _auc(pos: np.ndarray, neg: np.ndarray, higher_is_positive: bool = True) -> float:
    pos = pos.astype(np.float64, copy=False)
    neg = neg.astype(np.float64, copy=False)
    if not higher_is_positive:
        pos = pos[::-1]
        neg = neg[::-1]
    neg_less = np.cumsum(neg) - neg
    numerator = np.sum(pos * (neg_less + 0.5 * neg), dtype=np.float64)
    denominator = pos.sum(dtype=np.float64) * neg.sum(dtype=np.float64)
    return float(numerator / denominator)


def _threshold_rows(name: str, pos: np.ndarray, neg: np.ndarray, lower: float, upper: float) -> list[dict]:
    pos = pos.astype(np.float64, copy=False)
    neg = neg.astype(np.float64, copy=False)
    pos_tail = np.cumsum(pos[::-1])[::-1]
    neg_tail = np.cumsum(neg[::-1])[::-1]
    tpr = pos_tail / pos.sum()
    fpr = neg_tail / neg.sum()
    empirical_precision = pos_tail / np.maximum(pos_tail + neg_tail, 1.0)
    balanced_precision = tpr / np.maximum(tpr + fpr, 1e-30)
    edges = np.linspace(lower, upper, pos.size + 1)
    # At most 201 rows per score, while retaining both endpoints.
    indexes = np.unique(np.linspace(0, pos.size - 1, min(pos.size, 201)).round().astype(int))
    return [
        {
            "score": name,
            "threshold_ge": float(edges[index]),
            "wiki_recall_tpr": float(tpr[index]),
            "code_fpr": float(fpr[index]),
            "balanced_prior_precision": float(balanced_precision[index]),
            "observed_sample_mixture_precision": float(empirical_precision[index]),
        }
        for index in indexes
    ]


def _pattern_to_stable_counts(pattern_counts: np.ndarray) -> np.ndarray:
    patterns = np.arange(256, dtype=np.uint16)
    bit_count = np.asarray([int(value).bit_count() for value in patterns], dtype=np.int16)
    return np.asarray(
        [pattern_counts[bit_count == count].sum(dtype=np.uint64) for count in range(9)],
        dtype=np.uint64,
    )


def _render_distributions(rows: list[tuple[str, np.ndarray, np.ndarray]], path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    def font(size: int):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()

    image = Image.new("RGB", (1600, 1100), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (800, 22), "Code vs Wiki stability scores (Code-CDF calibrated)",
        fill="#111111", font=font(30), anchor="ma",
    )
    title_font, tick_font = font(21), font(16)
    for panel, (name, code, wiki) in enumerate(rows):
        row, column = divmod(panel, 2)
        left = 90 + column * 760
        top = 90 + row * 490
        right, bottom = left + 650, top + 390
        draw.rectangle((left, top, right, bottom), outline="#333333", width=2)
        draw.text(((left + right) // 2, top - 32), name, fill="#111111", font=title_font, anchor="ma")
        code_prob = code.astype(np.float64) / code.sum(dtype=np.uint64)
        wiki_prob = wiki.astype(np.float64) / wiki.sum(dtype=np.uint64)
        # Reduce 1000 bins to plot pixels using max probability per pixel.
        width = right - left
        xpixel = np.minimum((np.arange(code.size) * width) // code.size, width - 1)
        code_reduced = np.zeros(width, dtype=np.float64)
        wiki_reduced = np.zeros(width, dtype=np.float64)
        np.maximum.at(code_reduced, xpixel, code_prob)
        np.maximum.at(wiki_reduced, xpixel, wiki_prob)
        maximum = max(float(code_reduced.max()), float(wiki_reduced.max()), 1e-30)
        for values, color in ((code_reduced, "#1768ac"), (wiki_reduced, "#d62728")):
            points = [
                (left + x, int(round(bottom - value / maximum * (bottom - top))))
                for x, value in enumerate(values)
            ]
            draw.line(points, fill=color, width=2)
        for value in (0.0, 0.5, 1.0):
            x = int(round(left + value * width))
            draw.line((x, bottom, x, bottom + 5), fill="#333333")
            draw.text((x, bottom + 8), f"{value:g}", fill="#333333", font=tick_font, anchor="ma")
        draw.text(((left + right) // 2, bottom + 40), "score", fill="#333333", font=tick_font, anchor="ma")
    draw.line((1180, 1040, 1230, 1040), fill="#1768ac", width=4)
    draw.text((1240, 1040), "Code", fill="#222222", font=tick_font, anchor="lm")
    draw.line((1340, 1040, 1390, 1040), fill="#d62728", width=4)
    draw.text((1400, 1040), "Wiki positive", fill="#222222", font=tick_font, anchor="lm")
    image.save(path, format="PNG", optimize=True)


def _render_survival_and_roc(
    rows: list[tuple[str, np.ndarray, np.ndarray]], survival_path: Path, roc_path: Path
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    def font(size: int):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()

    survival = Image.new("RGB", (1600, 1100), "white")
    draw = ImageDraw.Draw(survival)
    draw.text((800, 22), "Score threshold curves", fill="#111111", font=font(30), anchor="ma")
    for panel, (name, code, wiki) in enumerate(rows):
        row, column = divmod(panel, 2)
        left, top = 90 + column * 760, 90 + row * 490
        right, bottom = left + 650, top + 390
        draw.rectangle((left, top, right, bottom), outline="#333333", width=2)
        draw.text(((left + right) // 2, top - 30), name, fill="#111111", font=font(21), anchor="ma")
        code_tail = np.cumsum(code[::-1], dtype=np.float64)[::-1] / code.sum()
        wiki_tail = np.cumsum(wiki[::-1], dtype=np.float64)[::-1] / wiki.sum()
        for values, color in ((code_tail, "#1768ac"), (wiki_tail, "#d62728")):
            values = np.maximum(values, 1e-6)
            points = []
            for index, value in enumerate(values):
                x = left + int(round(index / max(values.size - 1, 1) * (right - left)))
                y = bottom - int(round((math.log10(value) + 6.0) / 6.0 * (bottom - top)))
                points.append((x, y))
            draw.line(points, fill=color, width=2)
        for exponent in (0, -2, -4, -6):
            y = bottom - int(round((exponent + 6) / 6 * (bottom - top)))
            draw.line((left, y, right, y), fill="#e5e5e5")
            draw.text((left - 8, y), f"1e{exponent}", fill="#333333", font=font(15), anchor="rm")
        draw.text(((left + right) // 2, bottom + 38), "score threshold", fill="#333333", font=font(16), anchor="ma")
    draw.line((1160, 1040, 1210, 1040), fill="#1768ac", width=4)
    draw.text((1220, 1040), "Code FPR", fill="#222222", font=font(16), anchor="lm")
    draw.line((1345, 1040, 1395, 1040), fill="#d62728", width=4)
    draw.text((1405, 1040), "Wiki TPR", fill="#222222", font=font(16), anchor="lm")
    survival.save(survival_path, format="PNG", optimize=True)

    roc = Image.new("RGB", (1400, 900), "white")
    draw = ImageDraw.Draw(roc)
    draw.text((700, 24), "Wiki-positive ROC (binned)", fill="#111111", font=font(30), anchor="ma")
    left, top, right, bottom = 100, 90, 850, 790
    draw.rectangle((left, top, right, bottom), outline="#333333", width=2)
    draw.line((left, bottom, right, top), fill="#bbbbbb", width=2)
    colors = ("#1768ac", "#d62728", "#2ca02c", "#9467bd")
    for (name, code, wiki), color in zip(rows, colors):
        fpr = np.cumsum(code[::-1], dtype=np.float64) / code.sum()
        tpr = np.cumsum(wiki[::-1], dtype=np.float64) / wiki.sum()
        points = [
            (left + int(round(x * (right - left))), bottom - int(round(y * (bottom - top))))
            for x, y in zip(fpr, tpr)
        ]
        draw.line(points, fill=color, width=3)
    for index, ((name, _, _), color) in enumerate(zip(rows, colors)):
        y = 125 + index * 42
        draw.line((875, y, 920, y), fill=color, width=4)
        draw.text((930, y), name, fill="#222222", font=font(14), anchor="lm")
    draw.text(((left + right) // 2, bottom + 50), "Code FPR", fill="#222222", font=font(18), anchor="ma")
    draw.text((left - 65, (top + bottom) // 2), "Wiki TPR", fill="#222222", font=font(18), anchor="mm")
    roc.save(roc_path, format="PNG", optimize=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-analysis", required=True, type=Path)
    parser.add_argument("--wiki-analysis", required=True, type=Path)
    parser.add_argument("--code-cosine-counts", required=True, type=Path)
    parser.add_argument("--wiki-cosine-counts", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    code_meta = _load_json(args.code_analysis / "metadata.json")
    wiki_meta = _load_json(args.wiki_analysis / "metadata.json")
    if code_meta.get("layers") != list(range(2, 10)) or wiki_meta.get("layers") != list(range(2, 10)):
        raise RuntimeError("both analyses must contain Layers 2--9")
    code_cdf = str(args.code_cosine_counts.resolve())
    if str(Path(code_meta["cosine_counts"]).resolve()) != code_cdf:
        raise RuntimeError("Code analysis did not use the requested Code CDF")
    if str(Path(wiki_meta["cosine_counts"]).resolve()) != code_cdf:
        raise RuntimeError("Wiki analysis must use the Code CDF, not a Wiki-self CDF")

    with np.load(args.code_analysis / "cross_layer_numeric_summary.npz", allow_pickle=False) as data:
        code = {name: data[name].copy() for name in data.files}
    with np.load(args.wiki_analysis / "cross_layer_numeric_summary.npz", allow_pickle=False) as data:
        wiki = {name: data[name].copy() for name in data.files}
    if not np.array_equal(code["layer_numbers"], wiki["layer_numbers"]):
        raise RuntimeError("layer mismatch")
    if not np.array_equal(code["top_percents"], wiki["top_percents"]):
        raise RuntimeError("top-percent mismatch")

    score_rows: list[dict] = []
    threshold_rows: list[dict] = []
    for index, name in enumerate(AGGREGATE_NAMES):
        lower, upper = ((0.0, 1.0) if index < 4 else (-1.0, 1.0))
        pos = wiki["aggregate_hist"][index]
        neg = code["aggregate_hist"][index]
        score_rows.append(
            {
                "score": name,
                "binned_auroc_wiki_positive": _auc(pos, neg),
                "wiki_tokens": int(pos.sum(dtype=np.uint64)),
                "code_tokens": int(neg.sum(dtype=np.uint64)),
                "bin_width": (upper - lower) / pos.size,
            }
        )
        threshold_rows.extend(_threshold_rows(name, pos, neg, lower, upper))

    with np.load(args.code_cosine_counts, allow_pickle=False) as data:
        code_cos = data["counts"].copy()
        cosine_edges = data["bin_edges"].copy()
    with np.load(args.wiki_cosine_counts, allow_pickle=False) as data:
        wiki_cos = data["counts"].copy()
        wiki_edges = data["bin_edges"].copy()
    if not np.array_equal(cosine_edges, wiki_edges):
        raise RuntimeError("cosine histogram edge mismatch")

    layer_rows: list[dict] = []
    for layer in range(2, 10):
        layer_rows.append(
            {
                "metric": "raw_cosine",
                "layer": layer,
                "direction": "higher_is_wiki_like",
                "binned_auroc_wiki_positive": _auc(wiki_cos[layer - 1], code_cos[layer - 1]),
            }
        )
    # joint_metric dimensions: metric, layer, cosine-percentile-bin, metric-bin.
    # metric0=log10(relative L2), metric1=log norm ratio, metric2=symmetric relative L2,
    # metric3=log10(reference RMS). Lower drift and smaller |log norm ratio| are stable.
    for layer_index, layer in enumerate(range(2, 10)):
        wiki_rel = wiki["joint_metric"][0, layer_index].sum(axis=0, dtype=np.uint64)
        code_rel = code["joint_metric"][0, layer_index].sum(axis=0, dtype=np.uint64)
        layer_rows.append(
            {
                "metric": "relative_l2",
                "layer": layer,
                "direction": "lower_is_wiki_like",
                "binned_auroc_wiki_positive": _auc(wiki_rel, code_rel, higher_is_positive=False),
            }
        )
        centers = -4.0 + (np.arange(256) + 0.5) * (8.0 / 256.0)
        order = np.argsort(np.abs(centers))
        wiki_norm = wiki["joint_metric"][1, layer_index].sum(axis=0, dtype=np.uint64)[order]
        code_norm = code["joint_metric"][1, layer_index].sum(axis=0, dtype=np.uint64)[order]
        # Positive/negative log-ratio bins at equal absolute distance are ties.
        wiki_norm = wiki_norm.reshape(128, 2).sum(axis=1, dtype=np.uint64)
        code_norm = code_norm.reshape(128, 2).sum(axis=1, dtype=np.uint64)
        # Ordered from closest to 1 to farthest; reverse AUC direction.
        layer_rows.append(
            {
                "metric": "abs_log_norm_ratio",
                "layer": layer,
                "direction": "lower_is_wiki_like",
                "binned_auroc_wiki_positive": _auc(wiki_norm, code_norm, higher_is_positive=False),
            }
        )

    stable_rows: list[dict] = []
    for top_index, top_percent in enumerate(code["top_percents"]):
        code_counts = _pattern_to_stable_counts(code["pattern_counts"][top_index])
        wiki_counts = _pattern_to_stable_counts(wiki["pattern_counts"][top_index])
        auc = _auc(wiki_counts, code_counts)
        for minimum in (2, 4, 6, 8):
            code_rate = float(code_counts[minimum:].sum(dtype=np.uint64) / code_counts.sum())
            wiki_rate = float(wiki_counts[minimum:].sum(dtype=np.uint64) / wiki_counts.sum())
            stable_rows.append(
                {
                    "top_percent": int(top_percent),
                    "minimum_stable_layers": minimum,
                    "code_rate": code_rate,
                    "wiki_rate": wiki_rate,
                    "wiki_over_code_enrichment": wiki_rate / max(code_rate, 1e-30),
                    "stable_count_binned_auroc": auc,
                    "balanced_prior_precision": wiki_rate / max(wiki_rate + code_rate, 1e-30),
                }
            )

    _write_csv(output / "score_auroc.csv", score_rows)
    _write_csv(output / "score_threshold_curves.csv", threshold_rows)
    _write_csv(output / "layer_metric_auroc.csv", layer_rows)
    _write_csv(output / "stable_count_calibration.csv", stable_rows)
    plotted_scores = [
            (name, code["aggregate_hist"][index], wiki["aggregate_hist"][index])
            for index, name in enumerate(AGGREGATE_NAMES[:4])
    ]
    _render_distributions(plotted_scores, output / "code_wiki_score_distributions.png")
    _render_survival_and_roc(
        plotted_scores,
        output / "score_threshold_curves.png",
        output / "score_roc_curves.png",
    )

    report = {
        "schema": "code_wiki_old_like_calibration_v1",
        "passed": True,
        "gt_assigned": False,
        "wiki_is_analysis_positive_reference": True,
        "code_is_not_assumed_pure_negative": True,
        "percentile_calibration": "both domains transformed with the Code empirical cosine CDF",
        "code_tokens": int(code["aggregate_hist"][0].sum(dtype=np.uint64)),
        "wiki_tokens": int(wiki["aggregate_hist"][0].sum(dtype=np.uint64)),
        "score_auroc": score_rows,
        "best_aggregate_score": max(score_rows, key=lambda row: row["binned_auroc_wiki_positive"]),
        "stable_count_calibration": stable_rows,
        "notes": [
            "AUROC is histogram-binned; percentile-score resolution is 0.001 and raw-cosine-mean resolution is 0.002.",
            "Balanced-prior precision is reported because the 11M Wiki subset and 2.12B Code census sizes are not a meaningful deployment prior.",
            "This is calibration/feasibility analysis only; no hard or soft GT was assigned.",
        ],
    }
    _atomic_json(output / "calibration_summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
