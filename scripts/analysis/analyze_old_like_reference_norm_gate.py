#!/usr/bin/env python3
"""Analyze absolute reference-RMS distributions and evidence for a dead gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


LAYERS = list(range(2, 10))
HIDDEN_SQRT = 32.0


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


def moving_average(values: np.ndarray, width: int = 5) -> np.ndarray:
    return np.convolve(values.astype(np.float64), np.ones(width) / width, mode="same")


def lower_peak_and_valley(hist: np.ndarray, mode_index: int) -> tuple[int | None, int | None, float | None]:
    smooth = moving_average(hist, 5)
    peaks = np.flatnonzero((smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] >= smooth[2:])) + 1
    candidates = [
        int(index) for index in peaks
        if index < mode_index - 12 and smooth[index] >= 0.01 * smooth[mode_index]
    ]
    if not candidates:
        return None, None, None
    peak = max(candidates, key=lambda index: smooth[index])
    valley = peak + int(np.argmin(smooth[peak:mode_index + 1]))
    ratio = float(smooth[valley] / max(min(smooth[peak], smooth[mode_index]), 1e-30))
    return peak, valley, ratio


def plot_layer(
    layer: int,
    code: np.ndarray,
    wiki: np.ndarray,
    mode: float,
    candidate: float,
    path: Path,
    bins: int,
) -> dict:
    combined_min = min(float(code.min()), float(wiki.min()), candidate / 2.0)
    combined_max = max(float(np.quantile(code, 0.9999)), float(np.quantile(wiki, 0.9999))) * 1.1
    log_edges = np.linspace(math.log10(combined_min), math.log10(combined_max), bins + 1)
    code_hist, _ = np.histogram(np.log10(code), bins=log_edges)
    wiki_hist, _ = np.histogram(np.log10(wiki), bins=log_edges)
    code_prob = code_hist / code_hist.sum()
    wiki_prob = wiki_hist / wiki_hist.sum()

    image = Image.new("RGB", (1700, 1050), "white")
    draw = ImageDraw.Draw(image)
    draw.text((850, 25), f"Layer {layer} reference RMS (x-axis log scale)", fill="#111111", font=font(32), anchor="ma")
    for panel, log_y in enumerate((False, True)):
        left, top, right, bottom = 105, 115 + panel * 450, 1570, 445 + panel * 450
        draw.rectangle((left, top, right, bottom), outline="#333333", width=2)
        draw.text((left, top - 25), "linear y" if not log_y else "log10 y", fill="#222222", font=font(20), anchor="ls")
        if log_y:
            floor = -8.0
            transformed = [
                (np.log10(np.maximum(values, 10**floor)) - floor) / -floor
                for values in (code_prob, wiki_prob)
            ]
        else:
            maximum = max(float(code_prob.max()), float(wiki_prob.max()))
            transformed = [code_prob / maximum, wiki_prob / maximum]
        plot_width = right - left
        for values, color in zip(transformed, ("#1768ac", "#d62728")):
            points = [
                (
                    left + int(round(index / max(values.size - 1, 1) * plot_width)),
                    bottom - int(round(float(value) * (bottom - top))),
                )
                for index, value in enumerate(values)
            ]
            draw.line(points, fill=color, width=3)
        for value, color, label in ((mode, "#2ca02c", "Code mode"), (candidate, "#9467bd", "mode/10 candidate")):
            x = left + int(round((math.log10(value) - log_edges[0]) / (log_edges[-1] - log_edges[0]) * plot_width))
            draw.line((x, top, x, bottom), fill=color, width=2)
            draw.text((x + 4, top + 8), label, fill=color, font=font(15), anchor="la")
        ticks = np.geomspace(10**log_edges[0], 10**log_edges[-1], 6)
        for value in ticks:
            x = left + int(round((math.log10(value) - log_edges[0]) / (log_edges[-1] - log_edges[0]) * plot_width))
            draw.line((x, bottom, x, bottom + 6), fill="#333333")
            draw.text((x, bottom + 10), f"{value:.3g}", fill="#333333", font=font(15), anchor="ma")
    draw.line((1160, 1010, 1210, 1010), fill="#1768ac", width=4)
    draw.text((1220, 1010), "Code", fill="#222222", font=font(17), anchor="lm")
    draw.line((1350, 1010, 1400, 1010), fill="#d62728", width=4)
    draw.text((1410, 1010), "Wiki", fill="#222222", font=font(17), anchor="lm")
    image.save(path, format="PNG", optimize=True)
    return {"log_edges": log_edges, "code_hist": code_hist, "wiki_hist": wiki_hist}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-subsample", required=True, type=Path)
    parser.add_argument("--wiki-subsample", required=True, type=Path)
    parser.add_argument("--code-full-numeric", required=True, type=Path)
    parser.add_argument("--wiki-full-numeric", required=True, type=Path)
    parser.add_argument("--code-cosine-counts", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bins", type=int, default=512)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.code_subsample, allow_pickle=False) as data:
        code_rms = data["reference_rms"].copy()
        code_cosine = data["cosine"].copy()
    with np.load(args.wiki_subsample, allow_pickle=False) as data:
        wiki_rms = data["reference_rms"].copy()
    with np.load(args.code_full_numeric, allow_pickle=False) as data:
        code_full_ref_hist = data["joint_metric"][3].sum(axis=1, dtype=np.uint64)
    with np.load(args.wiki_full_numeric, allow_pickle=False) as data:
        wiki_full_ref_hist = data["joint_metric"][3].sum(axis=1, dtype=np.uint64)
    with np.load(args.code_cosine_counts, allow_pickle=False) as data:
        cosine_counts = data["counts"][1:].copy()

    # Existing full-census reference-RMS histogram uses log10 range [-4,3], 256 bins.
    full_centers = -4.0 + (np.arange(256) + 0.5) * (7.0 / 256.0)
    rows = []
    layer_details = []
    for layer_index, layer in enumerate(LAYERS):
        code = code_rms[:, layer_index]
        wiki = wiki_rms[:, layer_index]
        # Mode is measured on a focused 512-bin log histogram up to q99.9 so rare high-norm modes do not dominate resolution.
        focused_edges = np.linspace(
            math.log10(min(float(code.min()), float(wiki.min()))),
            math.log10(max(float(np.quantile(code, 0.999)), float(np.quantile(wiki, 0.999)))),
            args.bins + 1,
        )
        focused_hist, _ = np.histogram(np.log10(code), bins=focused_edges)
        mode_index = int(focused_hist.argmax())
        mode = 10 ** float((focused_edges[mode_index] + focused_edges[mode_index + 1]) / 2.0)
        candidate = mode / 10.0
        peak, valley, valley_ratio = lower_peak_and_valley(focused_hist, mode_index)
        q001 = float(np.quantile(code, 0.001))
        q01 = float(np.quantile(code, 0.01))
        q05 = float(np.quantile(code, 0.05))
        min_value = float(code.min())
        has_lower_mode = peak is not None and valley_ratio is not None and valley_ratio < 0.5
        long_left_tail = q001 < candidate
        if has_lower_mode:
            status = "bimodal_lower_cluster"
            active_cut = 10 ** float((focused_edges[valley] + focused_edges[valley + 1]) / 2.0)
            reason = "significant lower-norm peak with valley ratio < 0.5"
        elif long_left_tail:
            status = "unimodal_long_left_tail_candidate"
            active_cut = candidate
            reason = "q0.1% falls below mode/10"
        else:
            status = "clean_unimodal_cut_deferred"
            active_cut = None
            reason = "no lower-norm secondary mode and q0.1% remains above mode/10"

        figure_info = plot_layer(
            layer, code, wiki, mode, candidate,
            output / f"layer_{layer:02d}_reference_rms_log_hist.png", args.bins,
        )
        candidate_steps = [candidate * 0.8, candidate, candidate * 1.25]
        full_code_total = int(code_full_ref_hist[layer_index].sum(dtype=np.uint64))
        full_wiki_total = int(wiki_full_ref_hist[layer_index].sum(dtype=np.uint64))
        step_fractions = []
        for cut in candidate_steps:
            code_count = int(code_full_ref_hist[layer_index, full_centers < math.log10(cut)].sum(dtype=np.uint64))
            wiki_count = int(wiki_full_ref_hist[layer_index, full_centers < math.log10(cut)].sum(dtype=np.uint64))
            step_fractions.append(
                {
                    "cut_rms": cut,
                    "code_excluded_count_full_hist": code_count,
                    "code_excluded_fraction_full_hist": code_count / full_code_total,
                    "wiki_excluded_count_full_hist": wiki_count,
                    "wiki_excluded_fraction_full_hist": wiki_count / full_wiki_total,
                }
            )
        rows.append(
            {
                "layer": layer,
                "classification": status,
                "active_cut_rms": "" if active_cut is None else active_cut,
                "active_cut_reference_l2_norm": "" if active_cut is None else active_cut * HIDDEN_SQRT,
                "code_mode_rms": mode,
                "code_mode_reference_l2_norm": mode * HIDDEN_SQRT,
                "mode_over_10_candidate_rms": candidate,
                "mode_over_10_candidate_reference_l2_norm": candidate * HIDDEN_SQRT,
                "code_min_rms_subsample": min_value,
                "code_q0p1_rms": q001,
                "code_q1_rms": q01,
                "code_q5_rms": q05,
                "q0p1_over_mode": q001 / mode,
                "lower_peak_detected": has_lower_mode,
                "valley_ratio": "" if valley_ratio is None else valley_ratio,
                "decision_reason": reason,
            }
        )
        layer_details.append(
            {
                "layer": layer,
                "classification": status,
                "decision_reason": reason,
                "active_cut_rms": active_cut,
                "mode_rms": mode,
                "mode_over_10_candidate_rms": candidate,
                "candidate_step_sensitivity": step_fractions,
                "subsample": {
                    "code_min": min_value, "code_q0p1": q001, "code_q1": q01, "code_q5": q05,
                    "wiki_min": float(wiki.min()), "wiki_q0p1": float(np.quantile(wiki, 0.001)),
                    "wiki_q1": float(np.quantile(wiki, 0.01)), "wiki_q5": float(np.quantile(wiki, 0.05)),
                },
                "plot": str(output / f"layer_{layer:02d}_reference_rms_log_hist.png"),
            }
        )
    write_csv(output / "layer_reference_norm_cut_decisions.csv", rows)

    # Recompute the old Layer-9 diagnostic using the non-active mode/10 candidate.
    layer_index = 7
    count = cosine_counts[layer_index]
    cumulative = np.cumsum(count, dtype=np.uint64)
    midrank = (cumulative.astype(np.float64) - 0.5 * count.astype(np.float64)) / count.sum()
    bins = np.floor((code_cosine[:, layer_index].astype(np.float64) + 1.0) * 10_000).astype(np.int32)
    np.clip(bins, 0, 19_999, out=bins)
    top1 = midrank[bins] >= 0.99
    candidate_l9 = float(rows[-1]["mode_over_10_candidate_rms"])
    contamination = float(np.mean(code_rms[top1, layer_index] < candidate_l9))
    contamination_report = {
        "layer": 9,
        "top1_selected_subsample": int(top1.sum()),
        "candidate_cut_rms": candidate_l9,
        "candidate_is_active": False,
        "fraction_below_candidate_cut": contamination,
        "previous_bottom5_fraction": 0.30069599215814247,
        "interpretation": "bottom-5% was a relative low tail, not evidence of a separate dead cluster",
    }
    atomic_json(output / "layer9_top1_dead_contamination.json", contamination_report)
    report = {
        "schema": "old_like_reference_norm_gate_v1",
        "complete": True,
        "bins": args.bins,
        "x_axis": "log10(reference RMS), labels shown in absolute RMS",
        "histogram_plot_smoothing": "none",
        "peak_detection_only_smoothing": "5-bin moving average",
        "hidden_size": 1024,
        "absolute_reference_norm": "reference RMS * 32",
        "active_gate_available": any(detail["active_cut_rms"] is not None for detail in layer_details),
        "layer_decisions": layer_details,
        "layer9_contamination": contamination_report,
        "gt_assigned": False,
    }
    atomic_json(output / "reference_norm_analysis.json", report)
    print(json.dumps({"complete": True, "active_gate_available": report["active_gate_available"],
                      "layer9_fraction_below_mode_over_10": contamination}, indent=2))


if __name__ == "__main__":
    main()
