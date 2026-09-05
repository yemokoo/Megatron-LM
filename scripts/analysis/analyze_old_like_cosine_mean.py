#!/usr/bin/env python3
"""CPU-only cross-layer cosine analysis for old-like GT review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LAYERS = np.arange(2, 10)
COVERAGES = np.array([0.005, 0.01, 0.02, 0.05, 0.10, 0.20])


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def overlay(path: Path, code: np.ndarray, wiki: np.ndarray, title: str) -> None:
    edges = np.linspace(-1.0, 1.0, 401)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, log in zip(axes, (False, True)):
        ax.hist(code, bins=edges, weights=np.full(code.size, 1 / code.size),
                histtype="step", linewidth=1.8, label=f"Code (n={code.size:,})")
        ax.hist(wiki, bins=edges, weights=np.full(wiki.size, 1 / wiki.size),
                histtype="step", linewidth=1.8, label=f"Wiki (n={wiki.size:,})")
        ax.set_ylabel("token fraction / bin")
        if log:
            ax.set_yscale("log")
        ax.grid(alpha=.2)
        ax.legend()
    axes[-1].set_xlabel("mean cosine across layer 2-9")
    axes[-1].set_xlim(-1, 1)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def excluded_plot(path: Path, excluded: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    if excluded.size:
        ax.hist(excluded, bins=np.linspace(-1, 1, 401), weights=np.full(excluded.size, 1 / excluded.size),
                histtype="step", linewidth=1.8)
    else:
        ax.text(.5, .5, "No token excluded: no evidence-based dead cut", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
    ax.set(xlim=(-1, 1), xlabel="mean cosine across layer 2-9", ylabel="token fraction / bin",
           title="Tokens excluded by reference-norm gate")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--wiki", type=Path, required=True)
    p.add_argument("--cut-decisions", type=Path, required=True)
    p.add_argument("--code-cosine-counts", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def load(path: Path) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k].copy() for k in ("sample_ids", "positions", "token_ids", "cosine", "relative_l2", "reference_rms")}
    code, wiki = load(args.code), load(args.wiki)
    code_mean = code["cosine"].mean(axis=1, dtype=np.float64)
    wiki_mean = wiki["cosine"].mean(axis=1, dtype=np.float64)
    code_rel = code["relative_l2"].mean(axis=1, dtype=np.float64)
    wiki_rel = wiki["relative_l2"].mean(axis=1, dtype=np.float64)

    with args.cut_decisions.open(newline="", encoding="utf-8") as h:
        decisions = list(csv.DictReader(h))
    active = np.array([float(r["active_cut_rms"]) if r["active_cut_rms"] else np.nan for r in decisions])
    candidate = np.array([float(r["mode_over_10_candidate_rms"]) for r in decisions])
    active_available = bool(np.isfinite(active).all())
    # Under the stated rule, a clean-unimodal layer has no active cut. Diagnostic mode/10 is
    # still evaluated, but must never silently become the production gate.
    diagnostic_low_code = code["reference_rms"] < candidate
    diagnostic_low_wiki = wiki["reference_rms"] < candidate
    active_low_code = code["reference_rms"] < active if active_available else np.zeros_like(diagnostic_low_code)
    active_low_wiki = wiki["reference_rms"] < active if active_available else np.zeros_like(diagnostic_low_wiki)
    gates = {
        "before": (np.ones(code_mean.size, bool), np.ones(wiki_mean.size, bool)),
        "active_any": (~active_low_code.any(1), ~active_low_wiki.any(1)),
        "active_majority": (~(active_low_code.sum(1) >= 4), ~(active_low_wiki.sum(1) >= 4)),
        "mode_over_10_diagnostic_any": (~diagnostic_low_code.any(1), ~diagnostic_low_wiki.any(1)),
        "mode_over_10_diagnostic_majority": (~(diagnostic_low_code.sum(1) >= 4), ~(diagnostic_low_wiki.sum(1) >= 4)),
    }

    overlay(args.output_dir / "raw_cosine_mean_before_code_vs_wiki.png", code_mean, wiki_mean,
            "Raw mean cosine, layers 2-9 (before gate)")
    cmask, wmask = gates["active_majority"]
    overlay(args.output_dir / "raw_cosine_mean_after_majority_gate_code_vs_wiki.png", code_mean[cmask], wiki_mean[wmask],
            "Raw mean cosine, layers 2-9 (after majority gate)")
    excluded_plot(args.output_dir / "raw_cosine_mean_excluded_by_gate.png", code_mean[~cmask])

    rows = []
    for gate_name, (cg, wg) in gates.items():
        eligible = code_mean[cg]
        for coverage in COVERAGES:
            threshold = float(np.quantile(eligible, 1.0 - coverage, method="linear"))
            rows.append({
                "gate": gate_name,
                "target_code_coverage_among_eligible": float(coverage),
                "threshold_raw_cosine_mean": threshold,
                "code_gate_pass_fraction": float(cg.mean()),
                "wiki_gate_pass_fraction": float(wg.mean()),
                "actual_code_fraction_of_all": float(np.mean(cg & (code_mean >= threshold))),
                "wiki_recall_of_all": float(np.mean(wg & (wiki_mean >= threshold))),
                "code_mean_relative_l2_selected_median": float(np.median(code_rel[cg & (code_mean >= threshold)])),
                "wiki_mean_relative_l2_selected_median": float(np.median(wiki_rel[wg & (wiki_mean >= threshold)])),
            })
    save_csv(args.output_dir / "coverage_threshold_wiki_recall.csv", rows)

    # Layerwise full-Code percentiles and repeated high-cos layers in the fixed subset.
    with np.load(args.code_cosine_counts, allow_pickle=False) as z:
        counts = z["counts"][1:].astype(np.uint64)
    thresholds = {}
    stable_rows = []
    centers = -1.0 + (np.arange(counts.shape[1]) + .5) / 10000.0
    for pcent in (1, 5, 10, 20):
        qs = []
        for li in range(8):
            cdf = np.cumsum(counts[li], dtype=np.uint64) / counts[li].sum()
            qs.append(float(centers[np.searchsorted(cdf, 1 - pcent / 100)]))
        qs = np.array(qs)
        thresholds[str(pcent)] = qs.tolist()
        cc = (code["cosine"] >= qs).sum(1)
        wc = (wiki["cosine"] >= qs).sum(1)
        for nlayer in range(9):
            stable_rows.append({"top_percent": pcent, "stable_layer_count": nlayer,
                                "code_fraction": float(np.mean(cc == nlayer)),
                                "wiki_fraction": float(np.mean(wc == nlayer))})
    save_csv(args.output_dir / "stable_layer_count_distribution.csv", stable_rows)

    summary = {
        "complete": True,
        "layers": LAYERS.tolist(),
        "active_gate_available": active_available,
        "decision": "No active norm gate; all layers were clean-unimodal. mode/10 is diagnostic only.",
        "subsample_size": {"code": int(code_mean.size), "wiki": int(wiki_mean.size)},
        "raw_cosine_mean": {
            "code": {"mean": float(code_mean.mean()), "median": float(np.median(code_mean)),
                     "q90": float(np.quantile(code_mean,.9)), "q95": float(np.quantile(code_mean,.95)),
                     "q99": float(np.quantile(code_mean,.99)), "q995": float(np.quantile(code_mean,.995))},
            "wiki": {"mean": float(wiki_mean.mean()), "median": float(np.median(wiki_mean)),
                     "q90": float(np.quantile(wiki_mean,.9)), "q95": float(np.quantile(wiki_mean,.95)),
                     "q99": float(np.quantile(wiki_mean,.99)), "q995": float(np.quantile(wiki_mean,.995))},
        },
        "gate_pass_fractions": {name: {"code": float(c.mean()), "wiki": float(w.mean())}
                                for name, (c, w) in gates.items()},
        "layerwise_code_full_census_cosine_thresholds": thresholds,
        "top1_mean_relative_l2": {
            "before_code": float(np.median(code_rel[code_mean >= np.quantile(code_mean,.99)])),
            "before_wiki_at_code_threshold": float(np.median(wiki_rel[wiki_mean >= np.quantile(code_mean,.99)])),
            "after_active_majority_code": float(np.median(code_rel[cmask & (code_mean >= np.quantile(code_mean[cmask],.99))])),
        },
    }
    (args.output_dir / "cosine_mean_analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
