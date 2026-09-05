#!/usr/bin/env python3
"""Completion audit for Code/Wiki old-like feasibility artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--wiki-root", required=True, type=Path)
    parser.add_argument("--code-analysis", required=True, type=Path)
    parser.add_argument("--wiki-analysis", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    errors: list[str] = []

    required_code = [
        "metadata.json", "progress.json", "validation.json", "layer_percentile_correlation.csv",
        "layer_percentile_spearman.png", "top01_jaccard.csv", "top01_jaccard.png",
        "top05_jaccard.csv", "top05_jaccard.png", "top10_jaccard.csv", "top10_jaccard.png",
        "top20_jaccard.csv", "top20_jaccard.png", "stable_layer_count.csv",
        "stable_layer_count.png", "membership_pattern_counts.csv", "high_cos_relative_l2.csv",
        "high_cos_relative_l2.png", "high_cos_norm_ratio.csv", "high_cos_norm_ratio.png",
        "high_cos_reference_norm.csv", "high_cos_reference_norm.png", "candidate_reservoir.npz",
        "reference_rms_proxy_validation.json", "decision_report.md",
    ]
    required_calibration = [
        "calibration_summary.json", "score_auroc.csv", "score_threshold_curves.csv",
        "layer_metric_auroc.csv", "stable_count_calibration.csv",
        "code_wiki_score_distributions.png", "score_threshold_curves.png", "score_roc_curves.png",
    ]
    for base, names in ((args.code_analysis, required_code), (args.calibration, required_calibration)):
        for name in names:
            path = base / name
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing/empty artifact: {path}")

    code_extract = load(args.code_root / "validation.json")
    wiki_extract = load(args.wiki_root / "validation.json")
    code_validation = load(args.code_analysis / "validation.json")
    code_meta = load(args.code_analysis / "metadata.json")
    wiki_validation = load(args.wiki_analysis / "validation.json")
    wiki_meta = load(args.wiki_analysis / "metadata.json")
    calibration = load(args.calibration / "calibration_summary.json")
    proxy = load(args.code_analysis / "reference_rms_proxy_validation.json")

    checks = [
        (code_extract.get("passed") and code_extract.get("deep"), "Code deep extraction validation"),
        (code_extract.get("total_valid_tokens") == 2_123_366_400, "Code token total"),
        (wiki_extract.get("passed") and wiki_extract.get("deep"), "Wiki deep extraction validation"),
        (wiki_extract.get("total_valid_tokens") == 11_010_048, "Wiki token total"),
        (code_validation.get("passed") and code_validation.get("tokens") == 2_123_366_400, "Code cross-layer validation"),
        (wiki_validation.get("passed") and wiki_validation.get("tokens") == 11_010_048, "Wiki cross-layer validation"),
        (code_meta.get("layers") == list(range(2, 10)), "Code layers 2--9"),
        (wiki_meta.get("layers") == list(range(2, 10)), "Wiki layers 2--9"),
        (code_meta.get("excluded_layers") == [1] and wiki_meta.get("excluded_layers") == [1], "Layer 1 excluded"),
        (Path(code_meta.get("cosine_counts", "")).resolve() == Path(wiki_meta.get("cosine_counts", "")).resolve(), "shared Code CDF"),
        (calibration.get("passed") and not calibration.get("gt_assigned"), "calibration passed/no GT"),
        (calibration.get("code_tokens") == 2_123_366_400 and calibration.get("wiki_tokens") == 11_010_048, "calibration totals"),
        (proxy.get("passed") and proxy.get("valid_entries") == 32_768, "RMS proxy validation"),
    ]
    for condition, name in checks:
        if not condition:
            errors.append(f"failed invariant: {name}")

    for base, expected in ((args.code_analysis, 2_123_366_400), (args.wiki_analysis, 11_010_048)):
        with np.load(base / "cross_layer_numeric_summary.npz", allow_pickle=False) as data:
            patterns = data["pattern_counts"]
            aggregates = data["aggregate_hist"]
            if not np.all(patterns.sum(axis=1, dtype=np.uint64) == expected):
                errors.append(f"pattern total mismatch: {base}")
            if not np.all(aggregates.sum(axis=1, dtype=np.uint64) == expected):
                errors.append(f"aggregate total mismatch: {base}")

    with np.load(args.code_analysis / "candidate_reservoir.npz", allow_pickle=False) as data:
        if not np.array_equal(data["layer_numbers"], np.arange(2, 10)):
            errors.append("candidate layer mismatch")
        for index, top in enumerate((1, 5, 10, 20)):
            prefix = f"top{top:02d}_"
            count = data[prefix + "sample_ids"].shape[0]
            if count != 2048 or data[prefix + "cosine"].shape != (count, 8):
                errors.append(f"candidate shape mismatch top{top}")
            if not np.isfinite(data[prefix + "cosine"]).all():
                errors.append(f"candidate nonfinite top{top}")
            if not np.all(data[prefix + "stable_counts"][:, index] >= 6):
                errors.append(f"candidate qualification mismatch top{top}")

    report_text = (args.code_analysis / "decision_report.md").read_text(encoding="utf-8")
    if "Final decision: **A" not in report_text or "No GT label" not in report_text:
        errors.append("decision report is missing final-A/no-GT declarations")

    report = {
        "schema": "old_like_feasibility_completion_validation_v1",
        "passed": not errors,
        "errors": errors,
        "code_tokens": 2_123_366_400,
        "wiki_tokens": 11_010_048,
        "layers": list(range(2, 10)),
        "layer_1_excluded": True,
        "gt_assigned": False,
        "final_decision": "A" if not errors else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".inprogress")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
