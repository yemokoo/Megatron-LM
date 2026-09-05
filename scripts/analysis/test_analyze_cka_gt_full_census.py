#!/usr/bin/env python3
"""CPU-only tests for the full-census CKA threshold review."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from analyze_cka_gt_full_census import (  # noqa: E402
    LEVELS,
    analyze_reservoir,
    condition_consensus,
    document_bootstrap_intervals,
    histogram_threshold_bounds,
    load_frozen_bundles,
    pairwise_jaccard_rows,
    run_analysis,
)
from cka_gt_full_census import RESERVOIR_DTYPE  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def config_payload(raw_quantiles: str = "/does/not/exist/open_calibration.parquet") -> dict:
    candidates = {}
    values = {
        95: (0.90, 0.90, 0.10, 0.05),
        97: (0.80, 0.80, 0.20, 0.10),
        99: (0.70, 0.70, 0.30, 0.20),
    }
    for level, (b, t, l2, logr) in values.items():
        candidates[str(level)] = {
            "level": level,
            "B_lower_threshold": {
                "128": [b] * 8,
                "256": [b] * 8,
            },
            "T_lower_threshold": {
                "128": [t] * 8,
                "256": [t] * 8,
            },
            "relative_l2_upper_threshold": [l2] * 8,
            "abs_log_r_upper_threshold": [logr] * 8,
        }
    return {
        "schema": "cka_gt_pilot_postprocess_v1",
        "candidate_thresholds": candidates,
        "raw_quantiles": raw_quantiles,
        "sealed_test_path": "/must/not/be/opened/test_metrics.json",
        "input_inventory_digest_sha256": "a" * 64,
    }


def synthetic_scores() -> np.ndarray:
    rows = np.zeros(300, dtype=RESERVOIR_DTYPE)
    rows["priority"] = np.arange(300, dtype=np.uint64)
    rows["sample_order"] = np.arange(300) // 4
    rows["source_window_index"] = rows["sample_order"]
    rows["document_id"] = np.arange(300) // 10
    rows["window_offset"] = (np.arange(300) // 4) % 5 * 512
    rows["position"] = np.arange(300) % 512
    rows["token_id"] = np.arange(300) % 20

    # First 120 pass 95, next 80 pass 97, next 70 pass 99, last 30 fail.
    for start, stop, b, t, l2, logr in (
        (0, 120, 0.95, 0.95, 0.05, 0.02),
        (120, 200, 0.85, 0.85, 0.15, 0.08),
        (200, 270, 0.75, 0.75, 0.25, 0.15),
        (270, 300, 0.20, 0.20, 1.00, 1.00),
    ):
        rows["b_min"][start:stop] = b
        rows["t_min"][start:stop] = t
        rows["rel_l2"][start:stop] = l2
        rows["abs_log_r"][start:stop] = logr

    # These 20 still pass every condition-specific 7/8 rule, but failures are
    # on six different layers, so only two layers pass every condition jointly.
    for row in range(20):
        rows["b_min"][row, 0, 0] = 0.0
        rows["b_min"][row, 1, 1] = 0.0
        rows["t_min"][row, 0, 2] = -1.0
        rows["t_min"][row, 1, 3] = -1.0
        rows["rel_l2"][row, 4] = 2.0
        rows["abs_log_r"][row, 5] = 2.0
    return rows


def write_histograms(path: Path) -> None:
    ranges = {
        "raw_b_128": (0.0, 1.0), "raw_b_256": (0.0, 1.0),
        "token_min_b_128": (0.0, 1.0), "token_min_b_256": (0.0, 1.0),
        "token_min_t_128": (-10.0, 10.0), "token_min_t_256": (-10.0, 10.0),
        "relative_l2": (0.0, 4.0), "abs_log_r": (0.0, 2.0),
    }
    arrays = {}
    for metric, (low, high) in ranges.items():
        edges = np.linspace(low, high, 65)
        counts = np.zeros((8, 64), dtype=np.int64)
        counts[:, 10:55] = np.arange(1, 46, dtype=np.int64)[None, :]
        arrays[f"hist__{metric}__edges"] = edges
        arrays[f"hist__{metric}__counts"] = counts
        arrays[f"hist__{metric}__underflow"] = np.zeros(8, dtype=np.int64)
        arrays[f"hist__{metric}__overflow"] = np.zeros(8, dtype=np.int64)
        arrays[f"hist__{metric}__nonfinite"] = np.zeros(8, dtype=np.int64)
    np.savez_compressed(path, **arrays)


class ConsensusTest(unittest.TestCase):
    def test_six_valid_requires_six_passes_and_seven_requires_seven(self) -> None:
        values = np.ones((3, 8), dtype=np.float32)
        values[0, :2] = np.nan              # six valid, six pass
        values[1, :2] = np.nan
        values[1, 2] = 0.0                 # six valid, five pass
        values[2, 0] = np.nan
        values[2, 1] = 0.0                 # seven valid, six pass
        result = condition_consensus(values, np.full(8, 0.5), comparison="ge")
        self.assertEqual(result.eligible.tolist(), [True, True, True])
        self.assertEqual(result.passed.tolist(), [True, False, False])

    def test_condition_specific_and_same_layer_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analysis_config.json"
            path.write_text(json.dumps(config_payload()))
            _, bundles = load_frozen_bundles(path)
            summaries, masks = analyze_reservoir(synthetic_scores(), bundles, chunk_rows=37)
        self.assertEqual([summaries[level]["counts"]["selected"] for level in LEVELS], [120, 200, 270])
        self.assertEqual(summaries[95]["same_layer"]["same_layer_count"], 100)
        self.assertAlmostEqual(summaries[95]["same_layer"]["jaccard"], 100 / 120)
        overlaps = pairwise_jaccard_rows(masks)
        self.assertTrue(all(row["left_subset_of_right"] for row in overlaps))
        self.assertEqual(overlaps[0]["intersection"], 120)


class BootstrapAndHistogramTest(unittest.TestCase):
    def test_document_bootstrap_is_seed_deterministic(self) -> None:
        documents = np.repeat(np.arange(30), 10)
        masks = {
            95: np.arange(300) < 120,
            97: np.arange(300) < 200,
            99: np.arange(300) < 270,
        }
        first = document_bootstrap_intervals(
            documents, masks, full_eligible_tokens=3_000,
            repetitions=50, seed=11,
        )
        second = document_bootstrap_intervals(
            documents, masks, full_eligible_tokens=3_000,
            repetitions=50, seed=11,
        )
        self.assertEqual(first, second)
        self.assertEqual(first[95]["estimated_full_count"], 1200)

    def test_histogram_threshold_crossing_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            config.write_text(json.dumps(config_payload()))
            _, bundles = load_frozen_bundles(config)
            hist = root / "histograms.npz"
            write_histograms(hist)
            with np.load(hist, allow_pickle=False) as archive:
                rows = histogram_threshold_bounds(
                    {key: archive[key] for key in archive.files}, bundles
                )
        self.assertEqual(len(rows), 8 * 3 * 8)
        self.assertTrue(all(row["pass_count_liberal"] >= row["pass_count_conservative"] for row in rows))


class EndToEndTest(unittest.TestCase):
    def test_machine_report_and_sealed_test_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            census, output = root / "census", root / "review"
            census.mkdir()
            scores = synthetic_scores()
            reservoir = census / "token_score_reservoir.npz"
            np.savez(reservoir, token_scores=scores)
            histograms = census / "histograms.npz"
            write_histograms(histograms)
            summary = {
                "schema": "cka_gt_full_census_merged_summary_v2",
                "complete": True, "threshold_free": True, "final_gt_created": False,
                "processed_eligible_tokens": 30_000,
                "histograms": {"size_bytes": histograms.stat().st_size, "sha256": sha256(histograms)},
                "token_score_reservoir": {"size_bytes": reservoir.stat().st_size, "sha256": sha256(reservoir)},
            }
            (census / "summary.json").write_text(json.dumps(summary))
            config = root / "analysis_config.json"
            raw_quantiles = root / "raw_unit_quantiles.parquet"
            raw_quantiles.write_bytes(b"synthetic-open-wiki-calibration")
            quantile_manifest = {
                "schema": "cka_gt_raw_unit_exact_quantiles_v1",
                "output_sha256": sha256(raw_quantiles),
                "input_inventory_digest_sha256": "a" * 64,
                "groups": [
                    {"domain": "wiki", "split": "calibration", "metric": "cka"}
                ],
            }
            (root / "raw_unit_quantile_accumulator_manifest.json").write_text(
                json.dumps(quantile_manifest)
            )
            config.write_text(json.dumps(config_payload(str(raw_quantiles))))
            summary["pilot_overlay"] = {
                "path": str(config), "sha256": sha256(config),
                "candidate_thresholds": config_payload()["candidate_thresholds"],
            }
            summary["reservoir_rows"] = len(scores)
            (census / "summary.json").write_text(json.dumps(summary))
            payload = run_analysis(
                census_root=census, analysis_config=config, output_dir=output,
                bootstrap_repetitions=50, bootstrap_seed=7, chunk_rows=31,
                min_recommendation_hits=5,
            )
            self.assertTrue(payload["complete"])
            self.assertFalse(payload["sealed_test_opened"])
            self.assertFalse(payload["exact_gt_created"])
            self.assertEqual(payload["recommendation"]["recommended_bundle"], 97)
            self.assertEqual(payload["document_bootstrap"]["95"]["estimated_full_count"], 12_000)
            for relative in (
                "analysis.json", "REPORT.md", "calibration_binding.json",
                "tables/candidate_thresholds.csv", "tables/marginal_pass_fractions.csv",
                "tables/sequential_B_T_M.csv", "tables/same_layer_diagnostic.csv",
                "tables/pairwise_jaccard.csv", "tables/concentration.csv",
                "tables/token_id_repetition.csv",
                "tables/histogram_threshold_pass_bounds.csv",
                "histograms/token_min_b_128_zoom.svg",
            ):
                self.assertTrue((output / relative).is_file(), relative)
            report = (output / "REPORT.md").read_text()
            self.assertIn("does **not** create exact GT", report)
            self.assertIn("sealed test opened: **false**", report)


if __name__ == "__main__":
    unittest.main()
