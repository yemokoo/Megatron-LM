"""Apply the locked selector rule to the candidate-pass metrics: exact GT.

Rule (identical to the Code bundle95 lock):
  B128 >=7/8 & B256 >=7/8 & T128 >=7/8 & T256 >=7/8 & rel-L2 <=7/8 & |log-r| <=7/8
with per-layer cuts from the Wiki+Code calibration.  Token metrics are the
covering-chunk minima the census stored; negative T fails by the >= comparison.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

LAYERS = 8


def consensus(values: np.ndarray, cuts: np.ndarray, upper: bool, need: int) -> np.ndarray:
    finite = np.isfinite(values)
    passed = (values <= cuts) if upper else (values >= cuts)
    return (passed & finite).sum(axis=-1) >= need


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pass", required=True)
    parser.add_argument("--analysis-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--need", type=int, default=7)
    args = parser.parse_args()

    config = json.loads(Path(args.analysis_config).read_text())
    shards = sorted(glob.glob(f"{args.candidate_pass}/worker_*/shards/*.npz"))
    records = np.concatenate([np.load(f)["reservoir_candidates"] for f in shards])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"schema": "conv_exact_gt_v1", "candidate_tokens": int(records.size),
              "consensus": f">={args.need}/8 per condition, all six AND", "bundles": {}}
    for bundle in ("95", "97", "99"):
        th = config["candidate_thresholds"][bundle]
        b128 = np.asarray(th["B_lower_threshold"]["128"], np.float32)
        b256 = np.asarray(th["B_lower_threshold"]["256"], np.float32)
        t128 = np.asarray(th["T_lower_threshold"]["128"], np.float32)
        t256 = np.asarray(th["T_lower_threshold"]["256"], np.float32)
        rl2 = np.asarray(th["relative_l2_upper_threshold"], np.float32)
        alr = np.asarray(th["abs_log_r_upper_threshold"], np.float32)

        keep = (consensus(records["b_min"][:, 0], b128, False, args.need)
                & consensus(records["b_min"][:, 1], b256, False, args.need)
                & consensus(records["t_min"][:, 0], t128, False, args.need)
                & consensus(records["t_min"][:, 1], t256, False, args.need)
                & consensus(records["rel_l2"], rl2, True, args.need)
                & consensus(records["abs_log_r"], alr, True, args.need))
        selected = records[keep]
        order = np.argsort(selected, order=("source_window_index", "position"))
        selected = selected[order]
        np.save(out_dir / f"bundle_{bundle}_occurrences.npy", selected)
        report["bundles"][bundle] = {
            "selected": int(selected.size),
            "coverage_of_candidates": selected.size / float(records.size),
            "windows": int(np.unique(selected["source_window_index"]).size),
            "documents": int(np.unique(selected["document_id"]).size),
        }

    n95 = {tuple(r) for r in np.stack([np.load(out_dir/"bundle_95_occurrences.npy")["source_window_index"],
                                        np.load(out_dir/"bundle_95_occurrences.npy")["position"]], 1).tolist()}
    n97 = {tuple(r) for r in np.stack([np.load(out_dir/"bundle_97_occurrences.npy")["source_window_index"],
                                        np.load(out_dir/"bundle_97_occurrences.npy")["position"]], 1).tolist()}
    n99 = {tuple(r) for r in np.stack([np.load(out_dir/"bundle_99_occurrences.npy")["source_window_index"],
                                        np.load(out_dir/"bundle_99_occurrences.npy")["position"]], 1).tolist()}
    report["nesting_95_in_97"] = n95 <= n97
    report["nesting_97_in_99"] = n97 <= n99
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
