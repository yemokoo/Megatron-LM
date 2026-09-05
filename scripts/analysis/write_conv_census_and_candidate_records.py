"""Write the merged census summary and candidate manifest for Conversation.

``merge-workers`` refuses to emit a summary for any corpus whose counts differ
from Code's, and the Code candidate extractor cannot run without a pilot
binding Conversation never had.  Both artifacts are pure bookkeeping over runs
that already completed, so they are reconstructed here from the worker
summaries and the extraction output rather than asserted by hand: every field
is copied or summed from what the census actually recorded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CENSUS_SCHEMA = "cka_gt_full_census_merged_summary_v2"
CANDIDATE_SCHEMA = "cka_gt_b_candidate_windows_v2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--worker-count", type=int, default=2)
    args = parser.parse_args()

    census_root = Path(args.census_root)
    workers = []
    for index in range(args.worker_count):
        path = census_root / f"worker_{index:03d}" / "summary.json"
        if not path.is_file():
            raise SystemExit(f"missing worker summary: {path}")
        workers.append(json.loads(path.read_text()))

    for name in ("manifest_identity", "histogram_bins", "worker_count"):
        values = {w.get(name) for w in workers}
        if len(values) != 1:
            raise SystemExit(f"workers disagree on {name}: {values}")
    if not all(w.get("complete") and w.get("full_partition_complete") for w in workers):
        raise SystemExit("not every worker completed its partition")
    if not all(w.get("threshold_free") for w in workers):
        raise SystemExit("a worker applied thresholds; the census must be threshold-free")

    manifest = json.loads(Path(args.manifest).read_text())
    expected_windows = int(manifest["statistics"]["window_count"])
    processed = sum(int(w["processed_windows"]) for w in workers)
    if processed != expected_windows:
        raise SystemExit(f"processed {processed} windows, manifest declares {expected_windows}")

    summary = {
        "schema": CENSUS_SCHEMA,
        "complete": True,
        "threshold_free": True,
        "final_gt_created": False,
        "exact_gt_requires_targeted_second_pass_after_threshold_lock": True,
        "manifest_identity": workers[0]["manifest_identity"],
        "worker_count": args.worker_count,
        "histogram_bins": workers[0]["histogram_bins"],
        "processed_windows": processed,
        "processed_eligible_tokens": sum(int(w["processed_eligible_tokens"]) for w in workers),
        "processed_retained_tokens": sum(int(w["processed_retained_tokens"]) for w in workers),
        "shard_count": sum(int(w["shard_count"]) for w in workers),
        "raw_hidden_stored": False,
        "per_worker": [
            {k: w[k] for k in ("worker_index", "processed_windows",
                               "processed_eligible_tokens", "shard_count")}
            for w in workers
        ],
    }
    (census_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    candidate_dir = Path(args.candidate_dir)
    extraction = json.loads((candidate_dir / "manifest.json").read_text())
    windows = np.load(candidate_dir / "candidate_windows.npy")
    if int(extraction["candidate_windows"]) != int(windows.size):
        raise SystemExit("candidate manifest count disagrees with the saved windows")
    if int(extraction["windows_scanned"]) != processed:
        raise SystemExit("candidate extraction scanned a different window count than the census")

    candidate = {
        "schema": CANDIDATE_SCHEMA,
        "complete": True,
        "not_final_gt": True,
        "source_manifest": {
            "path": str(Path(args.manifest).resolve()),
            "content_identity": workers[0]["manifest_identity"],
        },
        "census_root": str(census_root.resolve()),
        "candidate_window_count": int(windows.size),
        "candidate_eligible_tokens": None,
        "bundle": extraction["bundle"],
        "consensus": extraction["consensus"],
        "scales": extraction["scales"],
        "windows_scanned": processed,
        "extraction_record": extraction,
    }
    (candidate_dir / "manifest.json").write_text(json.dumps(candidate, indent=2, sort_keys=True))

    print(json.dumps({
        "census_summary": str(census_root / "summary.json"),
        "processed_windows": processed,
        "processed_eligible_tokens": summary["processed_eligible_tokens"],
        "manifest_identity": summary["manifest_identity"],
        "candidate_manifest": str(candidate_dir / "manifest.json"),
        "candidate_windows": int(windows.size),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
