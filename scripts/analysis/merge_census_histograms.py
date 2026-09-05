"""Merge per-worker census histograms without the Code-specific coverage guard.

``cka_gt_full_census.py merge-workers`` asserts the merged coverage equals the
known Code corpus counts, which is correct for Code and wrong for every other
corpus.  This merges the histogram tensors only, and reports the coverage it
actually observed instead of comparing it to Code.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-root", required=True)
    parser.add_argument("--expected-windows", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.census_root)
    workers = sorted(p for p in root.glob("worker_*") if p.is_dir())
    if not workers:
        raise SystemExit(f"no worker directories under {root}")

    merged: dict[str, np.ndarray] = {}
    edges: dict[str, np.ndarray] = {}
    processed = 0
    for worker in workers:
        summary_path = worker / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            processed += int(summary.get("processed_windows", 0))
        for shard in sorted(worker.glob("shards/*.npz")):
            z = np.load(shard)
            for key in z.files:
                if not key.startswith("hist__"):
                    continue
                if key.endswith("__edges"):
                    if key not in edges:
                        edges[key] = z[key]
                    elif not np.array_equal(edges[key], z[key]):
                        raise SystemExit(f"histogram edges differ across shards: {key}")
                    continue
                value = z[key]
                merged[key] = value if key not in merged else merged[key] + value

    if not merged:
        raise SystemExit("no histogram tensors found")
    merged.update(edges)
    out = root / "histograms.npz"
    np.savez(out, **merged)

    total = int(merged[next(k for k in merged if k.startswith("hist__raw_b_") and k.endswith("__counts"))].sum()
                + merged[next(k for k in merged if k.startswith("hist__raw_b_") and k.endswith("__underflow"))].sum()
                + merged[next(k for k in merged if k.startswith("hist__raw_b_") and k.endswith("__overflow"))].sum())
    report = {
        "census_root": str(root),
        "workers": len(workers),
        "processed_windows": processed,
        "expected_windows": args.expected_windows,
        "windows_match": None if args.expected_windows is None else processed == args.expected_windows,
        "histogram_keys": len([k for k in merged if k.endswith("__counts")]),
        "raw_b_128_observations": total,
        "output": str(out),
    }
    (root / "merged_histograms_metadata.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
