#!/usr/bin/env python3
"""Summarize extraction progress without assigning GT or making histograms.

This is deliberately a lightweight control-plane utility.  Distribution plots
and preserved/adapted labels belong to the later analysis stage, after every
paired-forward shard has passed validation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".inprogress")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def summarize(root: Path, expected_workers: int) -> dict:
    workers = []
    for index in range(expected_workers):
        worker_dir = root / f"rank_{index:03d}"
        run = _read_json(worker_dir / "run_metadata.json")
        progress = _read_json(worker_dir / "progress.json")
        final = _read_json(worker_dir / "metadata.json")
        row = {
            "worker_index": index,
            "path": str(worker_dir),
            "started": run is not None,
            "completed": bool(final and final.get("completed")),
        }
        if run:
            row.update(
                {
                    "partition_start_sample": run["partition_start_sample"],
                    "partition_samples": run["partition_samples"],
                    "reference_tracker_step": run.get("reference_tracker_step"),
                    "current_tracker_step": run.get("current_tracker_step"),
                }
            )
        if progress:
            row.update(
                {
                    "completed_samples": progress["completed_samples"],
                    "completed_valid_tokens": progress["completed_valid_tokens"],
                    "tokens_per_second": progress.get("tokens_per_second_this_process"),
                    "eta_seconds": progress.get("eta_seconds"),
                    "output_bytes": progress.get("output_bytes", 0),
                    "nonfinite_count": progress.get("nonfinite_count", 0),
                    "shards": progress.get("next_shard", 0),
                }
            )
        workers.append(row)

    completed_samples = sum(int(row.get("completed_samples", 0)) for row in workers)
    partition_samples = sum(int(row.get("partition_samples", 0)) for row in workers)
    total_rate = sum(float(row.get("tokens_per_second") or 0.0) for row in workers)
    total_output = sum(int(row.get("output_bytes", 0)) for row in workers)
    total_nonfinite = sum(int(row.get("nonfinite_count", 0)) for row in workers)
    return {
        "schema": "code_token_hidden_pair_progress_v1",
        "root": str(root.resolve()),
        "expected_workers": expected_workers,
        "workers_started": sum(int(row["started"]) for row in workers),
        "workers_completed": sum(int(row["completed"]) for row in workers),
        "completed_samples": completed_samples,
        "partition_samples_visible": partition_samples,
        "completion_fraction": (
            completed_samples / partition_samples if partition_samples else 0.0
        ),
        "aggregate_dense_tokens_per_second": total_rate,
        "output_bytes": total_output,
        "nonfinite_count": total_nonfinite,
        "updated_at_unix": time.time(),
        "workers": workers,
        "gt_assigned": False,
        "histograms_generated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-workers", type=int, default=8)
    parser.add_argument(
        "--write", action="store_true", help="Atomically update ROOT/progress.json."
    )
    args = parser.parse_args()
    payload = summarize(args.root, args.expected_workers)
    if args.write:
        _atomic_json(args.root / "progress.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
