#!/usr/bin/env python3
"""Print a compact snapshot of paired hidden-extraction progress."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(str(args.root / "rank_*" / "progress.json"))):
        with open(path, encoding="utf-8") as handle:
            row = json.load(handle)
        rank = os.path.basename(os.path.dirname(path))
        done = int(row["completed_samples"])
        total = int(row["partition_samples"])
        speed = float(row["tokens_per_second_this_process"])
        eta = float(row["eta_seconds"])
        nonfinite = int(row["nonfinite_count"])
        output = int(row["output_bytes"])
        rows.append((rank, done, total, speed, eta, nonfinite, output))
        print(
            f"{rank} | {done:>7,}/{total:,} ({done / total:6.2%})"
            f" | {speed:>9,.0f} tok/s | ETA {eta / 60:5.1f}m"
            f" | NaN/Inf {nonfinite} | {output / 2**30:5.1f} GiB"
        )

    if not rows:
        print("No progress.json files found yet.")
        return
    done = sum(row[1] for row in rows)
    total = sum(row[2] for row in rows)
    speed = sum(row[3] for row in rows)
    output = sum(row[6] for row in rows)
    print("-" * 100)
    print(
        f"TOTAL    | {done:>7,}/{total:,} ({done / total:6.2%})"
        f" | {speed:>9,.0f} tok/s | max ETA {max(row[4] for row in rows) / 60:5.1f}m"
        f" | NaN/Inf {sum(row[5] for row in rows)} | {output / 2**30:5.1f} GiB"
    )


if __name__ == "__main__":
    main()
