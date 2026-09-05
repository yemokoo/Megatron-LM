#!/usr/bin/env python3
"""Relay legacy per-shard tqdm state to one human-readable stdout stream."""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path


PROGRESS = re.compile(
    r"(?P<pct>\d{1,3})%\|[^\r\n]*?\|\s*(?P<done>\d+)/(?P<total>\d+)")


def latest(path: Path):
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 256 * 1024))
            text = handle.read().decode("utf-8", errors="replace").replace(
                "\r", "\n")
    except OSError:
        return None
    matches = list(PROGRESS.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    return int(match["pct"]), int(match["done"]), int(match["total"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    summary = run_dir / "sparse15_summary.json"
    evaluation = run_dir / "evaluation"
    print(f"[LEGACY PROGRESS WATCH] run={run_dir}", flush=True)
    while not summary.is_file():
        now = time.time()
        rows = []
        for path in evaluation.glob("order*/*.log"):
            try:
                # Ignore old completed logs; active shard files keep changing.
                if now - path.stat().st_mtime > 10 * 60:
                    continue
            except OSError:
                continue
            progress = latest(path)
            if progress is not None:
                rows.append((str(path.relative_to(evaluation)), progress))
        stamp = time.strftime("%F %T")
        if not rows:
            print(f"[LEGACY STATUS] {stamp} loading/merging", flush=True)
        for label, (percent, done, total) in sorted(rows):
            print(
                f"[LEGACY PROGRESS] {stamp} {label} "
                f"{percent}%({done}/{total})",
                flush=True,
            )
        time.sleep(args.interval)
    print(f"[LEGACY PROGRESS COMPLETE] {time.strftime('%F %T')}", flush=True)


if __name__ == "__main__":
    main()
