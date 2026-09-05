#!/usr/bin/env python3
"""Follow the six-baseline chain log and print a compact line with an ETA.

Read-only: this never touches the run.  Useful even while a stage that was
launched with a finer --log-interval is still going, because it thins the
stream itself.

    python3 scripts/experiment/a100/baselines6/watch_chain.py            # follow
    python3 scripts/experiment/a100/baselines6/watch_chain.py --once     # one line
    python3 scripts/experiment/a100/baselines6/watch_chain.py --every 50
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
import time
from pathlib import Path

DEFAULT_ROOT = "/data2/seonghyeonnoh/LLM-continual-learning-runs/baselines6_wiki_code_conversation_20260816"
TOTAL_STAGES = 23

ITER_RE = re.compile(
    r"iteration\s+(\d+)/\s*(\d+).*?"
    r"elapsed time per iteration \(ms\):\s*([\d.]+).*?"
    r"lm loss:\s*([\d.E+-]+)"
)
START_RE = re.compile(r"\[START\] method=(\S+) task=(\S+)")
EVENT_RE = re.compile(r"\[(DONE|FAIL|SKIP|ALL DONE)\]")


def completed_stages(root: Path) -> int:
    """A stage is complete once it has written its final audit."""
    return len(list(root.glob("*/*/continual_audit/final.json")))


def human(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:
        return "?"
    seconds = int(seconds)
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def render(
    root: Path, stage: str, iteration: int, total: int, ms: float, loss: str,
    total_stages: int = TOTAL_STAGES,
) -> str:
    per_iter = ms / 1000.0
    stage_left = (total - iteration) * per_iter
    done = completed_stages(root)
    # Stages not yet started are estimated at the current stage's pace.
    other_left = max(total_stages - done - 1, 0) * total * per_iter
    chain_left = stage_left + other_left
    finish = dt.datetime.now() + dt.timedelta(seconds=chain_left)
    pct = 100.0 * iteration / total if total else 0.0
    return (
        f"[{dt.datetime.now():%H:%M:%S}] stage {done + 1}/{total_stages} {stage} | "
        f"{iteration}/{total} ({pct:.1f}%) | loss {loss} | {per_iter:.2f}s/it | "
        f"stage ETA {human(stage_left)} | chain ETA {human(chain_left)} "
        f"(~{finish:%m-%d %H:%M})"
    )


def scan_tail(path: Path, limit: int = 4_000_000):
    """Return (stage, last iteration match) from the end of the log."""
    size = path.stat().st_size
    with path.open("r", errors="replace") as handle:
        handle.seek(max(0, size - limit))
        lines = handle.readlines()
    stage, last = "?", None
    for line in lines:
        found = START_RE.search(line)
        if found:
            stage = f"{found.group(1)}/{found.group(2)}"
        match = ITER_RE.search(line)
        if match:
            last = match
    return stage, last


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=os.environ.get("BASELINES6_OUTPUT_ROOT", DEFAULT_ROOT))
    parser.add_argument("--every", type=int, default=20, help="print one line per N iterations")
    parser.add_argument("--once", action="store_true", help="print current status and exit")
    parser.add_argument(
        "--total", type=int, default=TOTAL_STAGES,
        help="stages to budget the chain ETA over (15 for the main sequence alone)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    log = root / "chain.log"
    if not log.exists():
        print(f"no chain.log under {root}", file=sys.stderr)
        return 1

    stage, last = scan_tail(log)
    if last:
        print(render(root, stage, int(last.group(1)), int(last.group(2)), float(last.group(3)), last.group(4), args.total))
    elif args.once:
        print("no iteration line found yet")
    if args.once:
        return 0

    with log.open("r", errors="replace") as handle:
        handle.seek(0, os.SEEK_END)
        while True:
            line = handle.readline()
            if not line:
                if not (root / "chain.pid").exists():
                    return 0
                time.sleep(2)
                continue
            found = START_RE.search(line)
            if found:
                stage = f"{found.group(1)}/{found.group(2)}"
            if EVENT_RE.search(line):
                print(line.rstrip())
                continue
            match = ITER_RE.search(line)
            if not match:
                continue
            iteration = int(match.group(1))
            if args.every > 1 and iteration % args.every:
                continue
            print(
                render(
                    root, stage, iteration, int(match.group(2)),
                    float(match.group(3)), match.group(4), args.total,
                ),
                flush=True,
            )


if __name__ == "__main__":
    raise SystemExit(main())
