#!/usr/bin/env python3
"""Record pristine and user-worktree provenance without modifying either."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


def run(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def inspect(label: str, path: Path, pristine: bool) -> dict:
    row = {"label": label, "path": str(path), "pristine_expected": pristine}
    try:
        row.update(
            {
                "head": run(path, "rev-parse", "HEAD"),
                "branch": run(path, "branch", "--show-current") or "(detached)",
                "remotes": run(path, "remote", "-v").splitlines(),
                "status_porcelain": run(path, "status", "--porcelain").splitlines(),
                "diff_stat": run(path, "diff", "--stat").splitlines(),
            }
        )
        row["dirty"] = bool(row["status_porcelain"])
    except subprocess.CalledProcessError as exc:
        row["error"] = exc.output
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "manifests" / "repositories.json"
    )
    args = parser.parse_args()
    rows = [
        inspect("local LLM-continual-learning", WORKSPACE / "LLM-continual-learning", False),
        inspect("local TRACE", WORKSPACE / "TRACE", False),
        inspect("pristine SLoRA", ROOT / "upstream" / "SLoRA", True),
        inspect("pristine O-LoRA", ROOT / "upstream" / "O-LoRA", True),
        inspect("pristine TRACE", ROOT / "upstream" / "TRACE", True),
    ]
    failures = [row["label"] for row in rows if row.get("pristine_expected") and row.get("dirty")]
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repositories": rows,
        "pristine_failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
