#!/usr/bin/env python3
"""Merge the current sparse-15 driver and per-job logs into one live stream."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def current_run_start(data: bytes) -> int:
    marker = b"\n[START] "
    pos = data.rfind(marker)
    return pos + 1 if pos >= 0 else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    offsets: dict[Path, int] = {}
    partial: dict[Path, bytes] = {}

    with args.output.open("ab", buffering=0) as output:
        output.write(
            f"\n===== merged logger started {time.strftime('%F %T')} =====\n".encode()
        )
        while True:
            paths = [args.driver] + sorted(args.jobs.glob("*.log"))
            for path in paths:
                if path == args.output or not path.is_file():
                    continue
                size = path.stat().st_size
                if path not in offsets:
                    data = path.read_bytes()
                    offsets[path] = (
                        0 if path == args.driver else current_run_start(data)
                    )
                if size < offsets[path]:
                    offsets[path] = 0
                    partial[path] = b""
                if size == offsets[path]:
                    continue
                with path.open("rb") as source:
                    source.seek(offsets[path])
                    chunk = source.read()
                    offsets[path] = source.tell()
                data = partial.get(path, b"") + chunk.replace(b"\r", b"\n")
                lines = data.split(b"\n")
                partial[path] = lines.pop()
                prefix = f"[{path.name}] ".encode()
                for line in lines:
                    if line:
                        output.write(prefix + line + b"\n")
            time.sleep(1)


if __name__ == "__main__":
    main()
