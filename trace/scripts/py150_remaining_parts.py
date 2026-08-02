#!/usr/bin/env python3
"""Prepare or merge four contiguous shards for the unfinished Py150 suffix."""

import argparse
import json
import math
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "merge"])
    parser.add_argument("--global-file", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--remaining", type=Path, required=True)
    args = parser.parse_args()

    if args.mode == "prepare":
        prefix = (
            args.global_file.read_bytes().splitlines(keepends=True)
            if args.global_file.exists() else []
        )
        questions = json.loads(args.source.read_text(encoding="utf-8"))
        remaining = questions[len(prefix):]
        args.remaining.write_text(
            json.dumps(remaining, ensure_ascii=False), encoding="utf-8")
        for shard in range(4):
            (args.task_dir / f"infer.shard{shard}.jsonl").write_bytes(b"")
        print(
            f"prefix={len(prefix)} remaining={len(remaining)} "
            f"per_shard~={(len(remaining) + 3) // 4}",
            flush=True,
        )
        return

    remaining = json.loads(args.remaining.read_text(encoding="utf-8"))
    parts = [args.task_dir / f"infer.shard{i}.jsonl" for i in range(4)]
    chunk = math.ceil(len(remaining) / 4)
    for shard, part in enumerate(parts):
        lines = part.read_text(encoding="utf-8").splitlines()
        expected = max(0, min(chunk, len(remaining) - shard * chunk))
        if len(lines) != expected:
            raise SystemExit(
                f"shard{shard} incomplete: {len(lines)} != {expected}")
        for line in lines:
            json.loads(line)
    temporary = args.global_file.with_suffix(".jsonl.merging")
    with temporary.open("wb") as output:
        if args.global_file.exists():
            output.write(args.global_file.read_bytes())
        for part in parts:
            output.write(part.read_bytes())
    os.replace(temporary, args.global_file)
    print(
        f"[PY150 MERGED] "
        f"{len(args.global_file.read_text(encoding='utf-8').splitlines())} records",
        flush=True,
    )


if __name__ == "__main__":
    main()
