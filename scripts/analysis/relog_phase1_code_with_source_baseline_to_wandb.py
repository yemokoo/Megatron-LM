#!/usr/bin/env python3
"""Replay a phase1 code-training log to W&B with a connected source baseline.

Use this for runs that start from a completed wiki checkpoint and expand experts
before code training. The source checkpoint's final probe metrics are logged at
the base display step, while the code run's local_iteration=0 probe is skipped so
it cannot overwrite the pre-expansion source value at the same step.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PROBE_RE = re.compile(
    r"probe\s+(\S+)\s+at iteration\s+(\d+)\s+\|\s+local_iteration:\s+(\d+)\s+\|\s+"
    r"next_token_acc:\s+([0-9.eE+-]+)\s+\|\s+ppl:\s+([0-9.eE+-]+)"
)
TIME_RE = re.compile(
    r"\[(.*?)\]\s+iteration\s+(\d+)/\s*(\d+)\s+\|\s+elapsed time per iteration \(ms\):\s*([0-9.eE+-]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-log", required=True, help="Source wiki run.log with pre-expansion final probes.")
    parser.add_argument("--code-dir", required=True, help="Completed code run directory.")
    parser.add_argument("--code-log", help="Code run log. Defaults to CODE_DIR/logs/run.log.")
    parser.add_argument("--source-step", type=int, default=1800)
    parser.add_argument("--display-base-step", type=int, default=1800)
    parser.add_argument("--train-iters", type=int, default=1800)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "flame-continual-top2-qv-lora"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--skip-events-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def add_metric(points: dict[int, dict[str, float]], step: int, key: str, value: float | int | None) -> None:
    if value is None:
        return
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return
    points[int(step)][key] = value


def parse_last_probe_payload(log_path: Path, source_step: int) -> dict[str, float]:
    last_by_name: dict[str, tuple[int, float, float]] = {}
    for line in read_lines(log_path):
        match = PROBE_RE.search(line)
        if not match:
            continue
        name, step, local_iteration, acc, ppl = match.groups()
        step_i = int(step)
        local_i = int(local_iteration)
        if step_i > source_step and local_i > source_step:
            continue
        previous = last_by_name.get(name)
        if previous is None or step_i >= previous[0]:
            last_by_name[name] = (step_i, float(acc), float(ppl))

    payload = {}
    for name, (_step, acc, ppl) in last_by_name.items():
        payload[f"{name}/next_token_accuracy"] = acc
        payload[f"{name}/ppl"] = ppl
    return payload


def inject_source_baseline(points: dict[int, dict[str, float]], source_log: Path, source_step: int, display_step: int) -> int:
    payload = parse_last_probe_payload(source_log, source_step)
    for key, value in payload.items():
        add_metric(points, display_step, key, value)
    add_metric(points, display_step, "source/pre_expansion_probe_marker", 1.0)
    add_metric(points, display_step, "local_iteration", source_step)
    return len(payload)


def parse_code_log(points: dict[int, dict[str, float]], code_log: Path, base_step: int, train_iters: int) -> dict[str, int]:
    counts = defaultdict(int)
    max_step = base_step + train_iters
    for line in read_lines(code_log):
        match = PROBE_RE.search(line)
        if match:
            name, step, local_iteration, acc, ppl = match.groups()
            step_i = int(step)
            local_i = int(local_iteration)
            # Skip local 0 so pre-expansion source probes remain visible at base_step.
            if local_i <= 0:
                counts["skipped_local0_probe"] += 1
                continue
            display_step = step_i if base_step < step_i <= max_step else base_step + local_i
            if display_step <= base_step or display_step > max_step:
                continue
            add_metric(points, display_step, f"{name}/next_token_accuracy", float(acc))
            add_metric(points, display_step, f"{name}/ppl", float(ppl))
            add_metric(points, display_step, "local_iteration", local_i)
            counts["probe"] += 1
            continue

        match = TIME_RE.search(line)
        if match:
            _timestamp, local_iteration, _total, elapsed_ms = match.groups()
            local_i = int(local_iteration)
            display_step = base_step + local_i
            if display_step <= base_step or display_step > max_step:
                continue
            add_metric(points, display_step, "timing/elapsed_ms_per_iteration", float(elapsed_ms))
            add_metric(points, display_step, "local_iteration", local_i)
            counts["time"] += 1
    return dict(counts)


def should_skip_event_tag(tag: str) -> bool:
    if tag.startswith("code_probe/") or tag.startswith("wiki_probe/") or tag.startswith("conversation_probe/"):
        return True
    if tag.startswith("validation/") or " validation" in tag:
        return True
    if " vs samples" in tag or tag.endswith(" vs samples"):
        return True
    return False


def event_display_step(local_or_display_step: int, base_step: int, train_iters: int) -> int | None:
    max_step = base_step + train_iters
    if 0 < local_or_display_step <= train_iters:
        return base_step + local_or_display_step
    if base_step < local_or_display_step <= max_step:
        return local_or_display_step
    return None


def parse_events(points: dict[int, dict[str, float]], code_dir: Path, base_step: int, train_iters: int) -> dict[str, int]:
    counts = defaultdict(int)
    for event_file in sorted(code_dir.rglob("events.out.tfevents*")):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            if should_skip_event_tag(tag):
                continue
            for event in accumulator.Scalars(tag):
                display_step = event_display_step(int(event.step), base_step, train_iters)
                if display_step is None or display_step <= base_step:
                    continue
                add_metric(points, display_step, tag, float(event.value))
                counts[tag] += 1
    return dict(counts)


def load_metadata(code_dir: Path) -> dict:
    metadata_path = code_dir / "logs" / "run_metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    args = parse_args()
    code_dir = Path(args.code_dir)
    code_log = Path(args.code_log) if args.code_log else code_dir / "logs" / "run.log"
    points: dict[int, dict[str, float]] = defaultdict(dict)

    source_count = inject_source_baseline(points, Path(args.source_log), args.source_step, args.display_base_step)
    code_counts = parse_code_log(points, code_log, args.display_base_step, args.train_iters)
    event_counts = {}
    if not args.probe_only:
        try:
            event_counts = parse_events(points, code_dir, args.display_base_step, args.train_iters)
        except Exception as exc:
            if not args.skip_events_on_error:
                raise
            print(f"[WARN] event parsing failed; continuing without events: {type(exc).__name__}: {exc}")

    steps = sorted(points)
    if not steps:
        raise SystemExit("No metrics parsed.")

    print(f"source_count={source_count}")
    print(f"code_counts={code_counts}")
    print(f"event_tag_count={len(event_counts)}")
    print(f"steps={len(steps)} first={steps[0]} last={steps[-1]}")
    for step in [steps[0], *(s for s in steps if s != steps[0])][-4:]:
        print(f"STEP {step}: {sorted(points[step])}")

    if args.dry_run:
        return

    init_kwargs = {
        "project": args.project,
        "id": args.run_id,
        "name": args.run_name,
        "resume": "never",
        "config": {
            "relog_source": "phase1_code_with_source_baseline",
            "source_log": str(args.source_log),
            "code_dir": str(code_dir),
            "code_log": str(code_log),
            "source_step": args.source_step,
            "display_base_step": args.display_base_step,
            "train_iters": args.train_iters,
            "skip_code_local0_probe": True,
            **load_metadata(code_dir),
        },
    }
    if args.entity:
        init_kwargs["entity"] = args.entity

    wandb.init(**init_kwargs)
    for step in steps:
        wandb.log(points[step], step=step)
    wandb.finish()


if __name__ == "__main__":
    main()
