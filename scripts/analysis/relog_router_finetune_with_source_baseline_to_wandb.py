#!/usr/bin/env python3
"""Replay a router-finetune run to W&B with a connected source baseline.

This is for phase3/router-only runs that start from an already code-trained
checkpoint. The source checkpoint's final probe metrics are logged at the
requested display base step, then the router-finetune metrics are shifted so
the curve continues from that point.
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
    r"\[(.*?)\]\s+(?:iteration|step)\s+(\d+)/\s*(\d+)\s+\|.*?"
    r"(?:elapsed time per iteration \(ms\):\s*([0-9.eE+-]+)|([0-9.eE+-]+)\s*s/it|([0-9.eE+-]+)\s*ms/iter)"
)
LOSS_RE = re.compile(r"\blm loss\s+([0-9.eE+-]+)")
SAVE_RE = re.compile(r"saving checkpoint(?: step| at iteration)?\s+(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-log", required=True, help="Code-trained source log containing final probes.")
    parser.add_argument("--retune-dir", required=True, help="Router-finetune run directory.")
    parser.add_argument("--retune-log", help="Router-finetune log. Inferred from RETUNE_DIR/logs if omitted.")
    parser.add_argument("--source-step", type=int, default=1800)
    parser.add_argument("--display-base-step", type=int, default=3600)
    parser.add_argument("--retune-iters", type=int, default=720)
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


def infer_log(run_dir: Path) -> Path:
    log_dir = run_dir / "logs"
    preferred = [
        log_dir / "phase3_run.log",
        log_dir / "phase3_moe_router_run.log",
        log_dir / "run.log",
    ]
    preferred.extend(sorted(log_dir.glob("phase3*.log")))
    preferred.extend(sorted(log_dir.glob("*.log")))
    for path in preferred:
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError(f"Could not infer router-finetune log under {log_dir}")


def parse_last_probe_payload(log_path: Path, max_step: int) -> dict[str, float]:
    last_by_name: dict[str, tuple[int, float, float]] = {}
    for line in read_lines(log_path):
        match = PROBE_RE.search(line)
        if not match:
            continue
        name, step, local_iteration, acc, ppl = match.groups()
        step_i = int(step)
        local_i = int(local_iteration)
        if step_i > max_step and local_i > max_step:
            continue
        previous = last_by_name.get(name)
        if previous is None or step_i >= previous[0]:
            last_by_name[name] = (step_i, float(acc), float(ppl))

    payload = {}
    for name, (_step, acc, ppl) in last_by_name.items():
        payload[f"{name}/next_token_accuracy"] = acc
        payload[f"{name}/ppl"] = ppl
    return payload


def inject_source_baseline(
    points: dict[int, dict[str, float]],
    source_log: Path,
    source_step: int,
    display_step: int,
) -> int:
    payload = parse_last_probe_payload(source_log, source_step)
    for key, value in payload.items():
        add_metric(points, display_step, key, value)
    add_metric(points, display_step, "source/pre_router_finetune_probe_marker", 1.0)
    add_metric(points, display_step, "local_iteration", source_step)
    return len(payload)


def infer_retune_step(
    raw_step: int,
    local_iteration: int,
    source_step: int,
    display_base_step: int,
    retune_iters: int,
) -> int | None:
    max_display = display_base_step + retune_iters

    if 0 < local_iteration <= retune_iters:
        return local_iteration
    if source_step < local_iteration <= source_step + retune_iters:
        return local_iteration - source_step
    if display_base_step < local_iteration <= max_display:
        return local_iteration - display_base_step

    if 0 < raw_step <= retune_iters:
        return raw_step
    if source_step < raw_step <= source_step + retune_iters:
        return raw_step - source_step
    if display_base_step < raw_step <= max_display:
        return raw_step - display_base_step

    return None


def parse_retune_log(
    points: dict[int, dict[str, float]],
    retune_log: Path,
    source_step: int,
    display_base_step: int,
    retune_iters: int,
) -> dict[str, int]:
    counts = defaultdict(int)
    for line in read_lines(retune_log):
        match = PROBE_RE.search(line)
        if match:
            name, step, local_iteration, acc, ppl = match.groups()
            raw_step = int(step)
            local_i = int(local_iteration)
            retune_step = infer_retune_step(raw_step, local_i, source_step, display_base_step, retune_iters)
            if retune_step is None or retune_step <= 0 or retune_step > retune_iters:
                counts["skipped_probe"] += 1
                continue
            display_step = display_base_step + retune_step
            add_metric(points, display_step, f"{name}/next_token_accuracy", float(acc))
            add_metric(points, display_step, f"{name}/ppl", float(ppl))
            add_metric(points, display_step, "local_iteration", source_step + retune_step)
            counts["probe"] += 1
            continue

        match = TIME_RE.search(line)
        if match:
            _timestamp, step, _total, elapsed_ms_a, elapsed_sec, elapsed_ms_b = match.groups()
            raw_step = int(step)
            retune_step = infer_retune_step(raw_step, raw_step, source_step, display_base_step, retune_iters)
            if retune_step is None or retune_step <= 0 or retune_step > retune_iters:
                continue
            elapsed_ms = None
            if elapsed_ms_a:
                elapsed_ms = float(elapsed_ms_a)
            elif elapsed_sec:
                elapsed_ms = float(elapsed_sec) * 1000.0
            elif elapsed_ms_b:
                elapsed_ms = float(elapsed_ms_b)
            display_step = display_base_step + retune_step
            add_metric(points, display_step, "timing/elapsed_ms_per_iteration", elapsed_ms)
            add_metric(points, display_step, "local_iteration", source_step + retune_step)
            counts["time"] += 1

            loss = LOSS_RE.search(line)
            if loss:
                add_metric(points, display_step, "lm loss", float(loss.group(1)))
                counts["lm loss"] += 1
            continue

        match = SAVE_RE.search(line)
        if match:
            raw_step = int(match.group(1))
            retune_step = infer_retune_step(raw_step, raw_step, source_step, display_base_step, retune_iters)
            if retune_step is None or retune_step <= 0 or retune_step > retune_iters:
                continue
            add_metric(points, display_base_step + retune_step, "checkpoint/saved", 1.0)
            counts["checkpoint"] += 1
    return dict(counts)


def should_skip_event_tag(tag: str) -> bool:
    if tag.startswith(("code_probe/", "wiki_probe/", "conversation_probe/")):
        return True
    if tag.startswith("validation/") or " validation" in tag:
        return True
    if " vs samples" in tag or tag.endswith(" vs samples"):
        return True
    return False


def parse_events(
    points: dict[int, dict[str, float]],
    retune_dir: Path,
    source_step: int,
    display_base_step: int,
    retune_iters: int,
) -> dict[str, int]:
    counts = defaultdict(int)
    for event_file in sorted(retune_dir.rglob("events.out.tfevents*")):
        if "/wandb/" in event_file.as_posix():
            continue
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            if should_skip_event_tag(tag):
                continue
            for event in accumulator.Scalars(tag):
                retune_step = infer_retune_step(
                    int(event.step),
                    int(event.step),
                    source_step,
                    display_base_step,
                    retune_iters,
                )
                if retune_step is None or retune_step <= 0 or retune_step > retune_iters:
                    continue
                add_metric(points, display_base_step + retune_step, tag, float(event.value))
                counts[tag] += 1
    return dict(counts)


def load_metadata(retune_dir: Path) -> dict:
    for path in [
        retune_dir / "logs" / "phase3_run_metadata.json",
        retune_dir / "logs" / "run_metadata.json",
        retune_dir / "PHASE3_SOURCE.txt",
    ]:
        if not path.exists():
            continue
        if path.suffix == ".json":
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
    return {}


def main() -> None:
    args = parse_args()
    retune_dir = Path(args.retune_dir)
    retune_log = Path(args.retune_log) if args.retune_log else infer_log(retune_dir)
    points: dict[int, dict[str, float]] = defaultdict(dict)

    source_count = inject_source_baseline(
        points,
        Path(args.source_log),
        args.source_step,
        args.display_base_step,
    )
    retune_counts = parse_retune_log(
        points,
        retune_log,
        args.source_step,
        args.display_base_step,
        args.retune_iters,
    )
    event_counts = {}
    if not args.probe_only:
        try:
            event_counts = parse_events(
                points,
                retune_dir,
                args.source_step,
                args.display_base_step,
                args.retune_iters,
            )
        except Exception as exc:
            if not args.skip_events_on_error:
                raise
            print(f"[WARN] event parsing failed; continuing without events: {type(exc).__name__}: {exc}")

    steps = sorted(points)
    if not steps:
        raise SystemExit("No metrics parsed.")

    print(f"source_count={source_count}")
    print(f"retune_counts={retune_counts}")
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
            "relog_source": "router_finetune_with_source_baseline",
            "source_log": str(args.source_log),
            "retune_dir": str(retune_dir),
            "retune_log": str(retune_log),
            "source_step": args.source_step,
            "display_base_step": args.display_base_step,
            "retune_iters": args.retune_iters,
            **load_metadata(retune_dir),
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
