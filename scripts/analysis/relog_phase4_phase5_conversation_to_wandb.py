#!/usr/bin/env python3
"""Replay phase4 conversation + phase5 router-finetune metrics to one W&B run.

The phase4 training run starts from a router-retuned wiki+code checkpoint whose
logical display step is 5400. The phase5 router-only run starts from the phase4
checkpoint whose local checkpoint step is 1800 but logical display step is 7200.

This script keeps those graphs continuous:

  phase4 event local step 0..1800       -> W&B step 5400..7200
  phase5 event local step 1800..target  -> W&B step 7200..target+5400

Probe lines in run.log already carry the intended logical iteration, so those
are logged at the iteration printed in the log.
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
SUMMARY_STEP_RE = re.compile(r"step\s+(\d+)/\s*(\d+)")
SUMMARY_MS_RE = re.compile(r"([0-9.]+)\s*ms/iter")
SUMMARY_SEC_RE = re.compile(r"([0-9.]+)\s*s/it")
SUMMARY_LOSS_RE = re.compile(r"\blm loss\s+([0-9.eE+-]+)")
VALID_RE = re.compile(r"validation step\s+(\d+)\s+\|\s+lm loss\s+([0-9.eE+-]+)")
SAVE_RE = re.compile(r"saving checkpoint step\s+(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase4-dir", required=True)
    parser.add_argument("--phase5-dir", required=True)
    parser.add_argument("--phase4-log")
    parser.add_argument("--phase5-log")
    parser.add_argument("--phase4-source-log", help="Log containing the previous router-retuned final probe values.")
    parser.add_argument("--phase4-source-step", type=int, default=5400)
    parser.add_argument(
        "--phase5-source-log",
        help="Log containing the phase4 final probe values. Defaults to --phase4-log when --connect-baselines is set.",
    )
    parser.add_argument("--phase5-source-step", type=int, default=7200)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "flame-continual-top2-qv-lora"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--phase4-base-step", type=int, default=5400)
    parser.add_argument("--phase4-iters", type=int, default=1800)
    parser.add_argument("--phase5-base-step", type=int, default=7200)
    parser.add_argument("--phase5-local-start", type=int, default=1800)
    parser.add_argument("--phase5-local-end", type=int)
    parser.add_argument(
        "--connect-baselines",
        action="store_true",
        help="Inject source final probe values at phase4/phase5 base display steps so W&B curves connect continuously.",
    )
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--skip-events-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_tracker(run_dir: Path) -> int | None:
    tracker = run_dir / "latest_checkpointed_iteration.txt"
    if not tracker.exists():
        return None
    try:
        return int(tracker.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def infer_phase4_log(run_dir: Path) -> Path | None:
    log_dir = run_dir / "logs"
    preferred = [
        log_dir / "phase4_conversation_ffn_only.log",
        log_dir / "run.log",
    ]
    preferred.extend(sorted(log_dir.glob("phase4_conversation*.log")))
    preferred.extend(sorted(log_dir.glob("*.log")))
    for path in preferred:
        if path.exists() and path.is_file():
            return path
    return None


def infer_phase5_log(run_dir: Path) -> Path | None:
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
    return None


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def add_metric(points: dict[int, dict[str, float]], step: int, key: str, value: float | int | None) -> None:
    if value is None:
        return
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return
    points[int(step)][key] = value


def parse_probe_log(points, log_path: Path, min_step: int, max_step: int, phase_name: str) -> dict[str, int]:
    counts = defaultdict(int)
    for line in read_lines(log_path):
        match = PROBE_RE.search(line)
        if not match:
            continue
        name, step, local_iteration, acc, ppl = match.groups()
        step_i = int(step)
        if step_i < min_step or step_i > max_step:
            continue
        add_metric(points, step_i, f"{name}/next_token_accuracy", float(acc))
        add_metric(points, step_i, f"{name}/ppl", float(ppl))
        add_metric(points, step_i, "local_iteration", int(local_iteration))
        add_metric(points, step_i, f"{phase_name}/active", 1.0)
        counts[name] += 1
    return dict(counts)


def parse_last_probe_payload(log_path: Path, max_step: int) -> dict[str, float]:
    last_by_name: dict[str, tuple[int, float, float]] = {}
    for line in read_lines(log_path):
        match = PROBE_RE.search(line)
        if not match:
            continue
        name, step, _local_iteration, acc, ppl = match.groups()
        step_i = int(step)
        if step_i > max_step:
            continue
        previous = last_by_name.get(name)
        if previous is None or step_i >= previous[0]:
            last_by_name[name] = (step_i, float(acc), float(ppl))

    payload = {}
    for name, (_step, acc, ppl) in last_by_name.items():
        payload[f"{name}/next_token_accuracy"] = acc
        payload[f"{name}/ppl"] = ppl
    return payload


def inject_probe_baseline(
    points: dict[int, dict[str, float]],
    log_path: Path | None,
    source_step: int,
    display_step: int,
    marker_name: str,
) -> dict[str, int | str]:
    if log_path is None:
        return {"count": 0, "source_log": ""}
    payload = parse_last_probe_payload(log_path, source_step)
    for key, value in payload.items():
        add_metric(points, display_step, key, value)
    add_metric(points, display_step, marker_name, 1.0)
    return {"count": len(payload), "source_log": str(log_path)}


def parse_summary_log(points, log_path: Path, local_min: int, local_max: int, step_offset: int, phase_name: str) -> dict[str, int]:
    counts = defaultdict(int)
    for line in read_lines(log_path):
        match = SUMMARY_STEP_RE.search(line)
        if match:
            local_step = int(match.group(1))
            if local_step < local_min or local_step > local_max:
                continue
            step = local_step + step_offset
            add_metric(points, step, "local_iteration", local_step)
            add_metric(points, step, f"{phase_name}/active", 1.0)
            ms = SUMMARY_MS_RE.search(line)
            sec = SUMMARY_SEC_RE.search(line)
            loss = SUMMARY_LOSS_RE.search(line)
            if ms:
                add_metric(points, step, "timing/elapsed_ms_per_iteration", float(ms.group(1)))
                counts["timing"] += 1
            if sec:
                add_metric(points, step, "timing/elapsed_ms_per_iteration", float(sec.group(1)) * 1000.0)
                counts["timing"] += 1
            if loss:
                add_metric(points, step, "lm loss", float(loss.group(1)))
                counts["lm loss"] += 1
            continue

        match = VALID_RE.search(line)
        if match:
            local_step = int(match.group(1))
            if local_step < local_min or local_step > local_max:
                continue
            add_metric(points, local_step + step_offset, "validation/lm_loss", float(match.group(2)))
            counts["validation"] += 1
            continue

        match = SAVE_RE.search(line)
        if match:
            local_step = int(match.group(1))
            if local_step < local_min or local_step > local_max:
                continue
            add_metric(points, local_step + step_offset, "checkpoint/saved", 1.0)
            counts["checkpoint"] += 1
    return dict(counts)


def should_skip_event_tag(tag: str) -> bool:
    if " vs samples" in tag or tag.endswith(" vs samples"):
        return True
    if tag.startswith(("code_probe/", "wiki_probe/", "conversation_probe/")):
        return True
    return False


def read_event_scalars(event_file: Path):
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    for tag in accumulator.Tags().get("scalars", []):
        if should_skip_event_tag(tag):
            continue
        for event in accumulator.Scalars(tag):
            yield tag, int(event.step), float(event.value)


def iter_event_files(run_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(run_dir.rglob("events.out.tfevents*"))
        if "/wandb/" not in path.as_posix()
    ]


def parse_events(
    points,
    run_dir: Path,
    local_min: int,
    local_max: int,
    step_offset: int,
    phase_name: str,
    skip_on_error: bool,
) -> dict[str, int]:
    counts = defaultdict(int)
    for event_file in iter_event_files(run_dir):
        try:
            scalars = list(read_event_scalars(event_file))
        except Exception as exc:
            if skip_on_error:
                print(f"[WARN] skipped event file after parse error: {event_file} ({type(exc).__name__}: {exc})")
                continue
            raise
        for tag, local_step, value in scalars:
            if local_step < local_min or local_step > local_max:
                continue
            step = local_step + step_offset
            add_metric(points, step, tag, value)
            add_metric(points, step, "local_iteration", local_step)
            add_metric(points, step, f"{phase_name}/active", 1.0)
            counts[tag] += 1
    return dict(counts)


def load_metadata(run_dir: Path) -> dict:
    metadata = {}
    for path in [
        run_dir / "logs" / "run_metadata.json",
        run_dir / "logs" / "phase3_run_metadata.json",
        run_dir / "logs" / "phase3_moe_router_metadata.json",
    ]:
        if not path.exists():
            continue
        try:
            metadata[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            metadata[path.stem] = str(path)
    return metadata


def main() -> None:
    args = parse_args()
    phase4_dir = Path(args.phase4_dir)
    phase5_dir = Path(args.phase5_dir)
    phase4_log = Path(args.phase4_log) if args.phase4_log else infer_phase4_log(phase4_dir)
    phase5_log = Path(args.phase5_log) if args.phase5_log else infer_phase5_log(phase5_dir)
    if phase4_log is None or not phase4_log.exists():
        raise SystemExit(f"Could not find phase4 log under {phase4_dir}")
    if phase5_log is None or not phase5_log.exists():
        raise SystemExit(f"Could not find phase5 log under {phase5_dir}")
    phase4_source_log = Path(args.phase4_source_log) if args.phase4_source_log else None
    phase5_source_log = Path(args.phase5_source_log) if args.phase5_source_log else phase4_log
    if args.connect_baselines:
        if phase4_source_log is None or not phase4_source_log.exists():
            raise SystemExit(f"--connect-baselines requires an existing --phase4-source-log: {phase4_source_log}")
        if phase5_source_log is None or not phase5_source_log.exists():
            raise SystemExit(f"--connect-baselines requires an existing --phase5-source-log or phase4 log: {phase5_source_log}")

    phase4_local_end = args.phase4_iters
    phase4_max_step = args.phase4_base_step + args.phase4_iters
    phase5_local_end = args.phase5_local_end or read_tracker(phase5_dir)
    if phase5_local_end is None:
        raise SystemExit(f"Could not infer phase5 local end from tracker: {phase5_dir}")
    phase5_step_offset = args.phase5_base_step - args.phase5_local_start
    phase5_max_step = phase5_local_end + phase5_step_offset

    points = defaultdict(dict)
    points[args.phase4_base_step]["phase4/source_marker"] = 1.0
    points[args.phase5_base_step]["phase5/source_marker"] = 1.0
    phase4_baseline_counts = {}
    phase5_baseline_counts = {}
    if args.connect_baselines:
        phase4_baseline_counts = inject_probe_baseline(
            points,
            phase4_source_log,
            args.phase4_source_step,
            args.phase4_base_step,
            "phase4/connected_source_probe_marker",
        )
        phase5_baseline_counts = inject_probe_baseline(
            points,
            phase5_source_log,
            args.phase5_source_step,
            args.phase5_base_step,
            "phase5/connected_source_probe_marker",
        )

    phase4_probe_counts = parse_probe_log(points, phase4_log, args.phase4_base_step, phase4_max_step, "phase4")
    phase5_probe_counts = parse_probe_log(points, phase5_log, args.phase5_base_step, phase5_max_step, "phase5")
    phase4_summary_counts = parse_summary_log(
        points, phase4_log, 0, phase4_local_end, args.phase4_base_step, "phase4"
    )
    phase5_summary_counts = parse_summary_log(
        points, phase5_log, args.phase5_local_start, phase5_local_end, phase5_step_offset, "phase5"
    )

    phase4_event_counts = {}
    phase5_event_counts = {}
    if not args.probe_only:
        phase4_event_counts = parse_events(
            points,
            phase4_dir,
            0,
            phase4_local_end,
            args.phase4_base_step,
            "phase4",
            args.skip_events_on_error,
        )
        phase5_event_counts = parse_events(
            points,
            phase5_dir,
            args.phase5_local_start,
            phase5_local_end,
            phase5_step_offset,
            "phase5",
            args.skip_events_on_error,
        )

    steps = sorted(points)
    if not steps:
        raise SystemExit("No metrics were parsed.")

    print(f"phase4_dir={phase4_dir}")
    print(f"phase5_dir={phase5_dir}")
    print(f"phase4_log={phase4_log}")
    print(f"phase5_log={phase5_log}")
    print(f"phase4 display: {args.phase4_base_step}->{phase4_max_step}")
    print(f"phase5 display: {args.phase5_base_step}->{phase5_max_step} (local {args.phase5_local_start}->{phase5_local_end})")
    print(f"phase4 probe counts: {phase4_probe_counts}")
    print(f"phase5 probe counts: {phase5_probe_counts}")
    print(f"phase4 connected baseline: {phase4_baseline_counts}")
    print(f"phase5 connected baseline: {phase5_baseline_counts}")
    print(f"phase4 summary counts: {phase4_summary_counts}")
    print(f"phase5 summary counts: {phase5_summary_counts}")
    print(f"phase4 event tag count: {len(phase4_event_counts)}")
    print(f"phase5 event tag count: {len(phase5_event_counts)}")
    print(f"steps: {len(steps)} first={steps[0]} last={steps[-1]}")
    for step in [steps[0], args.phase4_base_step, args.phase5_base_step, steps[-1]]:
        if step in points:
            print(f"STEP {step}: {sorted(points[step])[:24]}")

    if args.dry_run:
        return

    init_kwargs = {
        "project": args.project,
        "id": args.run_id,
        "name": args.run_name,
        "resume": "allow",
        "config": {
            "relog_source": "phase4_phase5_conversation",
            "phase4_dir": str(phase4_dir),
            "phase5_dir": str(phase5_dir),
            "phase4_log": str(phase4_log),
            "phase5_log": str(phase5_log),
            "phase4_source_log": str(phase4_source_log) if phase4_source_log else None,
            "phase4_source_step": args.phase4_source_step,
            "phase5_source_log": str(phase5_source_log) if phase5_source_log else None,
            "phase5_source_step": args.phase5_source_step,
            "connect_baselines": args.connect_baselines,
            "phase4_base_step": args.phase4_base_step,
            "phase4_iters": args.phase4_iters,
            "phase5_base_step": args.phase5_base_step,
            "phase5_local_start": args.phase5_local_start,
            "phase5_local_end": phase5_local_end,
            "phase4_metadata": load_metadata(phase4_dir),
            "phase5_metadata": load_metadata(phase5_dir),
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
