#!/usr/bin/env python3
"""Re-log a RouterKD run to W&B with base-model probe metrics at step 1800.

Probe metrics are parsed from run.log because those lines already contain the
intended global step. Training losses and auxiliary losses are parsed from
TensorBoard event files, whose steps are local training iterations, and shifted
by --step-offset exactly once.
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
KD_RE = re.compile(
    r"teacher-student router memory KD at iteration\s+(\d+)\s+\|\s+kl:\s+([0-9.eE+-]+)"
    r"\s+\|\s+batches:\s+(\d+)"
)
TIME_RE = re.compile(
    r"\[(.*?)\]\s+iteration\s+(\d+)/\s*(\d+)\s+\|\s+elapsed time per iteration \(ms\):\s*([0-9.eE+-]+)"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-log", required=True, help="Stage-1 wiki run.log with base step probe metrics.")
    parser.add_argument("--code-log", required=True, help="RouterKD code run.log.")
    parser.add_argument("--run-dir", required=True, help="RouterKD run directory containing TensorBoard events.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "flame-continual-top2-qv-lora"))
    parser.add_argument("--step-offset", type=int, default=1800)
    parser.add_argument("--train-iters", type=int, default=1800)
    parser.add_argument("--kd-coeff", type=float, default=10.0)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_lines(path: Path):
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def add_metric(points, step, key, value):
    if value is None:
        return
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return
    points[int(step)][key] = float(value)


def parse_base_probes(points, wiki_log: Path, step_offset: int):
    found = 0
    for line in read_lines(wiki_log):
        match = PROBE_RE.search(line)
        if not match:
            continue
        name, step, local_iteration, acc, ppl = match.groups()
        step = int(step)
        local_iteration = int(local_iteration)
        if step != step_offset and local_iteration != step_offset:
            continue
        add_metric(points, step_offset, f"{name}/next_token_accuracy", float(acc))
        add_metric(points, step_offset, f"{name}/ppl", float(ppl))
        found += 1
    points[step_offset]["source/base_step_marker"] = 1.0
    return found


def parse_code_log(points, code_log: Path, step_offset: int, train_iters: int):
    max_step = step_offset + train_iters
    counts = defaultdict(int)
    for line in read_lines(code_log):
        match = PROBE_RE.search(line)
        if match:
            name, step, local_iteration, acc, ppl = match.groups()
            step = int(step)
            if step <= step_offset or step > max_step:
                continue
            add_metric(points, step, f"{name}/next_token_accuracy", float(acc))
            add_metric(points, step, f"{name}/ppl", float(ppl))
            add_metric(points, step, "local_iteration", int(local_iteration))
            counts["probe"] += 1
            continue

        match = KD_RE.search(line)
        if match:
            local_iteration, kl, batches = match.groups()
            step = int(local_iteration) + step_offset
            if step <= step_offset or step > max_step:
                continue
            add_metric(points, step, "router_memory_teacher_student/kl", float(kl))
            add_metric(points, step, "router_memory_teacher_student/scaled_kl", float(kl))
            points[step]["router_memory_teacher_student/scaled_kl"] *= 1.0
            add_metric(points, step, "router_memory_teacher_student/batches", int(batches))
            add_metric(points, step, "local_iteration", int(local_iteration))
            counts["kd"] += 1
            continue

        match = TIME_RE.search(line)
        if match:
            _timestamp, local_iteration, _total, elapsed_ms = match.groups()
            step = int(local_iteration) + step_offset
            if step <= step_offset or step > max_step:
                continue
            add_metric(points, step, "timing/elapsed_ms_per_iteration", float(elapsed_ms))
            add_metric(points, step, "local_iteration", int(local_iteration))
            counts["time"] += 1
    return counts


def should_skip_tb_tag(tag: str):
    if " vs samples" in tag:
        return True
    if tag.endswith(" vs samples"):
        return True
    if tag.startswith("code_probe/") or tag.startswith("wiki_probe/"):
        return True
    if tag.startswith("validation/") or " validation" in tag:
        return True
    return False


def load_event_file_scalars(event_file: Path):
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    for tag in tags:
        for event in accumulator.Scalars(tag):
            yield tag, int(event.step), float(event.value)


def parse_tensorboard(points, run_dir: Path, step_offset: int, train_iters: int):
    max_step = step_offset + train_iters
    event_files = sorted(run_dir.rglob("events.out.tfevents*"))
    counts = defaultdict(int)
    for event_file in event_files:
        for tag, local_step, value in load_event_file_scalars(event_file):
            if should_skip_tb_tag(tag):
                continue
            if local_step <= 0 or local_step > train_iters:
                continue
            step = local_step + step_offset
            if step <= step_offset or step > max_step:
                continue
            add_metric(points, step, tag, value)
            add_metric(points, step, "local_iteration", local_step)
            counts[tag] += 1
    return counts


def add_derived_metrics(points, kd_coeff: float):
    for step, metrics in points.items():
        raw_kl = metrics.get("router_memory_teacher_student/kl")
        if raw_kl is not None:
            metrics["router_memory_teacher_student/scaled_kl"] = raw_kl * kd_coeff
        lm = metrics.get("lm loss")
        scaled_kl = metrics.get("router_memory_teacher_student/scaled_kl")
        if lm is not None and raw_kl is not None and raw_kl > 0:
            metrics["router_memory_teacher_student/lm_to_raw_kl"] = lm / raw_kl
        if lm is not None and scaled_kl is not None and scaled_kl > 0:
            metrics["router_memory_teacher_student/lm_to_scaled_kl"] = lm / scaled_kl
            metrics["combined/lm_plus_router_kd"] = lm + scaled_kl


def load_metadata(run_dir: Path):
    metadata_path = run_dir / "logs" / "run_metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main():
    args = parse_args()
    points = defaultdict(dict)

    base_count = parse_base_probes(points, Path(args.wiki_log), args.step_offset)
    code_counts = parse_code_log(points, Path(args.code_log), args.step_offset, args.train_iters)
    tb_counts = parse_tensorboard(points, Path(args.run_dir), args.step_offset, args.train_iters)
    add_derived_metrics(points, args.kd_coeff)

    steps = sorted(points)
    if not steps:
        raise SystemExit("No metrics were parsed.")

    print(f"base probe records at step {args.step_offset}: {base_count}")
    print(f"code log counts: {dict(code_counts)}")
    print(f"tensorboard tags parsed: {len(tb_counts)}")
    print(f"logged steps: {len(steps)} first={steps[0]} last={steps[-1]}")
    print(f"run_id={args.run_id}")
    print(f"run_name={args.run_name}")

    if args.dry_run:
        for step in steps[:3] + steps[-3:]:
            print(step, sorted(points[step])[:12], "...")
        return

    init_kwargs = {
        "project": args.project,
        "id": args.run_id,
        "name": args.run_name,
        "resume": "allow",
        "config": {
            "relog_source": "routerkd_base1800_allmetrics",
            "step_offset": args.step_offset,
            "train_iters": args.train_iters,
            "kd_coeff": args.kd_coeff,
            "source_run_dir": str(args.run_dir),
            **load_metadata(Path(args.run_dir)),
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
