#!/usr/bin/env python3
"""Replay an expert-expansion distillation pipeline to one lightweight W&B run.

The checkpoint weights are never uploaded.  Checkpoint trackers are used only
to validate the stages; probe and training metrics are parsed directly from the
stage logs and remapped onto one logical timeline:

    teacher -> distill initialization -> code training -> router retuning
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


PROBE_RE = re.compile(
    r"probe\s+(\S+)\s+at iteration\s+(\d+)\s+\|\s+local_iteration:\s+(\d+)\s+\|\s+"
    r"next_token_acc:\s+([0-9.eE+-]+)\s+\|\s+ppl:\s+([0-9.eE+-]+)"
)
ITER_RE = re.compile(r"\b(?:iteration|step)\s+(\d+)\s*/\s*(\d+)\b")
SAVE_RE = re.compile(r"saving checkpoint(?: step| at iteration)?\s+(\d+)")
ELAPSED_MS_RE = re.compile(r"elapsed time per iteration \(ms\):\s*([0-9.eE+-]+)")
ELAPSED_SEC_RE = re.compile(r"([0-9.eE+-]+)\s*s/it")
ALT_ELAPSED_MS_RE = re.compile(r"([0-9.eE+-]+)\s*ms/iter")

METRIC_PATTERNS = {
    "lm loss": re.compile(r"\blm loss\s*:?\s*([0-9.eE+-]+)"),
    "kd loss": re.compile(r"\bkd loss\s*:?\s*([0-9.eE+-]+)"),
    "hidden mse loss": re.compile(r"\bhidden mse loss\s*:?\s*([0-9.eE+-]+)"),
    "router prob kl loss": re.compile(r"\brouter prob kl loss\s*:?\s*([0-9.eE+-]+)"),
    "learning rate": re.compile(r"\blearning rate\s*:?\s*([0-9.eE+-]+)"),
    "grad norm": re.compile(r"\bgrad norm\s*:?\s*([0-9.eE+-]+)"),
    "loss scale": re.compile(r"\bloss scale\s*:?\s*([0-9.eE+-]+)"),
    "num zeros": re.compile(r"\b(?:num(?:ber)? of zeros|num-zeros)\s*:?\s*([0-9.eE+-]+)"),
    "batch size": re.compile(r"\b(?:batch size|batch-size)\s*:?\s*([0-9.eE+-]+)"),
}


@dataclass(frozen=True)
class Stage:
    name: str
    run_dir: Path
    log_path: Path
    display_base: int
    train_iters: int
    source_checkpoint_step: int
    keep_initial_probe: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", required=True)
    parser.add_argument("--distill-dir", required=True)
    parser.add_argument("--code-dir", required=True)
    parser.add_argument("--retune-dir", required=True)
    parser.add_argument("--teacher-log")
    parser.add_argument("--distill-log")
    parser.add_argument("--code-log")
    parser.add_argument("--retune-log")
    parser.add_argument("--teacher-step", type=int, default=1800)
    parser.add_argument("--distill-iters", type=int, default=1800)
    parser.add_argument("--code-iters", type=int, default=1800)
    parser.add_argument("--retune-iters", type=int, default=1800)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "flame-continual-top2-qv-lora"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--group")
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def tracker_step(run_dir: Path) -> int | None:
    path = run_dir / "latest_checkpointed_iteration.txt"
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8", errors="replace").strip()
    return int(value) if value.isdigit() else None


def infer_log(run_dir: Path, explicit: str | None, stage_name: str) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {stage_name} log: {path}")
        return path

    log_dir = run_dir / "logs"
    preferred_names = [
        "run.log",
        "phase3_run.log",
        "phase3_moe_router_run.log",
        "a_to_b_freeze.log",
    ]
    candidates: list[Path] = []
    for name in preferred_names:
        path = log_dir / name
        if path.is_file() and path not in candidates:
            candidates.append(path)
    for path in sorted(log_dir.glob("*.log")):
        if path not in candidates:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No log files found for {stage_name} under {log_dir}")

    def score(path: Path) -> tuple[int, int, int]:
        text = path.read_text(encoding="utf-8", errors="replace")
        return (len(PROBE_RE.findall(text)), len(ITER_RE.findall(text)), path.stat().st_size)

    return max(candidates, key=score)


def finite_float(value: str | float | int) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def add_metric(
    points: dict[int, dict[str, float]], step: int, key: str, value: str | float | int
) -> None:
    result = finite_float(value)
    if result is not None:
        points[int(step)][key] = result


def last_teacher_probes(log_path: Path, teacher_step: int) -> dict[str, tuple[float, float]]:
    records: dict[str, tuple[int, int, float, float]] = {}
    for line in read_lines(log_path):
        match = PROBE_RE.search(line)
        if not match:
            continue
        name, raw, local, acc, ppl = match.groups()
        raw_i, local_i = int(raw), int(local)
        if raw_i > teacher_step and local_i > teacher_step:
            continue
        rank = max(raw_i, local_i)
        previous = records.get(name)
        if previous is None or rank >= previous[0]:
            records[name] = (rank, local_i, float(acc), float(ppl))
    return {name: (record[2], record[3]) for name, record in records.items()}


def infer_relative_step(raw_step: int, local_step: int, stage: Stage) -> int | None:
    for value in (local_step, raw_step):
        if 0 <= value <= stage.train_iters:
            return value
        lower = stage.source_checkpoint_step
        if lower <= value <= lower + stage.train_iters:
            return value - lower
    return None


def infer_training_relative_step(raw_step: int, stage: Stage) -> int | None:
    if 0 < raw_step <= stage.train_iters:
        return raw_step
    lower = stage.source_checkpoint_step
    if lower < raw_step <= lower + stage.train_iters:
        return raw_step - lower
    return None


def parse_stage(
    points: dict[int, dict[str, float]],
    stage: Stage,
) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    seen_probe_names: set[str] = set()
    for line in read_lines(stage.log_path):
        probe = PROBE_RE.search(line)
        if probe:
            name, raw, local, acc, ppl = probe.groups()
            local_i = int(local)
            is_first_for_probe = name not in seen_probe_names
            seen_probe_names.add(name)
            if (
                not stage.keep_initial_probe
                and is_first_for_probe
                and local_i in (0, stage.source_checkpoint_step)
            ):
                counts["skipped_initial_probe"] += 1
                continue
            relative = infer_relative_step(int(raw), local_i, stage)
            if relative is None:
                counts["skipped_probe"] += 1
                continue
            if relative == 0:
                if not stage.keep_initial_probe:
                    counts["skipped_initial_probe"] += 1
                    continue
                # Preserve the post-expansion drop without overwriting the
                # teacher value logged at the exact stage boundary.
                display_step = stage.display_base + 1
                marker = f"stage/{stage.name}_post_expand_initial"
            else:
                display_step = stage.display_base + relative
                marker = f"stage/{stage.name}_active"
            add_metric(points, display_step, f"{name}/next_token_accuracy", acc)
            add_metric(points, display_step, f"{name}/ppl", ppl)
            add_metric(points, display_step, "local_iteration", relative)
            add_metric(points, display_step, marker, 1.0)
            counts["probe"] += 1
            continue

        iteration = ITER_RE.search(line)
        if iteration:
            raw_step = int(iteration.group(1))
            relative = infer_training_relative_step(raw_step, stage)
            if relative is None or relative <= 0:
                continue
            display_step = stage.display_base + relative
            for metric_name, pattern in METRIC_PATTERNS.items():
                match = pattern.search(line)
                if match:
                    add_metric(points, display_step, metric_name, match.group(1))
                    counts[metric_name] += 1
            elapsed = ELAPSED_MS_RE.search(line)
            if elapsed:
                add_metric(points, display_step, "timing/elapsed_ms_per_iteration", elapsed.group(1))
                counts["time"] += 1
            else:
                elapsed = ELAPSED_SEC_RE.search(line)
                if elapsed:
                    add_metric(
                        points,
                        display_step,
                        "timing/elapsed_ms_per_iteration",
                        float(elapsed.group(1)) * 1000.0,
                    )
                    counts["time"] += 1
                else:
                    elapsed = ALT_ELAPSED_MS_RE.search(line)
                    if elapsed:
                        add_metric(
                            points,
                            display_step,
                            "timing/elapsed_ms_per_iteration",
                            elapsed.group(1),
                        )
                        counts["time"] += 1
            add_metric(points, display_step, f"stage/{stage.name}_active", 1.0)
            continue

        saved = SAVE_RE.search(line)
        if saved:
            relative = infer_training_relative_step(int(saved.group(1)), stage)
            if relative is not None and relative > 0:
                add_metric(points, stage.display_base + relative, "checkpoint/saved", 1.0)
                counts["checkpoint"] += 1
    return dict(counts)


def load_metadata(run_dir: Path) -> dict:
    log_dir = run_dir / "logs"
    for path in [log_dir / "run_metadata.json", log_dir / "phase3_run_metadata.json"]:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def print_boundary(points: dict[int, dict[str, float]], step: int) -> None:
    payload = points.get(step, {})
    probes = {
        key: value
        for key, value in payload.items()
        if key.endswith(("/next_token_accuracy", "/ppl"))
    }
    print(f"BOUNDARY {step}: {probes or '[no probe at exact boundary]'}")


def main() -> None:
    args = parse_args()
    teacher_dir = Path(args.teacher_dir)
    distill_dir = Path(args.distill_dir)
    code_dir = Path(args.code_dir)
    retune_dir = Path(args.retune_dir)
    all_dirs = [teacher_dir, distill_dir, code_dir, retune_dir]
    for run_dir in all_dirs:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Missing run directory: {run_dir}")

    logs = {
        "teacher": infer_log(teacher_dir, args.teacher_log, "teacher"),
        "distill": infer_log(distill_dir, args.distill_log, "distill"),
        "code": infer_log(code_dir, args.code_log, "code"),
        "retune": infer_log(retune_dir, args.retune_log, "retune"),
    }
    trackers = {name: tracker_step(path) for name, path in zip(
        ("teacher", "distill", "code", "retune"), all_dirs
    )}

    distill_base = args.teacher_step
    code_base = distill_base + args.distill_iters
    retune_base = code_base + args.code_iters
    stages = [
        Stage("distill_init", distill_dir, logs["distill"], distill_base, args.distill_iters, args.teacher_step, True),
        Stage("code_train", code_dir, logs["code"], code_base, args.code_iters, trackers["distill"] or args.distill_iters),
        Stage("router_retune", retune_dir, logs["retune"], retune_base, args.retune_iters, trackers["code"] or args.code_iters),
    ]

    points: dict[int, dict[str, float]] = defaultdict(dict)
    teacher_probes = last_teacher_probes(logs["teacher"], args.teacher_step)
    if not teacher_probes:
        raise RuntimeError(f"No teacher probes found in {logs['teacher']}")
    for name, (acc, ppl) in teacher_probes.items():
        add_metric(points, args.teacher_step, f"{name}/next_token_accuracy", acc)
        add_metric(points, args.teacher_step, f"{name}/ppl", ppl)
    add_metric(points, args.teacher_step, "stage/teacher_final", 1.0)
    add_metric(points, args.teacher_step, "local_iteration", args.teacher_step)

    stage_counts = {stage.name: parse_stage(points, stage) for stage in stages}
    steps = sorted(points)
    if not steps:
        raise RuntimeError("No metrics were parsed.")

    print("=== checkpoint trackers ===")
    for name, run_dir in zip(("teacher", "distill", "code", "retune"), all_dirs):
        print(f"{name:8s}: tracker={trackers[name]} dir={run_dir}")
    print("=== selected logs ===")
    for name, path in logs.items():
        print(f"{name:8s}: {path}")
    print("=== parsed metrics ===")
    print(f"teacher probes={len(teacher_probes)} {teacher_probes}")
    for name, counts in stage_counts.items():
        print(f"{name:14s}: {counts}")
    print(f"steps={len(steps)} first={steps[0]} last={steps[-1]}")
    for boundary in [args.teacher_step, code_base, retune_base, steps[-1]]:
        print_boundary(points, boundary)

    if args.dry_run:
        return

    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is not installed; install it before online upload.") from exc

    config = {
        "relog_source": "expansion_distill_pipeline_logs",
        "teacher_step": args.teacher_step,
        "distill_iters": args.distill_iters,
        "code_iters": args.code_iters,
        "retune_iters_requested": args.retune_iters,
        "display_mapping": {
            "teacher_final": args.teacher_step,
            "distill_init": [distill_base, code_base],
            "code_train": [code_base, retune_base],
            "router_retune": [retune_base, retune_base + args.retune_iters],
        },
        "checkpoint_trackers": trackers,
        "run_dirs": {name: str(path) for name, path in zip(
            ("teacher", "distill", "code", "retune"), all_dirs
        )},
        "logs": {name: str(path) for name, path in logs.items()},
        "stage_metadata": {
            "distill": load_metadata(distill_dir),
            "code": load_metadata(code_dir),
            "retune": load_metadata(retune_dir),
        },
    }
    init_kwargs = {
        "project": args.project,
        "id": args.run_id,
        "name": args.run_name,
        "resume": "never",
        "config": config,
        "tags": args.tags,
        "settings": wandb.Settings(code_dir=None, console="off"),
    }
    if args.entity:
        init_kwargs["entity"] = args.entity
    if args.group:
        init_kwargs["group"] = args.group

    run = wandb.init(**init_kwargs)
    for step in steps:
        wandb.log(points[step], step=step)
    run.summary["parsed_first_step"] = steps[0]
    run.summary["parsed_last_step"] = steps[-1]
    run.summary["checkpoint_weights_uploaded"] = False
    wandb.finish()
    print(f"[UPLOADED] {args.run_name} id={args.run_id} steps={steps[0]}..{steps[-1]}")


if __name__ == "__main__":
    main()
