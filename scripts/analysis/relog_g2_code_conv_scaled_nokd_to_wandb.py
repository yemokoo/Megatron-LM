#!/usr/bin/env python3
"""Re-log completed G2 Code/Conversation stages without KD trajectories.

The completed one-phase stages each contain 1800 local optimizer updates.  For
comparison with the older two-phase x-axis, local steps are stretched by two:

    Code          local 0..1800 -> display 1800..5400
    Conversation  local 0..1800 -> display 5400..9000

The local-step-zero probes are the corresponding post-KD checkpoint values.
KD training metrics themselves are intentionally not read or uploaded.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

from relog_expansion_distill_pipeline_to_wandb import (
    PROBE_RE,
    Stage,
    add_metric,
    infer_log,
    load_metadata,
    parse_stage,
    print_boundary,
    read_lines,
    tracker_step,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-dir", required=True)
    parser.add_argument("--conv-dir", required=True)
    parser.add_argument("--code-log")
    parser.add_argument("--conv-log")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "flame-continual-top2-qv-lora"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--group", default="g2-3objective-code-conv-scaled-nokd")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def initial_probes(log_path: Path) -> dict[str, tuple[float, float]]:
    """Return the first local-step-zero probe for each dataset."""
    result: dict[str, tuple[float, float]] = {}
    fallback: dict[str, tuple[float, float]] = {}
    for line in read_lines(log_path):
        match = PROBE_RE.search(line)
        if not match:
            continue
        name, _raw, local, acc, ppl = match.groups()
        fallback.setdefault(name, (float(acc), float(ppl)))
        if int(local) == 0:
            result.setdefault(name, (float(acc), float(ppl)))
    return result or fallback


def add_boundary_probes(points, step: int, probes, marker: str) -> None:
    for name, (acc, ppl) in probes.items():
        add_metric(points, step, f"{name}/next_token_accuracy", acc)
        add_metric(points, step, f"{name}/ppl", ppl)
    add_metric(points, step, marker, 1.0)
    add_metric(points, step, "local_iteration", 0)


def main() -> None:
    args = parse_args()
    code_dir, conv_dir = Path(args.code_dir), Path(args.conv_dir)
    for path in (code_dir, conv_dir):
        if not path.is_dir():
            raise FileNotFoundError(path)

    code_log = infer_log(code_dir, args.code_log, "code")
    conv_log = infer_log(conv_dir, args.conv_log, "final")
    if tracker_step(code_dir) != 1800 or tracker_step(conv_dir) != 1800:
        raise RuntimeError(
            f"Expected completed 1800-step logs, got code={tracker_step(code_dir)} "
            f"conv={tracker_step(conv_dir)}"
        )

    code_initial = initial_probes(code_log)
    conv_initial = initial_probes(conv_log)
    if not code_initial or not conv_initial:
        raise RuntimeError("Both Code and Conversation logs need initial post-KD probes")

    points: dict[int, dict[str, float]] = defaultdict(dict)
    add_boundary_probes(points, 1800, code_initial, "source/code_post_kd")
    code_stage = Stage("code_train", code_dir, code_log, 1800, 1800, 600, False, 2)
    code_counts = parse_stage(points, code_stage)

    # At the shared boundary, the Conversation post-KD value intentionally
    # replaces Code's terminal probe for the same metric names.
    add_boundary_probes(points, 5400, conv_initial, "source/conversation_post_kd")
    conv_stage = Stage("conversation_train", conv_dir, conv_log, 5400, 1800, 600, False, 2)
    conv_counts = parse_stage(points, conv_stage)

    steps = sorted(points)
    if not steps or steps[0] != 1800 or steps[-1] != 9000:
        raise RuntimeError(f"Bad display range: {steps[0] if steps else None}..{steps[-1] if steps else None}")

    print(f"objective={args.objective}")
    print(f"code_log={code_log}")
    print(f"conv_log={conv_log}")
    print(f"code_initial={code_initial}")
    print(f"conv_initial={conv_initial}")
    print(f"code_counts={code_counts}")
    print(f"conv_counts={conv_counts}")
    print(f"steps={len(steps)} first={steps[0]} boundary=5400 last={steps[-1]}")
    for boundary in (1800, 5400, 9000):
        print_boundary(points, boundary)
    if args.dry_run:
        return

    import wandb

    init_kwargs = {
        "project": args.project,
        "id": args.run_id,
        "name": args.run_name,
        "resume": "never",
        "group": args.group,
        "tags": ["g2", "completed-logs", "no-kd-trajectory", "one-phase", "step-scale-2", args.objective],
        "config": {
            "relog_source": "completed_g2_code_conv_logs",
            "objective": args.objective,
            "kd_trajectory_uploaded": False,
            "checkpoint_weights_uploaded": False,
            "display_mapping": {
                "code": [1800, 5400],
                "conversation": [5400, 9000],
                "local_step_scale": 2,
            },
            "code_dir": str(code_dir),
            "conv_dir": str(conv_dir),
            "code_log": str(code_log),
            "conv_log": str(conv_log),
            "code_metadata": load_metadata(code_dir),
            "conv_metadata": load_metadata(conv_dir),
        },
        "settings": wandb.Settings(code_dir=None, console="off"),
    }
    if args.entity:
        init_kwargs["entity"] = args.entity
    run = wandb.init(**init_kwargs)
    for step in steps:
        wandb.log(points[step], step=step)
    run.summary["parsed_first_step"] = steps[0]
    run.summary["parsed_boundary_step"] = 5400
    run.summary["parsed_last_step"] = steps[-1]
    run.summary["kd_trajectory_uploaded"] = False
    run.summary["checkpoint_weights_uploaded"] = False
    wandb.finish()
    print(f"[UPLOADED] {args.run_name} id={args.run_id} steps=1800..9000")


if __name__ == "__main__":
    main()
