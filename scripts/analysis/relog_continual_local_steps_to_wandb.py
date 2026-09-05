#!/usr/bin/env python3
"""Re-log a completed continual-learning run on a requested logical step axis.

TensorBoard training scalars use local steps (1..train_iters). Probe lines in
the text log carry an explicit local_iteration.  Both are mapped to

    logical_step = step_offset + step_scale * local_iteration

This allows, for example, a local 0..1800 run to be plotted as logical
1800..5400 with ``step_offset=1800`` and ``step_scale=2``.
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "flame-continual-top2-qv-lora"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--step-offset", type=int, default=1800)
    parser.add_argument("--train-iters", type=int, default=1800)
    parser.add_argument("--step-scale", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def add(points, step, key, value):
    value = float(value)
    if math.isfinite(value):
        points[int(step)][key] = value


def parse_probes(points, log_path: Path, step_offset: int, train_iters: int, step_scale: int):
    counts = defaultdict(int)
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = PROBE_RE.search(line)
        if not match:
            continue
        name, printed_iteration, local_iteration, accuracy, ppl = match.groups()
        local_iteration = int(local_iteration)
        if not 0 <= local_iteration <= train_iters:
            continue
        logical_step = step_offset + step_scale * local_iteration
        add(points, logical_step, f"{name}/next_token_accuracy", accuracy)
        add(points, logical_step, f"{name}/ppl", ppl)
        add(points, logical_step, "local_iteration", local_iteration)
        add(points, logical_step, "source/printed_probe_iteration", printed_iteration)
        counts[name] += 1
    return counts


def skip_tensorboard_tag(tag: str):
    # These duplicate iteration-based metrics but use consumed samples as the
    # event step, so they cannot be shifted by a constant iteration offset.
    if " vs samples" in tag or tag.endswith(" vs samples"):
        return True
    # Probe events use the source checkpoint's printed offset.  The text log's
    # local_iteration is authoritative for the requested logical axis.
    if tag.startswith("code_probe/") or tag.startswith("wiki_probe/"):
        return True
    return False


def parse_tensorboard(points, run_dir: Path, step_offset: int, train_iters: int, step_scale: int):
    event_files = sorted(run_dir.rglob("events.out.tfevents*"))
    if not event_files:
        raise RuntimeError(f"No TensorBoard event files found below {run_dir}")
    counts = defaultdict(int)
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            if skip_tensorboard_tag(tag):
                continue
            for event in accumulator.Scalars(tag):
                local_iteration = int(event.step)
                if not 1 <= local_iteration <= train_iters:
                    continue
                logical_step = step_offset + step_scale * local_iteration
                add(points, logical_step, tag, event.value)
                add(points, logical_step, "local_iteration", local_iteration)
                counts[tag] += 1
    return counts, event_files


def load_metadata(run_dir: Path):
    path = run_dir / "logs" / "run_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    log_path = Path(args.log).resolve()
    points = defaultdict(dict)

    if args.step_scale <= 0:
        raise ValueError("--step-scale must be positive")
    probe_counts = parse_probes(
        points, log_path, args.step_offset, args.train_iters, args.step_scale
    )
    tb_counts, event_files = parse_tensorboard(
        points, run_dir, args.step_offset, args.train_iters, args.step_scale
    )
    points[args.step_offset]["source/base_step_marker"] = 1.0
    points[args.step_offset]["local_iteration"] = 0.0

    steps = sorted(points)
    expected_last = args.step_offset + args.step_scale * args.train_iters
    if not steps or steps[0] != args.step_offset or steps[-1] != expected_last:
        raise RuntimeError(
            f"Unexpected logical range: {steps[0] if steps else None}.."
            f"{steps[-1] if steps else None}; expected {args.step_offset}..{expected_last}"
        )
    if probe_counts.get("code_probe", 0) != probe_counts.get("wiki_probe", 0):
        raise RuntimeError(f"Mismatched probe counts: {dict(probe_counts)}")

    print(f"run_dir: {run_dir}")
    print(f"event_files: {len(event_files)}")
    print(f"tensorboard tags: {len(tb_counts)}")
    print(f"probe counts: {dict(probe_counts)}")
    print(f"logical steps: {steps[0]}..{steps[-1]} ({len(steps)} populated steps)")
    print(f"run_id: {args.run_id}")
    print(f"run_name: {args.run_name}")
    for step in (steps[0], steps[-1]):
        print(f"step {step}: {sorted(points[step])}")

    if args.dry_run:
        return

    metadata = load_metadata(run_dir)
    # Keep the external payload intentionally small.  In particular, do not
    # upload local checkpoint, dataset, scratch, or log paths from metadata.
    safe_metadata_keys = (
        "run_id",
        "dataset_name",
        "dataset_source",
        "train_iters",
        "micro_batch_size",
        "global_batch_size",
        "num_layers",
        "hidden_size",
        "ffn_hidden_size",
        "num_query_groups",
        "source_num_experts",
        "target_num_experts",
        "moe_ffn_hidden_size",
        "moe_router_topk",
        "precision",
        "lr",
        "min_lr",
        "lr_decay_style",
        "lr_decay_iters",
        "lr_wsd_decay_iters",
        "lr_warmup_fraction",
        "moe_joint_replay_old_data_hidden_mse",
        "moe_joint_replay_old_data_hidden_kl",
        "moe_joint_replay_old_data_kd",
        "moe_joint_replay_total_samples",
        "moe_joint_replay_micro_batch_size",
        "moe_joint_replay_old_like_unit",
        "old_hidden_mse_coeff",
        "old_hidden_mse_layers",
        "old_hidden_kl_coeff",
        "old_hidden_kl_temperature",
        "old_hidden_kl_layers",
        "old_model_kl_coeff",
        "old_model_kl_temperature",
        "probe_step_offset",
    )
    config = {
        **{key: metadata[key] for key in safe_metadata_keys if key in metadata},
        "relog_source": "continual_local_steps_to_wandb",
        "logical_step_offset": args.step_offset,
        "logical_step_scale": args.step_scale,
        "logical_final_step": expected_last,
        "source_probe_step_offset": metadata.get("probe_step_offset"),
    }
    objective_tags = []
    if metadata.get("moe_joint_replay_old_data_hidden_mse"):
        objective_tags.append("hidden-mse")
    if metadata.get("moe_joint_replay_old_data_hidden_kl"):
        objective_tags.append("hidden-kl")
    if metadata.get("moe_joint_replay_old_data_kd"):
        objective_tags.append("vocab-kl")
    init_kwargs = {
        "project": args.project,
        "id": args.run_id,
        "name": args.run_name,
        "resume": "allow",
        "config": config,
        "notes": (
            f"Re-logged completed continual run with local 0..{args.train_iters} mapped to "
            f"logical {args.step_offset}..{expected_last} at x{args.step_scale} scale."
        ),
        "tags": ["old-like-gt", *objective_tags, "logical-step-remap"],
    }
    if args.entity:
        init_kwargs["entity"] = args.entity

    run = wandb.init(**init_kwargs)
    for step in steps:
        payload = dict(points[step])
        wandb.log(payload, step=step)
    run.summary["logical_first_step"] = steps[0]
    run.summary["logical_last_step"] = steps[-1]
    run.summary["source_local_train_iters"] = args.train_iters
    print(f"wandb_url: {run.url}")
    wandb.finish()


if __name__ == "__main__":
    main()
