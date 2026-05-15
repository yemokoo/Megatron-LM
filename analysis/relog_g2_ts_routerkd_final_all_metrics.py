#!/usr/bin/env python3
"""Relog the G2 teacher-student router-KD run to W&B.

This script combines:
  - probe/KD/timing values parsed from run.log
  - scalar losses and auxiliary metrics parsed from TensorBoard events

TensorBoard scalar steps are not fully consistent across custom metrics:
some are logged with local training iteration (0..1800), while probe/router
metrics can already be logged with the global offset (1800..3600).  We infer
the mapping per tag to avoid adding the offset twice.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable

import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PROBE_RE = re.compile(
    r"probe (\S+) at iteration\s+(\d+)\s+\|\s+local_iteration:\s+(\d+)\s+\|"
    r"\s+next_token_acc:\s+([0-9.eE+-]+)\s+\|\s+ppl:\s+([0-9.eE+-]+)"
)
KD_RE = re.compile(
    r"teacher-student router memory KD at iteration\s+(\d+)\s+\|\s+kl:\s+([0-9.eE+-]+)"
    r"\s+\|\s+batches:\s+(\d+)"
)
TIME_RE = re.compile(
    r"iteration\s+(\d+)/\s*(\d+).*elapsed time per iteration \(ms\):\s*([0-9.eE+-]+)"
)


Rows = Dict[int, Dict[str, float]]


def read_lines(path: Path) -> Iterable[str]:
    return path.read_text(errors="ignore").splitlines()


def add(rows: Rows, step: int, payload: Dict[str, float]) -> None:
    rows.setdefault(int(step), {}).update(payload)


def parse_probes(path: Path):
    for line in read_lines(path):
        match = PROBE_RE.search(line)
        if not match:
            continue
        yield {
            "probe": match.group(1),
            "global_step": int(match.group(2)),
            "local_step": int(match.group(3)),
            "accuracy": float(match.group(4)),
            "ppl": float(match.group(5)),
        }


def should_skip_tb_tag(tag: str) -> bool:
    # These use samples as the TensorBoard x-axis, not iteration.
    if tag.endswith(" vs samples"):
        return True
    if tag == "samples vs steps":
        return True
    return False


def map_tb_step(raw_step: int, max_raw_step_for_tag: int, step_offset: int) -> int:
    """Map a TensorBoard event step to W&B step.

    Raw training scalars are usually logged as 0..train_iters, so add the
    offset. Probe/router custom scalars may already be logged as
    offset..offset+train_iters, so keep them as-is.
    """
    if max_raw_step_for_tag <= step_offset:
        return step_offset + raw_step
    return raw_step


def read_tensorboard_scalars(run_dir: Path, step_offset: int) -> tuple[Rows, list[str]]:
    rows: Rows = {}
    tags_seen: set[str] = set()
    event_files = sorted(
        run_dir.rglob("events.out.tfevents.*"), key=lambda path: path.stat().st_mtime
    )

    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            tags_seen.add(tag)
            if should_skip_tb_tag(tag):
                continue

            events = accumulator.Scalars(tag)
            positive_steps = [int(event.step) for event in events if int(event.step) > 0]
            if not positive_steps:
                continue
            max_raw_step = max(positive_steps)

            for event in events:
                raw_step = int(event.step)
                if raw_step <= 0:
                    continue
                mapped_step = map_tb_step(raw_step, max_raw_step, step_offset)
                # Replace the expanded-student initial point with the base wiki
                # model probe point that we log separately at step_offset.
                if mapped_step <= step_offset:
                    continue
                add(rows, mapped_step, {tag: float(event.value)})

    return rows, sorted(tags_seen)


def parse_base_probe_payload(wiki_log: Path, step_offset: int) -> Dict[str, float]:
    latest_by_probe = {}
    for row in parse_probes(wiki_log):
        if row["global_step"] <= step_offset:
            latest_by_probe[row["probe"]] = row

    payload: Dict[str, float] = {
        "local_iteration": 0,
        "source/base_step_marker": 1,
    }
    for probe, row in latest_by_probe.items():
        payload[f"{probe}/next_token_accuracy"] = row["accuracy"]
        payload[f"{probe}/ppl"] = row["ppl"]
    return payload


def parse_log_metrics(code_log: Path, rows: Rows, step_offset: int, kd_coeff: float) -> None:
    for line in read_lines(code_log):
        match = KD_RE.search(line)
        if match:
            local_step = int(match.group(1))
            kl = float(match.group(2))
            add(
                rows,
                step_offset + local_step,
                {
                    "local_iteration": local_step,
                    "router_memory_teacher_student/kl": kl,
                    "router_memory_teacher_student/scaled_kl": kd_coeff * kl,
                    "router_memory_teacher_student/batches": int(match.group(3)),
                },
            )

        match = TIME_RE.search(line)
        if match:
            local_step = int(match.group(1))
            add(
                rows,
                step_offset + local_step,
                {
                    "local_iteration": local_step,
                    "timing/elapsed_ms_per_iteration": float(match.group(3)),
                },
            )

    for row in parse_probes(code_log):
        local_step = row["local_step"]
        if local_step <= 0:
            continue
        add(
            rows,
            step_offset + local_step,
            {
                "local_iteration": local_step,
                f"{row['probe']}/next_token_accuracy": row["accuracy"],
                f"{row['probe']}/ppl": row["ppl"],
            },
        )


def add_combined_loss(rows: Rows, kd_coeff: float) -> int:
    combined_count = 0
    for payload in rows.values():
        lm_loss = payload.get("lm loss")
        kl_loss = payload.get("router_memory_teacher_student/kl")
        if lm_loss is None or kl_loss is None:
            continue
        payload["combined/lm_plus_router_kd"] = lm_loss + kd_coeff * kl_loss
        combined_count += 1
    return combined_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-log", type=Path, required=True)
    parser.add_argument("--code-log", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--step-offset", type=int, default=1800)
    parser.add_argument("--kd-coeff", type=float, default=1.0)
    args = parser.parse_args()

    rows, tb_tags = read_tensorboard_scalars(args.run_dir, args.step_offset)
    parse_log_metrics(args.code_log, rows, args.step_offset, args.kd_coeff)
    base_payload = parse_base_probe_payload(args.wiki_log, args.step_offset)
    combined_count = add_combined_loss(rows, args.kd_coeff)

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "flame-continual-top2-qv-lora"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        id=args.run_id,
        name=args.run_name,
        resume="never",
        config={
            "source_run_dir": str(args.run_dir),
            "step_offset": args.step_offset,
            "kd_coeff": args.kd_coeff,
            "base_step_uses_pre_expansion_wiki_model": True,
            "tb_scalar_tags_found": tb_tags,
            "tb_step_mapping": "per_tag_local_or_global_inferred",
        },
    )

    wandb.log(base_payload, step=args.step_offset)
    for step in sorted(rows):
        if step > args.step_offset:
            wandb.log(rows[step], step=step)
    run.finish()

    logged_tags = set()
    for payload in rows.values():
        logged_tags.update(payload)

    print(f"logged_steps={len(rows) + 1}")
    print(f"first_step={args.step_offset}")
    print(f"last_step={max(rows) if rows else args.step_offset}")
    print(f"has_lm_loss={'lm loss' in logged_tags}")
    print(f"has_combined={'combined/lm_plus_router_kd' in logged_tags}")
    print(f"combined_steps={combined_count}")
    print(f"run_id={args.run_id}")
    print("logged_tags:")
    for tag in sorted(logged_tags):
        print(f"  {tag}")


if __name__ == "__main__":
    main()
