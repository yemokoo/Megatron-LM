#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import wandb
from tensorboard.backend.event_processing import event_accumulator


PROBE_LINE_RE = re.compile(
    r"probe\s+(?P<name>\S+)\s+at iteration\s+(?P<step>\d+)\s+\|\s+"
    r"local_iteration:\s+(?P<local_step>\d+)\s+\|\s+"
    r"next_token_acc:\s+(?P<acc>[0-9.]+)\s+\|\s+ppl:\s+(?P<ppl>[0-9.Ee+-]+)"
)


def parse_candidates(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def find_event_file(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("events.out.tfevents*"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one events.out.tfevents* under {run_dir}, found {len(candidates)}"
        )
    return candidates[0]


def load_scalars(event_file: Path):
    acc = event_accumulator.EventAccumulator(
        str(event_file),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    acc.Reload()
    return acc


def find_scalar_at_or_before_step(acc, tag: str, step: int):
    try:
        events = acc.Scalars(tag)
    except KeyError:
        return None
    chosen = None
    for event in events:
        if int(event.step) <= step:
            chosen = event
        else:
            break
    return chosen


def resolve_metric(acc, candidates: list[str], suffix: str, step: int):
    for candidate in candidates:
        tag = f"{candidate}/{suffix}"
        event = find_scalar_at_or_before_step(acc, tag, step)
        if event is not None:
            return candidate, float(event.value), int(event.step)
    raise KeyError(
        f"Could not find any '{suffix}' scalar at or before step {step} for candidates {candidates}"
    )


def parse_probe_history(log_file: Path):
    history = []
    seen = set()
    with log_file.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            match = PROBE_LINE_RE.search(raw_line)
            if match is None:
                continue
            step = int(match.group("step"))
            probe_name = match.group("name")
            key = (probe_name, step)
            if key in seen:
                continue
            seen.add(key)
            history.append(
                {
                    "probe_name": probe_name,
                    "step": step,
                    "local_step": int(match.group("local_step")),
                    "next_token_accuracy": float(match.group("acc")),
                    "ppl": float(match.group("ppl")),
                }
            )
    history.sort(key=lambda item: (item["step"], item["probe_name"]))
    return history


def log_step(run, step: int, payload: dict):
    wandb.log(payload, step=step)


def main():
    parser = argparse.ArgumentParser(
        description="Replay source baseline probe values plus continual probe history into a new W&B run."
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--continual-log", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--entity")
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--mode", default="online", choices=["online", "offline"])
    parser.add_argument("--baseline-step", type=int, required=True)
    parser.add_argument("--primary-source-candidates", required=True)
    parser.add_argument("--primary-target-name", required=True)
    parser.add_argument("--secondary-source-candidates")
    parser.add_argument("--secondary-target-name")
    args = parser.parse_args()

    source_event_file = find_event_file(args.source_run_dir)
    acc = load_scalars(source_event_file)

    baseline_payload = {}

    primary_candidates = parse_candidates(args.primary_source_candidates)
    primary_acc_name, primary_acc, _ = resolve_metric(
        acc, primary_candidates, "next_token_accuracy", args.baseline_step
    )
    primary_ppl_name, primary_ppl, _ = resolve_metric(
        acc, primary_candidates, "ppl", args.baseline_step
    )
    if primary_acc_name != primary_ppl_name:
        raise RuntimeError(
            f"Primary metrics resolved from different source tags: {primary_acc_name} vs {primary_ppl_name}"
        )
    baseline_payload[f"{args.primary_target_name}/next_token_accuracy"] = primary_acc
    baseline_payload[f"{args.primary_target_name}/ppl"] = primary_ppl

    if args.secondary_source_candidates and args.secondary_target_name:
        secondary_candidates = parse_candidates(args.secondary_source_candidates)
        secondary_acc_name, secondary_acc, _ = resolve_metric(
            acc, secondary_candidates, "next_token_accuracy", args.baseline_step
        )
        secondary_ppl_name, secondary_ppl, _ = resolve_metric(
            acc, secondary_candidates, "ppl", args.baseline_step
        )
        if secondary_acc_name != secondary_ppl_name:
            raise RuntimeError(
                f"Secondary metrics resolved from different source tags: {secondary_acc_name} vs {secondary_ppl_name}"
            )
        baseline_payload[f"{args.secondary_target_name}/next_token_accuracy"] = secondary_acc
        baseline_payload[f"{args.secondary_target_name}/ppl"] = secondary_ppl

    history = parse_probe_history(args.continual_log)
    if not history:
        raise RuntimeError(f"No probe history found in {args.continual_log}")

    init_kwargs = {
        "project": args.project,
        "name": args.run_name,
        "id": args.run_id,
        "resume": "never",
        "mode": args.mode,
    }
    if args.entity:
        init_kwargs["entity"] = args.entity
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        init_kwargs["dir"] = str(args.save_dir)

    run = wandb.init(**init_kwargs)
    try:
        log_step(run, args.baseline_step, baseline_payload)
        current_step = None
        current_payload = {}
        for item in history:
            step = item["step"]
            if current_step is None:
                current_step = step
            if step != current_step:
                log_step(run, current_step, current_payload)
                current_payload = {}
                current_step = step
            probe_name = item["probe_name"]
            current_payload[f"{probe_name}/next_token_accuracy"] = item["next_token_accuracy"]
            current_payload[f"{probe_name}/ppl"] = item["ppl"]
        if current_step is not None and current_payload:
            log_step(run, current_step, current_payload)
    finally:
        run.finish()

    print(f"Logged baseline step {args.baseline_step} from {source_event_file}")
    print(f"Replayed {len(history)} probe records from {args.continual_log}")


if __name__ == "__main__":
    main()
