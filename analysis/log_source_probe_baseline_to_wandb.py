#!/usr/bin/env python3
import argparse
from pathlib import Path

import wandb
from tensorboard.backend.event_processing import event_accumulator


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


def main():
    parser = argparse.ArgumentParser(
        description="Log source-model probe baselines into the target continual W&B run."
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", default="allow", choices=["never", "allow", "must", "auto"])
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--primary-source-candidates", required=True)
    parser.add_argument("--primary-target-name", required=True)
    parser.add_argument("--secondary-source-candidates")
    parser.add_argument("--secondary-target-name")
    args = parser.parse_args()

    event_file = find_event_file(args.source_run_dir)
    acc = load_scalars(event_file)

    primary_candidates = parse_candidates(args.primary_source_candidates)
    primary_acc_name, primary_acc, primary_acc_step = resolve_metric(
        acc, primary_candidates, "next_token_accuracy", args.step
    )
    primary_ppl_name, primary_ppl, primary_ppl_step = resolve_metric(
        acc, primary_candidates, "ppl", args.step
    )
    if primary_acc_name != primary_ppl_name:
        raise RuntimeError(
            f"Primary metrics resolved from different source tags: "
            f"{primary_acc_name} vs {primary_ppl_name}"
        )

    payload = {
        f"{args.primary_target_name}/next_token_accuracy": primary_acc,
        f"{args.primary_target_name}/ppl": primary_ppl,
    }

    messages = [
        f"primary {primary_acc_name} -> {args.primary_target_name} "
        f"(acc step={primary_acc_step}, ppl step={primary_ppl_step})"
    ]

    if args.secondary_source_candidates and args.secondary_target_name:
        secondary_candidates = parse_candidates(args.secondary_source_candidates)
        secondary_acc_name, secondary_acc, secondary_acc_step = resolve_metric(
            acc, secondary_candidates, "next_token_accuracy", args.step
        )
        secondary_ppl_name, secondary_ppl, secondary_ppl_step = resolve_metric(
            acc, secondary_candidates, "ppl", args.step
        )
        if secondary_acc_name != secondary_ppl_name:
            raise RuntimeError(
                f"Secondary metrics resolved from different source tags: "
                f"{secondary_acc_name} vs {secondary_ppl_name}"
            )
        payload.update(
            {
                f"{args.secondary_target_name}/next_token_accuracy": secondary_acc,
                f"{args.secondary_target_name}/ppl": secondary_ppl,
            }
        )
        messages.append(
            f"secondary {secondary_acc_name} -> {args.secondary_target_name} "
            f"(acc step={secondary_acc_step}, ppl step={secondary_ppl_step})"
        )

    init_kwargs = {
        "project": args.project,
        "name": args.run_name,
        "id": args.run_id,
        "resume": args.resume,
        "config": {
            "source_run_dir": str(args.source_run_dir),
            "source_event_file": str(event_file),
            "source_probe_baseline_step": args.step,
            "logged_source_probe_baseline": True,
        },
    }
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        init_kwargs["dir"] = str(args.save_dir)

    run = wandb.init(**init_kwargs)
    try:
        wandb.log(payload, step=args.step)
    finally:
        run.finish()

    print(f"Logged source probe baseline at step {args.step} from {event_file}")
    for message in messages:
        print(message)


if __name__ == "__main__":
    main()
