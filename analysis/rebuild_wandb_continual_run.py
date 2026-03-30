#!/usr/bin/env python3
import argparse
from collections import defaultdict
from pathlib import Path

import wandb
from tensorboard.backend.event_processing import event_accumulator


def parse_mapping(values):
    mapping = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid mapping '{value}'. Use old_prefix=new_prefix.")
        old, new = value.split("=", 1)
        old = old.strip()
        new = new.strip()
        if not old or not new:
            raise ValueError(f"Invalid mapping '{value}'. Use old_prefix=new_prefix.")
        mapping[old] = new
    return mapping


def remap_tag(tag, mapping):
    for old_prefix, new_prefix in mapping.items():
        if tag == old_prefix:
            return new_prefix
        if tag.startswith(old_prefix + "/"):
            return new_prefix + tag[len(old_prefix) :]
    return tag


def find_event_file(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("events.out.tfevents*"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one events.out.tfevents* under {run_dir}, found {len(candidates)}"
        )
    return candidates[0]


def load_scalars(event_file: Path, mapping: dict[str, str]):
    acc = event_accumulator.EventAccumulator(
        str(event_file),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    by_step = defaultdict(dict)
    for tag in tags:
        out_tag = remap_tag(tag, mapping)
        for scalar_event in acc.Scalars(tag):
            by_step[int(scalar_event.step)][out_tag] = float(scalar_event.value)
    return by_step


def load_baseline_values(event_file: Path, mapping: dict[str, str], step: int):
    acc = event_accumulator.EventAccumulator(
        str(event_file),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    baseline = {}
    for tag in tags:
        out_tag = remap_tag(tag, mapping)
        chosen = None
        for scalar_event in acc.Scalars(tag):
            event_step = int(scalar_event.step)
            if event_step <= step:
                chosen = scalar_event
            else:
                break
        if chosen is not None:
            baseline[out_tag] = float(chosen.value)
    return baseline


def load_scalars_up_to_step(event_file: Path, mapping: dict[str, str], max_step: int):
    acc = event_accumulator.EventAccumulator(
        str(event_file),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    by_step = defaultdict(dict)
    for tag in tags:
        out_tag = remap_tag(tag, mapping)
        for scalar_event in acc.Scalars(tag):
            step = int(scalar_event.step)
            if step > max_step:
                break
            by_step[step][out_tag] = float(scalar_event.value)
    return by_step


def should_replace(tag: str, replace_prefixes: list[str]) -> bool:
    for prefix in replace_prefixes:
        if tag == prefix or tag.startswith(prefix + "/"):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild a continual W&B run from logs: use source-model baseline at a fixed step "
            "and continual-run history after that step."
        )
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--continual-run-dir", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", default="never", choices=["never", "allow", "must", "auto"])
    parser.add_argument("--baseline-step", type=int, default=1800)
    parser.add_argument(
        "--include-source-history-through-step",
        action="store_true",
        help=(
            "Include the full source-run history up to --baseline-step, then append "
            "continual-run history after that step."
        ),
    )
    parser.add_argument(
        "--source-map",
        dest="source_mappings",
        action="append",
        default=[],
        help="Source tag prefix mapping in old_prefix=new_prefix form. Repeatable.",
    )
    parser.add_argument(
        "--continual-map",
        dest="continual_mappings",
        action="append",
        default=[],
        help="Continual tag prefix mapping in old_prefix=new_prefix form. Repeatable.",
    )
    parser.add_argument(
        "--replace-prefix",
        dest="replace_prefixes",
        action="append",
        default=[],
        help=(
            "Metric prefix to replace at baseline step with source values. "
            "Repeatable. Example: wiki_probe"
        ),
    )
    parser.add_argument(
        "--tag",
        dest="tags",
        action="append",
        default=[],
        help="Optional W&B tag. Repeatable.",
    )
    args = parser.parse_args()

    source_mapping = parse_mapping(args.source_mappings)
    continual_mapping = parse_mapping(args.continual_mappings)

    source_event = find_event_file(args.source_run_dir)
    continual_event = find_event_file(args.continual_run_dir)

    rebuilt = defaultdict(dict)

    if args.include_source_history_through_step:
        source_by_step = load_scalars_up_to_step(source_event, source_mapping, args.baseline_step)
        for step, payload in source_by_step.items():
            rebuilt[step].update(payload)
    else:
        source_baseline = load_baseline_values(source_event, source_mapping, args.baseline_step)

    continual_by_step = load_scalars(continual_event, continual_mapping)
    for step, payload in continual_by_step.items():
        if step < args.baseline_step:
            continue
        rebuilt[step].update(payload)

    if args.replace_prefixes and not args.include_source_history_through_step:
        baseline_payload = {}
        for tag, value in source_baseline.items():
            if should_replace(tag, args.replace_prefixes):
                baseline_payload[tag] = value
        if not baseline_payload:
            raise ValueError(
                f"No source baseline tags matched replace prefixes {args.replace_prefixes}"
            )
        for tag in list(rebuilt[args.baseline_step].keys()):
            if should_replace(tag, args.replace_prefixes):
                del rebuilt[args.baseline_step][tag]
        rebuilt[args.baseline_step].update(baseline_payload)

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        id=args.run_id,
        resume=args.resume,
        tags=args.tags,
        config={
            "source_run_dir": str(args.source_run_dir),
            "continual_run_dir": str(args.continual_run_dir),
            "source_event_file": str(source_event),
            "continual_event_file": str(continual_event),
            "baseline_step": args.baseline_step,
            "include_source_history_through_step": args.include_source_history_through_step,
            "source_mapping": source_mapping,
            "continual_mapping": continual_mapping,
            "replace_prefixes": args.replace_prefixes,
            "rebuilt_from_existing_logs": True,
        },
    )
    try:
        for step in sorted(rebuilt):
            wandb.log(rebuilt[step], step=step)
    finally:
        run.finish()

    print(
        f"Rebuilt W&B run '{args.run_name}' using source baseline from {source_event} "
        f"and continual history from {continual_event}."
    )


if __name__ == "__main__":
    main()
