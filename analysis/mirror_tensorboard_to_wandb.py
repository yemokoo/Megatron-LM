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


def find_event_file(run_dir: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path
    candidates = sorted(run_dir.glob("events.out.tfevents*"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one events.out.tfevents* under {run_dir}, found {len(candidates)}"
        )
    return candidates[0]


def load_all_scalars(event_file: Path, mapping: dict[str, str]):
    acc = event_accumulator.EventAccumulator(
        str(event_file),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    if not tags:
        raise ValueError(f"No scalar tags found in {event_file}")

    by_step = defaultdict(dict)
    for tag in tags:
        out_tag = remap_tag(tag, mapping)
        for scalar_event in acc.Scalars(tag):
            by_step[int(scalar_event.step)][out_tag] = float(scalar_event.value)
    return by_step


def main():
    parser = argparse.ArgumentParser(description="Mirror TensorBoard scalar logs to a new W&B run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--event-file", type=Path)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", default="never", choices=["never", "allow", "must", "auto"])
    parser.add_argument(
        "--map",
        dest="mappings",
        action="append",
        default=[],
        help="Tag prefix mapping in old_prefix=new_prefix form. Repeatable.",
    )
    parser.add_argument(
        "--tag",
        dest="tags",
        action="append",
        default=[],
        help="Optional W&B tag. Repeatable.",
    )
    args = parser.parse_args()

    mapping = parse_mapping(args.mappings)
    event_file = find_event_file(args.run_dir, args.event_file)
    by_step = load_all_scalars(event_file, mapping)

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        id=args.run_id,
        resume=args.resume,
        tags=args.tags,
        config={
            "source_run_dir": str(args.run_dir),
            "source_event_file": str(event_file),
            "tag_mapping": mapping,
            "mirrored_from_tensorboard": True,
        },
    )
    try:
        for step in sorted(by_step):
            wandb.log(by_step[step], step=step)
    finally:
        run.finish()

    print(f"Mirrored {len(by_step)} steps from {event_file} to W&B run '{args.run_name}'.")


if __name__ == "__main__":
    main()
