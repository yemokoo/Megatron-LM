#!/usr/bin/env python3
"""Clone the metric history of a completed W&B run into another project."""

import argparse
from pathlib import Path

import wandb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True)
    parser.add_argument("--source-project", required=True)
    parser.add_argument("--target-project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--target-run-id")
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--save-dir", type=Path)
    args = parser.parse_args()

    api = wandb.Api()
    source = api.run(f"{args.entity}/{args.source_project}/{args.run_id}")
    if source.state == "running":
        raise SystemExit(f"Refusing to clone a running run: {source.path}")

    target_run_id = args.target_run_id or args.run_id
    init_kwargs = dict(
        entity=args.entity,
        project=args.target_project,
        id=target_run_id,
        name=source.name,
        config=dict(source.config),
        resume="never",
        mode="online",
        tags=list(source.tags or []) + [f"cloned-from:{args.source_project}"],
        settings=wandb.Settings(init_timeout=300),
    )
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        init_kwargs["dir"] = str(args.save_dir)
    target = wandb.init(**init_kwargs)
    count = 0
    try:
        for row in source.scan_history(page_size=1000):
            step = row.pop("_step", None)
            row.pop("_runtime", None)
            row.pop("_timestamp", None)
            payload = {
                key: value
                for key, value in row.items()
                if value is not None and not key.startswith("_")
            }
            if step is None or not payload:
                continue
            target.log(payload, step=int(step) + args.step_offset)
            count += 1
    finally:
        target.finish()
    print(
        f"Cloned {count} history rows to "
        f"{args.entity}/{args.target_project}/{target_run_id} "
        f"with step_offset={args.step_offset}"
    )


if __name__ == "__main__":
    main()
