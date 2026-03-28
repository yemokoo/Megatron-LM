#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import wandb


PROBE_RE = re.compile(
    r"probe\s+(?P<name>\S+)\s+at iteration\s+(?P<iteration>\d+).*?"
    r"next_token_acc:\s+(?P<accuracy>[0-9.]+)\s+\|\s+ppl:\s+(?P<ppl>[0-9.E+-]+)"
)


def parse_mapping(values):
    mapping = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid mapping '{value}'. Use old_name=new_name.")
        old, new = value.split("=", 1)
        old = old.strip()
        new = new.strip()
        if not old or not new:
            raise ValueError(f"Invalid mapping '{value}'. Use old_name=new_name.")
        mapping[old] = new
    return mapping


def resolve_log_path(run_dir: Path | None, log_path: Path | None) -> Path:
    if log_path is not None:
        return log_path
    if run_dir is None:
        raise ValueError("Either --run-dir or --log-path is required.")
    logs_dir = run_dir / "logs"
    if not logs_dir.is_dir():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")
    candidates = sorted(logs_dir.glob("*.log"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .log file under {logs_dir}, found {len(candidates)}: "
            + ", ".join(str(path.name) for path in candidates)
        )
    return candidates[0]


def collect_points(log_path: Path, mapping: dict[str, str]):
    rows = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = PROBE_RE.search(line)
            if match is None:
                continue
            source_name = match.group("name")
            target_name = mapping.get(source_name)
            if target_name is None:
                continue
            rows.append(
                {
                    "source_name": source_name,
                    "target_name": target_name,
                    "step": int(match.group("iteration")),
                    "accuracy": float(match.group("accuracy")),
                    "ppl": float(match.group("ppl")),
                }
            )
    if not rows:
        raise ValueError(f"No matching probe rows found in {log_path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Re-log probe metrics from an existing log to a new W&B run.")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", default="never", choices=["never", "allow", "must", "auto"])
    parser.add_argument(
        "--map",
        dest="mappings",
        action="append",
        required=True,
        help="Probe mapping in old_name=new_name form. Repeatable.",
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
    log_path = resolve_log_path(args.run_dir, args.log_path)
    rows = collect_points(log_path, mapping)

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        id=args.run_id,
        resume=args.resume,
        tags=args.tags,
        config={
            "source_log_path": str(log_path),
            "probe_mapping": mapping,
            "relogged_only_probe_metrics": True,
        },
    )
    try:
        for row in rows:
            wandb.log(
                {
                    f"{row['target_name']}/next_token_accuracy": row["accuracy"],
                    f"{row['target_name']}/ppl": row["ppl"],
                    "relog/source_probe_name": row["source_name"],
                },
                step=row["step"],
            )
    finally:
        run.finish()

    print(f"Re-logged {len(rows)} probe rows from {log_path} to W&B run '{args.run_name}'.")


if __name__ == "__main__":
    main()
