#!/usr/bin/env python3
"""Print and assert the batch/cardinality contract before a real task run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--expected-samples", required=True, type=int)
    parser.add_argument("--micro-batch", required=True, type=int)
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--gradient-accumulation", required=True, type=int)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--logging-steps", type=int)
    args = parser.parse_args()
    with args.train_json.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise SystemExit("train JSON root must be a list")
    if len(records) != args.expected_samples:
        raise SystemExit(
            f"{args.task}: expected {args.expected_samples} records, got {len(records)}"
        )
    if min(args.micro_batch, args.world_size, args.gradient_accumulation, args.epochs) < 1:
        raise SystemExit("batch, world-size, accumulation and epochs must be positive")
    examples_per_micro_step = args.micro_batch * args.world_size
    batches = math.ceil(len(records) / examples_per_micro_step)
    steps_per_epoch = math.ceil(batches / args.gradient_accumulation)
    payload = {
        "task": args.task,
        "loaded_examples": len(records),
        "dataloader_batches_per_epoch": batches,
        "per_device_micro_batch": args.micro_batch,
        "world_size": args.world_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "effective_global_batch": examples_per_micro_step * args.gradient_accumulation,
        "optimizer_steps_per_epoch": steps_per_epoch,
        "epochs": args.epochs,
        "total_optimizer_steps": steps_per_epoch * args.epochs,
        "logging_steps": args.logging_steps,
        "logging_steps_is_batch_size": False,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
