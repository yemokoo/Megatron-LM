#!/usr/bin/env python3
"""Inspect one W&B run and nearby project runs without mutating them."""

from __future__ import annotations

import argparse
import json

import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    api = wandb.Api(timeout=60)
    run = api.run(f"{args.entity}/{args.project}/{args.run_id}")
    print("TARGET")
    print(json.dumps({
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "url": run.url,
        "group": run.group,
        "job_type": run.job_type,
        "tags": list(run.tags),
        "created_at": run.created_at,
        "heartbeat_at": getattr(run, "heartbeat_at", None),
        "logical_first_step": run.summary.get("logical_first_step"),
        "logical_last_step": run.summary.get("logical_last_step"),
        "summary_step": run.summary.get("_step"),
        "config_logical_offset": run.config.get("logical_step_offset"),
        "metric_definitions": run._attrs.get("metricDefinitions"),
        "history_keys": run._attrs.get("historyKeys"),
    }, indent=2, default=str))

    print("RECENT_PROJECT_RUNS")
    rows = []
    for index, other in enumerate(api.runs(f"{args.entity}/{args.project}", order="-created_at")):
        if index >= args.limit:
            break
        rows.append({
            "id": other.id,
            "name": other.name,
            "state": other.state,
            "group": other.group,
            "job_type": other.job_type,
            "tags": list(other.tags),
            "created_at": other.created_at,
            "summary_step": other.summary.get("_step"),
        })
    print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
