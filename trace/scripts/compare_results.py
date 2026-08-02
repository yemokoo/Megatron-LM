#!/usr/bin/env python3
"""Validate and compare continual-learning result matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    tasks = value["tasks"]
    matrix = value["score_matrix"]
    if len(matrix) != len(tasks) or any(len(row) != len(tasks) for row in matrix):
        raise ValueError("score_matrix must be square and match tasks")
    for trained, row in enumerate(matrix):
        for evaluated, score in enumerate(row):
            if evaluated > trained and score is not None:
                raise ValueError("upper triangle must be null before a task is learned")
    return value


def summarize(value: dict) -> dict:
    tasks = value["tasks"]
    matrix = value["score_matrix"]
    final = matrix[-1]
    if any(score is None for score in final):
        raise ValueError("final score row must contain every task")
    final_average = sum(final) / len(final)
    per_task_forgetting = {}
    for task_index, task in enumerate(tasks[:-1]):
        learned = matrix[task_index][task_index]
        later = [matrix[row][task_index] for row in range(task_index + 1, len(tasks))]
        if learned is None or any(score is None for score in later):
            raise ValueError(f"missing trajectory for {task}")
        per_task_forgetting[task] = sum(learned - score for score in later) / len(later)
    afr = sum(per_task_forgetting.values()) / len(per_task_forgetting)
    return {
        "run_id": value["run_id"],
        "provenance": value["provenance"],
        "final_per_task": dict(zip(tasks, final)),
        "final_average": final_average,
        "per_task_average_forgetting": per_task_forgetting,
        "average_forgetting_rate": afr,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summaries = [summarize(load(path)) for path in args.results]
    payload = {"schema_version": 1, "runs": summaries}
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        print(args.output)
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
