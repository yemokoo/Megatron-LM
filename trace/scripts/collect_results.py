#!/usr/bin/env python3
"""Collect SLoRA or TRACE evaluation artifacts into one CL score matrix."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path


TASKS = [
    "C-STANCE",
    "FOMC",
    "MeetingBank",
    "Py150",
    "ScienceQA",
    "NumGLUE-cm",
    "NumGLUE-ds",
    "20Minuten",
]

PRIMARY_METRIC = {
    "C-STANCE": "accuracy",
    "FOMC": "accuracy",
    "MeetingBank": "rouge-l",
    "Py150": "similarity",
    "ScienceQA": "accuracy",
    "NumGLUE-cm": "accuracy",
    "NumGLUE-ds": "accuracy",
    "20Minuten": "sari",
}

SLORA_METHOD_DIR = {
    "seq_lora": "seq",
    "slora_pre": "pre",
    "slora_post": "post",
}

PAPER_BASELINE_METHOD_DIR = {
    "loramoe": "loramoe",
    "ours_lora_moe_v1": "ours_lora_moe_v1",
    "ours_lora_moe_v2": "ours_lora_moe_v2",
    "ours_lora_moe_v2_new": "ours_lora_moe_v2_new",
    "ours_lora_moe_v2_new_top4": "ours_lora_moe_v2_new_top4",
    "ours_lora_moe_v2_5": "ours_lora_moe_v2_5",
    "ours_lora_moe_v3": "ours_lora_moe_v3",
    "ours_lora_moe_v3_new": "ours_lora_moe_v3_new",
    "ours_lora_moe_v3_new_top4": "ours_lora_moe_v3_new_top4",
}

TRACE_METHOD_DIR = {
    "ewc": "ewc_upstream",
    "lwf": "lwf_upstream",
}


def normalize_keys(metrics: dict) -> dict:
    return {str(key).lower(): value for key, value in metrics.items()}


def metric_points(task: str, metrics: dict) -> float:
    normalized = normalize_keys(metrics)
    key = PRIMARY_METRIC[task]
    if key not in normalized:
        raise KeyError(f"{task}: missing primary metric {key}; got {sorted(normalized)}")
    value = normalized[key]
    if isinstance(value, dict) and "sari" in value:
        value = value["sari"]
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    score = float(value)
    if key in {"accuracy", "rouge-l"} and abs(score) <= 1.0:
        score *= 100.0
    return score


def parse_slora_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    matches = re.findall(r"^In [^:]+:\s*(\{.*\})\s*$", text, flags=re.MULTILINE)
    if not matches:
        raise ValueError(f"No metric dictionary found in {path}")
    value = ast.literal_eval(matches[-1])
    if not isinstance(value, dict):
        raise TypeError(f"Metric payload is not a dictionary: {path}")
    return value


def required_cell(round_index: int, task_index: int, sparse_15: bool) -> bool:
    if sparse_15:
        return round_index == len(TASKS) or (
            round_index < len(TASKS) and task_index == round_index - 1
        )
    return task_index < round_index


def collect_slora(
    run_dir: Path,
    sparse_15: bool = False,
    allow_partial: bool = False,
) -> list[list[float | None]]:
    matrix: list[list[float | None]] = []
    for round_index in range(1, len(TASKS) + 1):
        row: list[float | None] = []
        for task_index, task in enumerate(TASKS):
            if not required_cell(round_index, task_index, sparse_15):
                row.append(None)
                continue
            try:
                metrics = parse_slora_log(
                    run_dir / "evaluation" / f"order{round_index}" / task / "eval.log"
                )
                row.append(metric_points(task, metrics))
            except (OSError, ValueError, KeyError, TypeError, SyntaxError):
                if not allow_partial:
                    raise
                row.append(None)
        matrix.append(row)
    return matrix


def collect_trace(run_dir: Path) -> list[list[float | None]]:
    matrix: list[list[float | None]] = []
    evaluation = run_dir / "evaluation"
    for round_index in range(len(TASKS)):
        row: list[float | None] = []
        for task_index, task in enumerate(TASKS):
            if task_index > round_index:
                row.append(None)
                continue
            path = evaluation / f"results-{round_index}-{task_index}-{task}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            row.append(metric_points(task, payload["eval"]))
        matrix.append(row)
    return matrix


def collect_paper_baseline(
    run_dir: Path,
    sparse_15: bool = False,
    allow_partial: bool = False,
) -> list[list[float | None]]:
    matrix: list[list[float | None]] = []
    for round_index in range(1, len(TASKS) + 1):
        row: list[float | None] = []
        for task_index, task in enumerate(TASKS):
            if not required_cell(round_index, task_index, sparse_15):
                row.append(None)
                continue
            path = (
                run_dir
                / "evaluation"
                / f"order{round_index}"
                / f"results-{task}.json"
            )
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                row.append(metric_points(task, payload["eval"]))
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                if not allow_partial:
                    raise
                row.append(None)
        matrix.append(row)
    return matrix


def git_identity(repo: Path) -> dict:
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    diff = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--binary", "HEAD"], text=False
    )
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    )
    untracked = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard"],
        text=True,
    ).splitlines()
    digest = hashlib.sha256(diff)
    for relative in sorted(untracked):
        path = repo / relative
        if not path.is_file():
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return {
        "head": head,
        "worktree_state_sha256": digest.hexdigest(),
        "dirty": bool(status),
    }


def resolve(root: Path, method: str, model: str) -> tuple[str, Path, Path]:
    if method == "slora_pre_released":
        family = "slora"
        run_dir = root / "results" / "full_runs_upstream_code" / model / "pre"
        repo = root / "implementations" / "SLoRA-upstream-port"
    elif method in SLORA_METHOD_DIR:
        family = "slora"
        run_dir = root / "results" / "full_runs" / model / SLORA_METHOD_DIR[method]
        repo = root / "implementations" / "SLoRA-repro"
    elif method in PAPER_BASELINE_METHOD_DIR:
        family = "paper_baseline"
        run_dir = (
            root / "results" / "full_runs" / model / PAPER_BASELINE_METHOD_DIR[method]
        )
        repo = root / "implementations" / "llmcl_benchmark"
    else:
        family = "trace"
        run_dir = root / "results" / "full_runs" / model / TRACE_METHOD_DIR.get(method, method)
        repo = root / "implementations" / "TRACE-repro"
    return family, run_dir, repo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--model", required=True, choices=["llama31", "qwen25_7b"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-dir", type=Path,
                        help="Override the method's conventional run directory.")
    parser.add_argument("--family", choices=["slora", "paper_baseline", "trace"],
                        help="Override result artifact family with --run-dir.")
    parser.add_argument(
        "--sparse-15",
        action="store_true",
        help="Require only rounds 1-7 diagonal cells and all eight round-8 cells.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=("Keep missing required cells as null and write diagnostic partial "
              "metrics instead of failing."),
    )
    args = parser.parse_args()

    family, run_dir, repo = resolve(args.root, args.method, args.model)
    if args.run_dir is not None:
        run_dir = args.run_dir
    if args.family is not None:
        family = args.family
    if family == "slora":
        matrix = collect_slora(run_dir, args.sparse_15, args.allow_partial)
    elif family == "paper_baseline":
        matrix = collect_paper_baseline(
            run_dir, args.sparse_15, args.allow_partial)
    else:
        matrix = collect_trace(run_dir)
    diagonal = [matrix[index][index] for index in range(len(TASKS) - 1)]
    final_scores = matrix[-1]
    missing_cells = [
        f"order{round_index}.{task}"
        for round_index in range(1, len(TASKS) + 1)
        for task_index, task in enumerate(TASKS)
        if required_cell(round_index, task_index, args.sparse_15)
        and matrix[round_index - 1][task_index] is None
    ]
    if missing_cells and not args.allow_partial:
        raise ValueError("Required diagonal/final score is missing; refusing summary")
    final_available = [float(value) for value in final_scores if value is not None]
    op = (
        sum(final_available) / len(final_available)
        if len(final_available) == len(final_scores) else None)
    bwt_terms = [
        (float(final_scores[index]) - float(diagonal[index])
         if final_scores[index] is not None and diagonal[index] is not None
         else None)
        for index in range(len(TASKS) - 1)
    ]
    bwt = (
        sum(float(value) for value in bwt_terms) / len(bwt_terms)
        if all(value is not None for value in bwt_terms) else None)
    available_bwt_terms = [float(value) for value in bwt_terms if value is not None]
    payload = {
        "schema_version": 1,
        "run_id": f"{args.model}-{args.method}",
        "provenance": {
            "classification": (
                "corrected"
                if args.method.endswith("_corrected")
                else (
                    "local-compatible-port"
                    if family == "paper_baseline"
                    else "public-code-port"
                )
            ),
            "family": family,
            "method": args.method,
            "model": args.model,
            "code": git_identity(repo),
            "run_dir": str(run_dir.resolve()),
            "data": "TRACE LLM-CL-Benchmark_5000; 5,000 train records per task",
        },
        "tasks": TASKS,
        "metric": PRIMARY_METRIC,
        "evaluation_mode": "sparse_15" if args.sparse_15 else "lower_triangle",
        "score_matrix": matrix,
        "diagonal_scores_rounds_1_to_7": diagonal,
        "final_scores_round_8": final_scores,
        "OP": op,
        "final_average": op,
        "BWT": bwt,
        "BWT_terms": bwt_terms,
        "complete": not missing_cells,
        "missing_cells": missing_cells,
        "available_final_average": (
            sum(final_available) / len(final_available)
            if final_available else None),
        "available_BWT": (
            sum(available_bwt_terms) / len(available_bwt_terms)
            if available_bwt_terms else None),
        "note": (
            None if not missing_cells else
            "Partial diagnostic: OP/BWT remain null until every required cell exists."),
        "formulas": {
            "OP": "mean(score_matrix[7][0:8])",
            "BWT": "mean(score_matrix[7][i] - score_matrix[i][i] for i=0..6)",
        },
    }
    output = args.output or (
        args.root / "results" / "summaries" / args.model / f"{args.method}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
