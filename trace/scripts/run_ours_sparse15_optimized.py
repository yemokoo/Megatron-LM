#!/usr/bin/env python3
"""Resource-aware Ours TRACE evaluator.

Unlike ``run_ours_sparse15_efficient.py``, every sample shard is an independent
queue job.  A GPU is therefore returned as soon as its shard exits instead of
being held until the other three members of a 4-way helper finish.

The generation/scoring contract is intentionally unchanged:

* max prompt length is unlimited;
* max_new_tokens is 1024 for every task;
* task-specific generation limits are disabled;
* Llama-3 chat formatting and greedy decoding are used;
* shard outputs are merged in original ``sample_indices`` order and rescored.

This is a new opt-in runner.  Existing/live evaluators do not import it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO


ROOT = Path(__file__).resolve().parents[1]
LLMCL = ROOT / "implementations/llmcl_benchmark"
DEFAULT_DATA = ROOT / "data/trace"
TASKS = [
    "C-STANCE", "FOMC", "MeetingBank", "Py150",
    "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten",
]
CANONICAL_TASKS = list(TASKS)
DEFAULT_SHARDED_TASKS = {"MeetingBank", "Py150", "ScienceQA"}
PROGRESS_RE = re.compile(
    r"(?P<pct>\d{1,3})%\|[^\r\n]*?\|\s*(?P<done>\d+)/(?P<total>\d+)")


@dataclass(frozen=True)
class Cell:
    round_id: int
    task: str

    @property
    def label(self) -> str:
        return f"order{self.round_id}.{self.task}"


@dataclass(frozen=True)
class Job:
    cell: Cell
    shard_id: int
    num_shards: int

    @property
    def is_shard(self) -> bool:
        return self.num_shards > 1

    @property
    def label(self) -> str:
        if not self.is_shard:
            return self.cell.label
        return (
            f"{self.cell.label}.shard{self.shard_id + 1}-of-"
            f"{self.num_shards}")

    @property
    def suffix(self) -> str:
        if not self.is_shard:
            return ""
        return f".shard{self.shard_id}-of-{self.num_shards}"


@dataclass
class ActiveJob:
    process: subprocess.Popen[bytes]
    job: Job
    gpu: int
    log: IO[bytes]
    log_path: Path
    started_at: float


def sparse15_cells() -> list[Cell]:
    """Seven acquisition cells plus all eight final cells."""
    diagonal = [Cell(index + 1, TASKS[index]) for index in range(7)]
    final = [Cell(8, task) for task in TASKS]
    return diagonal + final


def lower_triangle_cells() -> list[Cell]:
    """All 36 cells at or below the continual-learning diagonal."""
    return [
        Cell(round_id, TASKS[task_index])
        for round_id in range(1, len(TASKS) + 1)
        for task_index in range(round_id)
    ]


def diagonal7_cells() -> list[Cell]:
    """The seven acquisition cells only, no final row -- for a run whose
    final row is scored separately (a post-hoc residual-expert arm)."""
    return sparse15_cells()[:7]


def cells_for_mode(mode: str) -> list[Cell]:
    if mode == "sparse15":
        return sparse15_cells()
    if mode == "diagonal7":
        return diagonal7_cells()
    if mode == "lower_triangle":
        return lower_triangle_cells()
    raise ValueError(f"unknown matrix mode: {mode}")


def expected_indices(total: int, shard_id: int, num_shards: int) -> list[int]:
    return list(range(shard_id, total, num_shards))


def result_path(run_dir: Path, job: Job) -> Path:
    return (
        run_dir / "evaluation" / f"order{job.cell.round_id}"
        / f"results-{job.cell.task}{job.suffix}.json")


def expected_count(data_root: Path, task: str) -> int:
    return len(json.loads((data_root / task / "test.json").read_text(
        encoding="utf-8")))


def result_is_complete(
    path: Path,
    total: int,
    shard_id: int = 0,
    num_shards: int = 1,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    expected = expected_indices(total, shard_id, num_shards)
    results = payload.get("results")
    indices = payload.get("sample_indices")
    # Older complete unsharded results did not always carry sample_indices.
    if num_shards == 1 and indices is None:
        return isinstance(results, list) and len(results) == total
    return indices == expected and isinstance(results, list) and len(results) == len(expected)


def parse_latest_progress(text: str) -> tuple[int, int, int] | None:
    matches = list(PROGRESS_RE.finditer(text.replace("\r", "\n")))
    if not matches:
        return None
    match = matches[-1]
    return (
        int(match.group("pct")),
        int(match.group("done")),
        int(match.group("total")),
    )


def read_latest_progress(path: Path, tail_bytes: int = 512 * 1024):
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - tail_bytes))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    return parse_latest_progress(text)


def parse_run_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw_line or raw_line.lstrip().startswith("#"):
            continue
        key, value = raw_line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def task_batch(task: str, args: argparse.Namespace) -> int:
    return {
        "ScienceQA": args.scienceqa_batch,
        "MeetingBank": args.meetingbank_batch,
        "Py150": args.py150_batch,
        # A small batch prevents one rare 1024-token continuation from keeping
        # 31 already-finished rows alive for the entire decode.  This does not
        # change max_new_tokens or scoring.
        "20Minuten": args.twenty_minuten_batch,
    }.get(task, args.eval_batch)


def make_jobs(
    run_dir: Path,
    data_root: Path,
    sharded_tasks: set[str],
    num_shards: int,
    cells: list[Cell] | None = None,
) -> tuple[list[Job], list[Cell]]:
    jobs: list[Job] = []
    complete_cells: list[Cell] = []
    for cell in cells or sparse15_cells():
        total = expected_count(data_root, cell.task)
        merged = run_dir / "evaluation" / f"order{cell.round_id}" / f"results-{cell.task}.json"
        if result_is_complete(merged, total):
            complete_cells.append(cell)
            continue
        count = num_shards if cell.task in sharded_tasks else 1
        for shard_id in range(count):
            job = Job(cell, shard_id, count)
            if not result_is_complete(
                result_path(run_dir, job), total, shard_id, count):
                jobs.append(job)
    return jobs, complete_cells


def checkpoint_dir_for(
    cell: Cell, run_dir: Path, diagonal_suffix: str = "") -> Path:
    """Resolve the checkpoint a cell is scored from.

    The plain round directory is the model the next task continued from. The
    2-phase ablation arm also writes a pre-router-retune checkpoint per round;
    scoring the diagonal (acquisition) cells there measures plasticity before
    the router correction, while the final row stays on the continued model.
    """
    name = str(cell.round_id - 1)
    if diagonal_suffix and TASKS.index(cell.task) == cell.round_id - 1:
        name += diagonal_suffix
    return run_dir / name


def build_eval_command(
    job: Job,
    run_dir: Path,
    model_path: Path,
    data_root: Path,
    output_dir: Path,
    batch: int,
    python: Path,
    trace_generation_stops: bool = False,
    diagonal_suffix: str = "",
) -> list[str]:
    command = [
        str(python), "-u", "evaluate_Ours_LoRA_MoE.py",
        "--checkpoint_dir", str(
            checkpoint_dir_for(job.cell, run_dir, diagonal_suffix)),
        "--base_model_name_or_path", str(model_path),
        "--data_path", str(data_root),
        "--inference_tasks", job.cell.task,
        "--inference_output_path", str(output_dir),
        "--summary_filename", (
            f"{job.cell.task}{job.suffix}.summary.json"),
        "--max_prompt_len", "0",
        "--max_ans_len", "1024",
        "--no-task_generation_limits",
        "--slora_conv_mode", os.environ.get("SPARSE15_CONV_MODE", "llama3"),
        "--per_device_eval_batch_size", str(batch),
        "--temperature", "0",
    ]
    if job.is_shard:
        command.extend([
            "--num_sample_shards", str(job.num_shards),
            "--sample_shard_id", str(job.shard_id),
            "--result_suffix", job.suffix,
        ])
    if trace_generation_stops:
        command.append("--trace_generation_stops")
    # extra evaluator flags, e.g. the bos_guard switches for header-guarded checkpoints
    extra = os.environ.get("SPARSE15_EVAL_EXTRA_ARGS", "").strip()
    if extra:
        command.extend(shlex.split(extra))
    return command


def merge_command(
    cell: Cell,
    run_dir: Path,
    num_shards: int,
    python: Path,
) -> list[str]:
    output_dir = run_dir / "evaluation" / f"order{cell.round_id}"
    command = [
        str(python), "-u",
        str(LLMCL / "scripts/merge_trace_shards.py"),
        "--input_dir", str(output_dir),
        "--task", cell.task,
        "--num_shards", str(num_shards),
        "--summary_filename", f"{cell.task}.summary.json",
    ]
    if cell.task == "20Minuten":
        command.append("--with_sari")
    return command


def all_shards_complete(
    cell: Cell,
    run_dir: Path,
    data_root: Path,
    num_shards: int,
) -> bool:
    total = expected_count(data_root, cell.task)
    return all(
        result_is_complete(
            result_path(run_dir, Job(cell, shard_id, num_shards)),
            total,
            shard_id,
            num_shards,
        )
        for shard_id in range(num_shards)
    )


PRIMARY_METRICS = {
    "C-STANCE": ("accuracy", 100.0),
    "FOMC": ("accuracy", 100.0),
    "MeetingBank": ("rouge-L", 100.0),
    "Py150": ("similarity", 1.0),
    "ScienceQA": ("accuracy", 100.0),
    "NumGLUE-cm": ("accuracy", 100.0),
    "NumGLUE-ds": ("accuracy", 100.0),
    "20Minuten": ("sari", 1.0),
}


def read_primary_score(run_dir: Path, cell: Cell) -> float:
    path = (
        run_dir / "evaluation" / f"order{cell.round_id}"
        / f"results-{cell.task}.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    key, scale = PRIMARY_METRICS[cell.task]
    value = payload.get("eval", {}).get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"missing numeric eval.{key} in {path}")
    return float(value) * scale


def write_lower_triangle_summary(
    run_dir: Path,
    cells: list[Cell],
    method: str,
) -> Path:
    size = len(TASKS)
    matrix: list[list[float | None]] = [
        [None for _ in range(size)] for _ in range(size)]
    for cell in cells:
        task_index = TASKS.index(cell.task)
        matrix[cell.round_id - 1][task_index] = read_primary_score(
            run_dir, cell)
    diagonal = [matrix[index][index] for index in range(size)]
    final = matrix[-1]
    if any(value is None for value in diagonal + final):
        raise ValueError("lower-triangle summary is missing diagonal/final scores")
    diagonal_values = [float(value) for value in diagonal]
    final_values = [float(value) for value in final]
    forgetting_by_task = [
        diagonal_values[index] - final_values[index]
        for index in range(size - 1)
    ]
    payload = {
        "method": method,
        "tasks": TASKS,
        "matrix_mode": "lower_triangle",
        "score_matrix": matrix,
        "round_averages": [
            sum(float(value) for value in row if value is not None)
            / sum(value is not None for value in row)
            for row in matrix
        ],
        "diagonal": diagonal_values,
        "final": final_values,
        "average_accuracy": sum(final_values) / size,
        "forgetting_by_task": forgetting_by_task,
        "average_forgetting": sum(forgetting_by_task) / len(forgetting_by_task),
    }
    output = run_dir / "lower_triangle_summary.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"[COLLECT] wrote {output}", flush=True)
    return output


def write_partial_lower_triangle_summary(
    run_dir: Path,
    cells: list[Cell],
    method: str,
    failed_jobs: list[dict[str, object]],
) -> Path:
    """Write a non-authoritative matrix without inventing missing scores."""
    size = len(TASKS)
    matrix: list[list[float | None]] = [
        [None for _ in range(size)] for _ in range(size)]
    missing_cells: list[str] = []
    for cell in cells:
        try:
            matrix[cell.round_id - 1][TASKS.index(cell.task)] = (
                read_primary_score(run_dir, cell))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            missing_cells.append(cell.label)
    payload = {
        "schema_version": 1,
        "complete": not missing_cells,
        "authoritative_cl_metrics_available": False,
        "method": method,
        "tasks": TASKS,
        "matrix_mode": "lower_triangle",
        "score_matrix": matrix,
        "round_averages_available": [
            (sum(float(value) for value in row if value is not None)
             / sum(value is not None for value in row))
            if any(value is not None for value in row) else None
            for row in matrix
        ],
        "missing_cells": missing_cells,
        "failed_jobs": failed_jobs,
        "note": (
            "Partial diagnostic only. AA/forgetting are intentionally omitted "
            "because one or more required evaluation cells are missing."),
    }
    output = run_dir / "lower_triangle_partial_summary.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"[COLLECT PARTIAL] wrote {output}", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", default=os.environ.get("OURS_LORAMOE_OUTPUT_ROOT"),
        help="Training run root. Defaults to OURS_LORAMOE_OUTPUT_ROOT.")
    parser.add_argument(
        "--method", default=os.environ.get("SPARSE15_METHODS"),
        help="Method label passed to collect_results.py.")
    parser.add_argument(
        "--matrix-mode", choices=("sparse15", "diagonal7", "lower_triangle"),
        default=os.environ.get("TRACE_MATRIX_MODE", "sparse15"),
        help="Evaluate the publication sparse-15 cells or all 36 lower-"
             "triangular cells.")
    parser.add_argument(
        "--gpus", default=os.environ.get("SPARSE15_GPUS", "0,1,2,3"))
    parser.add_argument(
        "--data-root", default=os.environ.get("TRACE_DATA_ROOT", str(DEFAULT_DATA)))
    parser.add_argument(
        "--model-path", default=os.environ.get("SLORA_LLAMA31_PATH"),
        help="Defaults to model_path in RUN_DIR/run.env.")
    parser.add_argument(
        "--python", default=os.environ.get(
            "TRACE_PYTHON", str(ROOT / ".venv-runtime/bin/python")))
    parser.add_argument(
        "--sharded-tasks",
        default=os.environ.get(
            "SPARSE15_SHARDED_TASKS", ",".join(sorted(DEFAULT_SHARDED_TASKS))))
    parser.add_argument(
        "--num-shards", type=int,
        default=int(os.environ.get("SPARSE15_NUM_SAMPLE_SHARDS", "0")),
        help="Default 0 means one shard per configured GPU.")
    parser.add_argument(
        "--eval-batch", type=int,
        default=int(os.environ.get("SPARSE15_EVAL_BATCH", "32")))
    parser.add_argument(
        "--scienceqa-batch", type=int,
        default=int(os.environ.get("SPARSE15_SCIENCEQA_BATCH", "128")))
    parser.add_argument(
        "--meetingbank-batch", type=int,
        default=int(os.environ.get("SPARSE15_MEETINGBANK_BATCH", "1")))
    parser.add_argument(
        "--py150-batch", type=int,
        default=int(os.environ.get("SPARSE15_PY150_BATCH", "8")))
    parser.add_argument(
        "--twenty-minuten-batch", type=int,
        default=int(os.environ.get("SPARSE15_20MINUTEN_BATCH", "8")))
    parser.add_argument(
        "--status-interval", type=float,
        default=float(os.environ.get("SPARSE15_STATUS_INTERVAL", "60")))
    parser.add_argument(
        "--trace-generation-stops",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("SPARSE15_EXACT_STOP_MARKERS", "0") == "1",
        help="Stop each row at TRACE's score-normalization marker. This keeps "
             "the normalized scored answer unchanged while avoiding discarded "
             "continuation tokens.")
    parser.add_argument(
        "--status-file", default=None,
        help="Default RUN_DIR/evaluation/optimized_status.tsv.")
    parser.add_argument(
        "--diagonal-checkpoint-suffix",
        default=os.environ.get("SPARSE15_DIAGONAL_CKPT_SUFFIX", ""),
        help="Score diagonal (acquisition) cells from "
             "RUN_DIR/<round><suffix> instead of RUN_DIR/<round>. Use "
             "_prephase2 for the 2-phase ablation arm.")
    parser.add_argument(
        "--task-order",
        default=os.environ.get("SPARSE15_TASK_ORDER", ""),
        help=("Comma-separated training order, when the run did not use the "
              "canonical TRACE sequence (e.g. the reversed-order HP "
              "sensitivity cell).  Round r's diagonal cell and the checkpoint "
              "index both follow this order.  Must be a permutation of the "
              "eight TRACE tasks."),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-collect", action="store_true")
    parser.add_argument(
        "--continue-on-cell-error",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("SPARSE15_CONTINUE_ON_CELL_ERROR", "0") == "1",
        help=("Record a failed (round, task) cell, skip its remaining shards, "
              "and continue evaluating later cells."),
    )
    return parser.parse_args()


def set_task_order(order: list) -> None:
    """Re-point the module-level TASKS at a different training order.

    Every cell/checkpoint/score helper reads the module global, so mutating it
    in place is enough; replacing the binding would leave stale references.
    """
    if sorted(order) != sorted(CANONICAL_TASKS):
        raise SystemExit(
            f"--task-order must be a permutation of {CANONICAL_TASKS}, got {order}")
    TASKS[:] = order


def main() -> int:
    args = parse_args()
    if args.task_order:
        set_task_order([item.strip() for item in args.task_order.split(",") if item.strip()])
        print(f"[sparse15] task order: {' -> '.join(TASKS)}", flush=True)
    if not args.run_dir:
        raise SystemExit("--run-dir or OURS_LORAMOE_OUTPUT_ROOT is required")
    if not args.method:
        raise SystemExit("--method or SPARSE15_METHODS is required")
    run_dir = Path(args.run_dir).resolve()
    data_root = Path(args.data_root).resolve()
    # Do not resolve the venv launcher symlink to its base interpreter.  Calling
    # the resolved target bypasses pyvenv.cfg and can silently import an older
    # site-packages tree (notably a Transformers without StopStringCriteria).
    python = Path(args.python).absolute()
    gpus = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("--gpus must contain distinct GPU ids")
    num_shards = args.num_shards or len(gpus)
    if num_shards < 1:
        raise SystemExit("--num-shards must be positive")
    sharded_tasks = {
        item.strip() for item in args.sharded_tasks.split(",") if item.strip()}
    unknown = sharded_tasks - set(TASKS)
    if unknown:
        raise SystemExit(f"unknown sharded tasks: {sorted(unknown)}")
    cells = cells_for_mode(args.matrix_mode)
    run_env = parse_run_env(run_dir / "run.env")
    model_value = args.model_path or run_env.get("model_path")
    if not model_value:
        raise SystemExit(
            "--model-path/SLORA_LLAMA31_PATH missing and run.env has no model_path")
    model_path = Path(model_value).resolve()
    default_status_name = (
        "optimized_status.tsv" if args.matrix_mode in ("sparse15", "diagonal7")
        else "optimized_lower_triangle_status.tsv")
    status_path = Path(args.status_file).resolve() if args.status_file else (
        run_dir / "evaluation" / default_status_name)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    jobs, complete_cells = make_jobs(
        run_dir, data_root, sharded_tasks, num_shards, cells)
    print(
        f"[OPT-QUEUE] run={run_dir} method={args.method} "
        f"matrix_mode={args.matrix_mode} cells={len(cells)} gpus={gpus} "
        f"protocol=max_new1024/no_task_limits/{os.environ.get('SPARSE15_CONV_MODE', 'llama3')}/greedy "
        f"exact_stop_markers={int(args.trace_generation_stops)} "
        f"sharded={sorted(sharded_tasks)} shards={num_shards} "
        f"complete_cells={len(complete_cells)} pending_jobs={len(jobs)}",
        flush=True,
    )

    # If all shard files survived a prior interruption, merge them before
    # deciding the queue is done.
    for cell in cells:
        if cell.task not in sharded_tasks:
            continue
        merged = run_dir / "evaluation" / f"order{cell.round_id}" / f"results-{cell.task}.json"
        if not result_is_complete(merged, expected_count(data_root, cell.task)):
            if all_shards_complete(cell, run_dir, data_root, num_shards):
                command = merge_command(cell, run_dir, num_shards, python)
                print(f"[MERGE READY] {cell.label}: {shlex.join(command)}", flush=True)
                if not args.dry_run:
                    subprocess.run(command, cwd=ROOT, check=True)

    # Rebuild after interruption-time merges so no stale pending jobs remain.
    jobs, complete_cells = make_jobs(
        run_dir, data_root, sharded_tasks, num_shards, cells)
    if args.dry_run:
        for job in jobs:
            output_dir = run_dir / "evaluation" / f"order{job.cell.round_id}"
            command = build_eval_command(
                job, run_dir, model_path, data_root, output_dir,
                task_batch(job.cell.task, args), python,
                args.trace_generation_stops,
                args.diagonal_checkpoint_suffix)
            print(f"[DRY-RUN] {job.label}: {shlex.join(command)}", flush=True)
        print(f"[DRY-RUN] {len(jobs)} GPU jobs", flush=True)
        return 0

    for cell in cells:
        checkpoint = (
            checkpoint_dir_for(cell, run_dir, args.diagonal_checkpoint_suffix)
            / "lora_moe_meta.json")
        if not checkpoint.is_file():
            raise SystemExit(f"missing checkpoint metadata: {checkpoint}")

    thread_count = os.environ.get("SPARSE15_CPU_THREADS", "4")
    common_env = {
        "OMP_NUM_THREADS": thread_count,
        "MKL_NUM_THREADS": thread_count,
        "OPENBLAS_NUM_THREADS": thread_count,
        "NUMEXPR_NUM_THREADS": thread_count,
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # os.environ is copied for every child as well, but keeping this value
        # explicit makes the queue contract visible and prevents a later
        # environment-filtering refactor from silently dropping it.
        "TORCHINDUCTOR_COMPILE_THREADS": os.environ.get(
            "TORCHINDUCTOR_COMPILE_THREADS", "4"),
    }
    status_new = not status_path.exists()
    status_handle = status_path.open("a", encoding="utf-8", buffering=1)
    if status_new:
        status_handle.write("time\tevent\tgpu\tjob\tprogress\tpending\n")

    pending = list(jobs)
    free = sorted(gpus)
    active: dict[int, ActiveJob] = {}
    completed_jobs = 0
    merged_cells: set[Cell] = set(complete_cells)
    failed_cells: set[Cell] = set()
    failed_jobs: list[dict[str, object]] = []
    last_status = 0.0

    def emit(event: str, gpu: str, label: str, progress: str = "-") -> None:
        stamp = time.strftime("%F %T")
        line = f"{stamp}\t{event}\t{gpu}\t{label}\t{progress}\t{len(pending)}"
        status_handle.write(line + "\n")

    def stop_active() -> None:
        for current in active.values():
            if current.process.poll() is None:
                try:
                    os.killpg(current.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            current.log.close()

    try:
        while pending or active:
            while pending and free:
                gpu = free.pop(0)
                job = pending.pop(0)
                if job.cell in failed_cells:
                    free.append(gpu)
                    free.sort()
                    continue
                output_dir = run_dir / "evaluation" / f"order{job.cell.round_id}"
                output_dir.mkdir(parents=True, exist_ok=True)
                log_path = output_dir / f"{job.cell.task}{job.suffix}.optimized.log"
                command = build_eval_command(
                    job, run_dir, model_path, data_root, output_dir,
                    task_batch(job.cell.task, args), python,
                    args.trace_generation_stops,
                    args.diagonal_checkpoint_suffix)
                env = os.environ.copy()
                env.update(common_env)
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                log = log_path.open("ab", buffering=0)
                log.write(
                    f"\n[START] {time.strftime('%F %T')} physical_gpu={gpu} "
                    f"batch={task_batch(job.cell.task, args)} "
                    f"command={shlex.join(command)}\n".encode())
                process = subprocess.Popen(
                    command, cwd=LLMCL, env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                active[gpu] = ActiveJob(
                    process, job, gpu, log, log_path, time.monotonic())
                print(
                    f"[START] gpu={gpu} {job.label} pid={process.pid} "
                    f"batch={task_batch(job.cell.task, args)} "
                    f"pending={len(pending)}",
                    flush=True)
                emit("START", str(gpu), job.label)

            now = time.monotonic()
            if now - last_status >= args.status_interval:
                print(
                    f"[STATUS] active={len(active)} free={free} "
                    f"pending={len(pending)} completed_jobs={completed_jobs}",
                    flush=True)
                for gpu, current in sorted(active.items()):
                    progress = read_latest_progress(current.log_path)
                    progress_text = (
                        f"{progress[0]}%({progress[1]}/{progress[2]})"
                        if progress else "loading/no-tqdm-yet")
                    elapsed = (now - current.started_at) / 60
                    print(
                        f"[PROGRESS] gpu={gpu} {current.job.label} "
                        f"{progress_text} elapsed={elapsed:.1f}m "
                        f"log={current.log_path}", flush=True)
                    emit("PROGRESS", str(gpu), current.job.label, progress_text)
                last_status = now

            finished_gpu = next(
                (gpu for gpu, current in active.items()
                 if current.process.poll() is not None), None)
            if finished_gpu is None:
                time.sleep(2)
                continue

            current = active.pop(finished_gpu)
            rc = current.process.returncode
            current.log.write(
                f"[END] {time.strftime('%F %T')} rc={rc}\n".encode())
            current.log.close()
            total = expected_count(data_root, current.job.cell.task)
            valid = result_is_complete(
                result_path(run_dir, current.job), total,
                current.job.shard_id, current.job.num_shards)
            if rc != 0 or not valid:
                emit("FAIL", str(finished_gpu), current.job.label, f"rc={rc}")
                if not args.continue_on_cell_error:
                    stop_active()
                    raise RuntimeError(
                        f"{current.job.label} failed/incomplete rc={rc}; "
                        f"see {current.log_path}")
                failed_cells.add(current.job.cell)
                failed_jobs.append({
                    "cell": current.job.cell.label,
                    "job": current.job.label,
                    "returncode": rc,
                    "result_valid": valid,
                    "log": str(current.log_path),
                })
                pending = [
                    queued for queued in pending
                    if queued.cell != current.job.cell]
                free.append(finished_gpu)
                free.sort()
                print(
                    f"[SKIP CELL] {current.job.cell.label} after "
                    f"{current.job.label} failed rc={rc}; continuing with "
                    f"{len(pending)} jobs", flush=True)
                continue
            free.append(finished_gpu)
            free.sort()
            completed_jobs += 1
            print(
                f"[DONE] gpu={finished_gpu} {current.job.label} "
                f"free={free} pending={len(pending)}", flush=True)
            emit("DONE", str(finished_gpu), current.job.label, "100%")

            cell = current.job.cell
            if (current.job.is_shard and cell not in merged_cells
                    and cell not in failed_cells):
                if all_shards_complete(
                    cell, run_dir, data_root, current.job.num_shards):
                    command = merge_command(
                        cell, run_dir, current.job.num_shards, python)
                    print(f"[MERGE] {cell.label}", flush=True)
                    try:
                        subprocess.run(command, cwd=ROOT, check=True)
                    except subprocess.CalledProcessError as exc:
                        if not args.continue_on_cell_error:
                            raise
                        failed_cells.add(cell)
                        failed_jobs.append({
                            "cell": cell.label,
                            "job": f"{cell.label}.merge",
                            "returncode": exc.returncode,
                            "result_valid": False,
                            "log": None,
                        })
                        emit("FAIL", "-", cell.label, "merge")
                        print(
                            f"[SKIP CELL] {cell.label} shard merge failed; "
                            "continuing", flush=True)
                        continue
                    merged_cells.add(cell)
                    emit("MERGED", "-", cell.label, "100%")
            elif not current.job.is_shard:
                merged_cells.add(cell)
    except BaseException:
        stop_active()
        raise
    finally:
        status_handle.close()

    missing = [
        cell.label for cell in cells
        if not result_is_complete(
            run_dir / "evaluation" / f"order{cell.round_id}" / f"results-{cell.task}.json",
            expected_count(data_root, cell.task))
    ]
    failure_manifest = run_dir / "evaluation" / f"{args.matrix_mode}_failures.json"
    if failed_jobs or missing:
        failure_manifest.write_text(json.dumps({
            "schema_version": 1,
            "matrix_mode": args.matrix_mode,
            "continue_on_cell_error": args.continue_on_cell_error,
            "failed_jobs": failed_jobs,
            "failed_cells": sorted({entry["cell"] for entry in failed_jobs}),
            "missing_cells": missing,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif failure_manifest.exists():
        failure_manifest.unlink()
    if missing and not args.continue_on_cell_error:
        raise RuntimeError(
            f"{args.matrix_mode} cells still incomplete: {missing}")

    if not args.no_collect and args.matrix_mode in ("sparse15", "diagonal7"):
        # diagonal7 has no final row, so collect_results must accept a partial
        # matrix; it still fills diagonal_scores_rounds_1_to_7.
        force_partial = bool(missing) or args.matrix_mode == "diagonal7"
        summary_name = (
            "diagonal7_summary.json" if args.matrix_mode == "diagonal7"
            else "sparse15_partial_summary.json" if missing
            else "sparse15_summary.json")
        collect = [
            str(python), "-u", str(ROOT / "scripts/collect_results.py"),
            "--method", args.method, "--model", os.environ.get("SPARSE15_MODEL_LABEL", "llama31"), "--sparse-15",
            "--run-dir", str(run_dir), "--family", "paper_baseline",
            "--output", str(run_dir / summary_name),
        ]
        if force_partial:
            collect.append("--allow-partial")
        print(f"[COLLECT] {shlex.join(collect)}", flush=True)
        subprocess.run(collect, cwd=ROOT, check=True)
    elif not args.no_collect and missing:
        write_partial_lower_triangle_summary(
            run_dir, cells, args.method, failed_jobs)
    elif not args.no_collect:
        write_lower_triangle_summary(run_dir, cells, args.method)
    state = "partial" if missing else "complete"
    print(f"[OPT-QUEUE] {args.matrix_mode} {state}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
