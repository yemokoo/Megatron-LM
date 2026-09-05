#!/usr/bin/env python3
"""Evaluate baseline/treatment round-6 checkpoints and write one comparison."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-runtime/bin/python"
EVALUATOR = ROOT / "implementations/llmcl_benchmark/evaluate_Ours_LoRA_MoE.py"
BASE_MODEL = ROOT / "models/Llama-3.1-8B-Instruct"
DATA = ROOT / "data/trace"
BASELINE = Path(os.environ.get(
    "CM_QUOTA_BASELINE_RUN",
    "/data2/seonghyeonnoh/LLM-continual-learning-runs/"
    "v2_new_retrain_20260807"))
TREATMENT = Path(os.environ.get(
    "CM_QUOTA_TREATMENT_RUN",
    "/data2/seonghyeonnoh/LLM-continual-learning-runs/"
    "v2_new_cm_expert_quota_50_30_20260810"))
TASKS = [
    "C-STANCE", "FOMC", "MeetingBank", "Py150",
    "ScienceQA", "NumGLUE-cm",
]
BATCH = {
    "C-STANCE": 32, "FOMC": 32, "ScienceQA": 128,
    "NumGLUE-cm": 64,
}
GPUS = [
    int(value) for value in os.environ.get(
        "CM_QUOTA_GPUS", "4,5,6,7").split(",") if value.strip()
]


def expected_count(task: str) -> int:
    return len(json.loads((DATA / task / "test.json").read_text()))


def complete(path: Path, task: str, forced: int | None = None) -> bool:
    try:
        payload = json.loads(path.read_text())
        if len(payload.get("results", [])) != expected_count(task):
            return False
        return forced is None or payload.get("forced_expert_index") == forced
    except (OSError, ValueError, TypeError):
        return False


def direct_job(run: Path, task: str, gpu: int,
               forced: int | None = None) -> tuple[subprocess.Popen, object]:
    output_subdir = (
        Path("evaluation_task_expert/order6")
        if forced is not None else Path("evaluation/order6"))
    output = run / output_subdir
    output.mkdir(parents=True, exist_ok=True)
    result = output / f"results-{task}.json"
    if complete(result, task, forced):
        print(f"[SKIP] {run.name} {task} forced={forced}", flush=True)
        return None, None
    log_dir = run / "eval_round6_comparison_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"forced{forced}" if forced is not None else "natural"
    log = (log_dir / f"{task}.{suffix}.log").open("ab", buffering=0)
    command = [
        str(PYTHON), str(EVALUATOR),
        "--checkpoint_dir", str(run / "5"),
        "--base_model_name_or_path", str(BASE_MODEL),
        "--data_path", str(DATA), "--inference_tasks", task,
        "--inference_output_path", str(output),
        "--summary_filename", f"{task}.{suffix}.summary.json",
        "--max_prompt_len", "0", "--max_ans_len", "1024",
        "--no-task_generation_limits", "--slora_conv_mode", "llama3",
        "--per_device_eval_batch_size", str(BATCH[task]),
        "--temperature", "0",
    ]
    if forced is not None:
        command.extend(["--force_expert_index", str(forced)])
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONNOUSERSITE": "1", "WANDB_MODE": "offline",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
    })
    log.write(
        f"\n[START] {time.strftime('%F %T')} gpu={gpu} forced={forced}\n"
        .encode())
    process = subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True)
    print(
        f"[START] gpu={gpu} {run.name} {task} forced={forced} "
        f"pid={process.pid}", flush=True)
    return process, log


def wait_jobs(jobs) -> None:
    failed = []
    for process, log, label in jobs:
        if process is None:
            continue
        code = process.wait()
        log.write(f"[END] {time.strftime('%F %T')} rc={code}\n".encode())
        log.close()
        if code:
            failed.append((label, code))
        else:
            print(f"[DONE] {label}", flush=True)
    if failed:
        raise RuntimeError(f"round-6 direct evaluations failed: {failed}")


def run_direct_wave(specs, gpus=None) -> None:
    gpus = GPUS if gpus is None else list(gpus)
    if len(specs) > len(gpus):
        raise ValueError(
            f"direct wave has {len(specs)} jobs but only {len(gpus)} GPUs")
    jobs = []
    for gpu, (run, task, forced) in zip(gpus, specs):
        process, log = direct_job(run, task, gpu, forced)
        jobs.append((process, log, f"{run.name}:{task}:forced={forced}"))
    wait_jobs(jobs)


def run_long_wave(task: str) -> None:
    processes = []
    for run, gpus in ((BASELINE, "0,1,2,3"), (TREATMENT, "4,5,6,7")):
        output = run / "evaluation/order6" / f"results-{task}.json"
        if complete(output, task):
            print(f"[LONG SKIP] {run.name} {task}", flush=True)
            continue
        command = [
            str(PYTHON), str(ROOT / "scripts/run_ours_py150_4way.py"),
            "--run-dir", str(run), "--round", "6", "--task", task,
            "--batch", "1" if task == "MeetingBank" else "8",
            "--gpus", gpus,
        ]
        env = os.environ.copy()
        env.update({
            "TRACE_PYTHON": str(PYTHON), "TRACE_DATA_ROOT": str(DATA),
            "PYTHONNOUSERSITE": "1", "WANDB_MODE": "offline",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        })
        process = subprocess.Popen(command, cwd=ROOT, env=env)
        processes.append((process, f"{run.name}:{task}"))
    failed = []
    for process, label in processes:
        code = process.wait()
        if code:
            failed.append((label, code))
        else:
            print(f"[LONG DONE] {label}", flush=True)
    if failed:
        raise RuntimeError(f"round-6 long evaluations failed: {failed}")


def score(payload: dict, task: str) -> float:
    metrics = payload["eval"]
    if task in {"C-STANCE", "FOMC", "ScienceQA", "NumGLUE-cm"}:
        return 100.0 * float(metrics["accuracy"])
    if task == "MeetingBank":
        return 100.0 * float(metrics["rouge-L"])
    if task == "Py150":
        return float(metrics["similarity"])
    raise KeyError(task)


def collect() -> None:
    rows = {}
    for task in TASKS:
        values = {}
        for label, run in (("baseline", BASELINE), ("treatment", TREATMENT)):
            path = run / "evaluation/order6" / f"results-{task}.json"
            if not complete(path, task):
                raise RuntimeError(f"missing complete evaluation: {path}")
            values[label] = score(json.loads(path.read_text()), task)
        values["delta"] = values["treatment"] - values["baseline"]
        rows[task] = values

    forced = {}
    for label, run in (("baseline", BASELINE), ("treatment", TREATMENT)):
        path = (run / "evaluation_task_expert/order6"
                / "results-NumGLUE-cm.json")
        if not complete(path, "NumGLUE-cm", forced=5):
            raise RuntimeError(f"missing complete forced evaluation: {path}")
        forced[label] = score(json.loads(path.read_text()), "NumGLUE-cm")
    forced["delta"] = forced["treatment"] - forced["baseline"]

    old_tasks = TASKS[:-1]
    old_deltas = [rows[task]["delta"] for task in old_tasks]
    summary = {
        "schema_version": 1,
        "checkpoint_round": 6,
        "baseline_run": str(BASELINE),
        "treatment_run": str(TREATMENT),
        "natural_scores": rows,
        "forced_own_expert_numglue_cm": forced,
        "old_task_baseline_average": sum(
            rows[task]["baseline"] for task in old_tasks) / len(old_tasks),
        "old_task_treatment_average": sum(
            rows[task]["treatment"] for task in old_tasks) / len(old_tasks),
        "old_task_average_delta": sum(old_deltas) / len(old_deltas),
        "old_task_worst_delta": min(old_deltas),
    }
    path = TREATMENT / "round6_baseline_comparison.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    if not (TREATMENT / "5" / "pytorch_model.bin").is_file():
        raise SystemExit(f"treatment checkpoint is not ready: {TREATMENT / '5'}")
    short_specs = [
        (TREATMENT, "C-STANCE", None),
        (TREATMENT, "FOMC", None),
        (TREATMENT, "ScienceQA", None),
        (TREATMENT, "NumGLUE-cm", None),
        (BASELINE, "C-STANCE", None),
        (BASELINE, "FOMC", None),
        (BASELINE, "ScienceQA", None),
    ]
    run_direct_wave(short_specs)
    run_direct_wave([
        (BASELINE, "NumGLUE-cm", 5),
        (TREATMENT, "NumGLUE-cm", 5),
    ])
    run_long_wave("MeetingBank")
    run_long_wave("Py150")
    collect()


if __name__ == "__main__":
    main()
