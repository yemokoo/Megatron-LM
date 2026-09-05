#!/usr/bin/env python3
"""Evaluate final V2-new checkpoint with task-index expert forced at weight 1."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ["OURS_LORAMOE_OUTPUT_ROOT"]).resolve()
BASE = Path(os.environ.get(
    "OURS_BASE_MODEL",
    "/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct"))
DATA = Path(os.environ.get("TRACE_DATA_ROOT", ROOT / "data/trace"))
OUT = RUN / "evaluation_task_expert" / "order8"
LOGS = RUN / "eval_task_expert_logs"
TASKS = [
    "C-STANCE", "FOMC", "MeetingBank", "Py150",
    "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten",
]
BATCH = {
    "C-STANCE": 32, "FOMC": 32, "MeetingBank": 1, "Py150": 8,
    "ScienceQA": 128, "NumGLUE-cm": 32, "NumGLUE-ds": 32,
    "20Minuten": 32,
}


def expected_count(task: str) -> int:
    return len(json.loads((DATA / task / "test.json").read_text()))


def complete(task: str) -> bool:
    path = OUT / f"results-{task}.json"
    try:
        payload = json.loads(path.read_text())
        return (
            len(payload.get("results", [])) == expected_count(task)
            and payload.get("forced_expert_index") == TASKS.index(task)
        )
    except (OSError, ValueError, TypeError):
        return False


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    active = []
    for expert_index, task in enumerate(TASKS):
        if complete(task):
            print(f"[TASK-EXPERT SKIP] {task} expert={expert_index}", flush=True)
            continue
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": str(expert_index),
            "PYTHONNOUSERSITE": "1",
            "WANDB_MODE": "offline",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "TOKENIZERS_PARALLELISM": "false",
        })
        cmd = [
            sys.executable, str(ROOT / "implementations/llmcl_benchmark/evaluate_Ours_LoRA_MoE.py"),
            "--checkpoint_dir", str(RUN / "7"),
            "--base_model_name_or_path", str(BASE),
            "--data_path", str(DATA),
            "--inference_tasks", task,
            "--inference_output_path", str(OUT),
            "--summary_filename", f"{task}.summary.json",
            "--max_prompt_len", "0", "--max_ans_len", "1024",
            "--no-task_generation_limits", "--slora_conv_mode", "llama3",
            "--per_device_eval_batch_size", str(BATCH[task]),
            "--temperature", "0", "--force_expert_index", str(expert_index),
        ]
        log = (LOGS / f"{expert_index}_{task}.log").open("ab", buffering=0)
        log.write(f"\n[START] {time.strftime('%F %T')} gpu={expert_index} expert={expert_index}\n".encode())
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        active.append((proc, task, expert_index, log))
        print(f"[TASK-EXPERT START] {task} gpu={expert_index} expert={expert_index}", flush=True)

    failed = []
    for proc, task, expert_index, log in active:
        rc = proc.wait()
        log.write(f"[END] {time.strftime('%F %T')} rc={rc}\n".encode())
        log.close()
        path = OUT / f"results-{task}.json"
        if rc == 0 and path.is_file():
            payload = json.loads(path.read_text())
            payload["forced_expert_index"] = expert_index
            path.write_text(json.dumps(payload, ensure_ascii=False))
        if rc != 0 or not complete(task):
            failed.append((task, rc))
        else:
            print(f"[TASK-EXPERT DONE] {task} expert={expert_index}", flush=True)
    if failed:
        raise RuntimeError(f"task-expert evaluations failed: {failed}")

    summary = {}
    for expert_index, task in enumerate(TASKS):
        payload = json.loads((OUT / f"results-{task}.json").read_text())
        summary[task] = {
            "forced_expert_index": expert_index,
            "eval": payload["eval"],
            "samples": len(payload["results"]),
        }
    (RUN / "task_expert_final_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print("[TASK-EXPERT COMPLETE] all 8 final-checkpoint evaluations", flush=True)


if __name__ == "__main__":
    main()
