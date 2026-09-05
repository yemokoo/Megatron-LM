#!/usr/bin/env python3
"""Resume Ours sparse-15 evaluation with mixed 1-GPU and 4-GPU jobs."""

from __future__ import annotations

import os
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import run_sparse15_gpu_queue as queue


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = Path(os.environ["OURS_LORAMOE_OUTPUT_ROOT"]).resolve()
METHOD = os.environ.get("SPARSE15_METHODS", "ours_lora_moe_v3")
GPUS = queue.GPUS
THREAD_ENV = {
    "OMP_NUM_THREADS": os.environ.get("SPARSE15_CPU_THREADS", "4"),
    "MKL_NUM_THREADS": os.environ.get("SPARSE15_CPU_THREADS", "4"),
    "OPENBLAS_NUM_THREADS": os.environ.get("SPARSE15_CPU_THREADS", "4"),
    "NUMEXPR_NUM_THREADS": os.environ.get("SPARSE15_CPU_THREADS", "4"),
    "TOKENIZERS_PARALLELISM": "false",
}


def normal_batch(task: str) -> int:
    return {
        "ScienceQA": int(os.environ.get("SPARSE15_SCIENCEQA_BATCH", "128")),
        "20Minuten": int(os.environ.get("SPARSE15_20MINUTEN_BATCH", "32")),
    }.get(task, int(os.environ.get("SPARSE15_EVAL_BATCH", "32")))


def normal_jobs() -> list[queue.Cell]:
    candidates = queue.cells(METHOD, final=False) + queue.cells(METHOD, final=True)
    return [
        cell for cell in candidates
        if cell.task not in {"MeetingBank", "Py150"}
        and not queue.is_complete(cell)
    ]


def long_jobs() -> list[queue.Cell]:
    candidates = [
        queue.Cell(METHOD, 2, 3, "MeetingBank"),
        queue.Cell(METHOD, 9, 8, "MeetingBank"),
        queue.Cell(METHOD, 3, 4, "Py150"),
        queue.Cell(METHOD, 10, 8, "Py150"),
    ]
    return [cell for cell in candidates if not queue.is_complete(cell)]


def start_normal(cell: queue.Cell, gpu: int):
    command, env = queue.command(cell, gpu)
    env.update(THREAD_ENV)
    env["OURS_LORAMOE_EVAL_BATCH"] = str(normal_batch(cell.task))
    log_path = queue.LOG_ROOT / f"efficient.{cell.label}.gpu{gpu}.log"
    log = log_path.open("ab", buffering=0)
    log.write(
        f"\n[START] {time.strftime('%F %T')} gpu={gpu} "
        f"batch={normal_batch(cell.task)}\n".encode())
    process = subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True)
    return process, log


def start_long(cell: queue.Cell, gpus: list[int]):
    batch = (
        int(os.environ.get("SPARSE15_MEETINGBANK_BATCH", "1"))
        if cell.task == "MeetingBank"
        else int(os.environ.get("SPARSE15_PY150_BATCH", "8")))
    command = [
        sys.executable, str(ROOT / "scripts/run_ours_py150_4way.py"),
        "--run-dir", str(RUN_DIR), "--round", str(cell.round_id),
        "--task", cell.task, "--batch", str(batch),
        "--gpus", ",".join(map(str, gpus)),
    ]
    env = os.environ.copy()
    env.update(THREAD_ENV)
    log_path = queue.LOG_ROOT / f"efficient.{cell.label}.4way.log"
    log = log_path.open("ab", buffering=0)
    log.write(
        f"\n[START] {time.strftime('%F %T')} gpus={gpus} batch={batch}\n".encode())
    process = subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True)
    data_path = queue.DATA / cell.task / "test.json"
    expected_count = len(json.loads(data_path.read_text()))
    used_gpus = []
    for shard, gpu in enumerate(gpus):
        result_path = (
            RUN_DIR / "evaluation" / f"order{cell.round_id}"
            / f"results-{cell.task}.shard{shard}.json")
        complete = False
        if result_path.is_file():
            try:
                payload = json.loads(result_path.read_text())
                complete = payload.get("sample_indices") == list(
                    range(shard, expected_count, 4))
            except (OSError, ValueError):
                pass
        if not complete:
            used_gpus.append(gpu)
    return process, log, used_gpus


def stop_all(active) -> None:
    for process, _, _, log in active:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        log.close()


def main() -> None:
    if len(GPUS) not in {4, 8}:
        raise SystemExit(
            "efficient sparse-15 resume requires exactly four or eight GPUs")
    queue.LOG_ROOT.mkdir(parents=True, exist_ok=True)
    normals = normal_jobs()
    longs = long_jobs()
    free = list(GPUS)
    active = []
    print(
        f"[RESUME] completed={15 - len(normals) - len(longs)} "
        f"normal_pending={len(normals)} long_pending={len(longs)}",
        flush=True)

    while normals or longs or active:
        # Single-GPU cells go first; a four-GPU shard job fills the largest
        # remaining block. As short jobs finish, a second shard job can overlap.
        while normals and free:
            cell = normals.pop(0)
            gpu = free.pop(0)
            process, log = start_normal(cell, gpu)
            active.append((process, cell, [gpu], log))
            print(f"[START 1GPU] {cell.label} gpu={gpu}", flush=True)
        while longs and len(free) >= 4:
            cell = longs.pop(0)
            gpu_group, free = free[:4], free[4:]
            process, log, used_gpus = start_long(cell, gpu_group)
            free.extend(gpu for gpu in gpu_group if gpu not in used_gpus)
            free.sort()
            active.append((process, cell, used_gpus, log))
            print(f"[START 4GPU] {cell.label} gpus={used_gpus} "
                  f"reserved_map={gpu_group}", flush=True)

        if not active:
            raise RuntimeError("pending jobs cannot fit available GPUs")
        finished = None
        while finished is None:
            for index, (process, _, _, _) in enumerate(active):
                if process.poll() is not None:
                    finished = index
                    break
            if finished is None:
                time.sleep(2)
        process, cell, used_gpus, log = active.pop(finished)
        code = process.returncode
        log.write(f"[END] {time.strftime('%F %T')} rc={code}\n".encode())
        log.close()
        if code != 0 or not queue.is_complete(cell):
            stop_all(active)
            raise RuntimeError(f"{cell.label} failed or incomplete (rc={code})")
        free.extend(used_gpus)
        free.sort()
        print(f"[DONE] {cell.label} free={free}", flush=True)

    queue.collect()
    print("[RESUME] sparse-15 complete", flush=True)


if __name__ == "__main__":
    main()
