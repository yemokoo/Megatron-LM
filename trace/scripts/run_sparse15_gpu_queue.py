#!/usr/bin/env python3
"""Run five sparse-15 evaluations with a method-major four-GPU job queue."""

from __future__ import annotations

import json
import os
import subprocess
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("TRACE_DATA_ROOT", ROOT / "data/trace"))
LOG_ROOT = Path(os.environ.get(
    "SPARSE15_LOG_ROOT",
    ROOT / "results/eval_chains/llama31_sparse15_five_20260728/gpu_queue",
))
TASKS = [
    "C-STANCE", "FOMC", "MeetingBank", "Py150",
    "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten",
]
DEFAULT_METHODS = [
    "slora_pre_released",
    "slora_post",
    "seq_lora",
    "ours_lora_moe_v1",
    "ours_lora_moe_v2",
]
METHODS = [
    method.strip()
    for method in os.environ.get(
        "SPARSE15_METHODS", ",".join(DEFAULT_METHODS)
    ).split(",")
    if method.strip()
]
GPUS = [
    int(gpu.strip())
    for gpu in os.environ.get("SPARSE15_GPUS", "0,1,2,3").split(",")
    if gpu.strip()
]
BATCH = int(os.environ.get("SPARSE15_EVAL_BATCH", "16"))
OURS_RUN_DIR = (
    Path(os.environ["OURS_LORAMOE_OUTPUT_ROOT"]).resolve()
    if os.environ.get("OURS_LORAMOE_OUTPUT_ROOT") else None
)


@dataclass(frozen=True)
class Cell:
    method: str
    cell_index: int
    round_id: int
    task: str

    @property
    def label(self) -> str:
        return f"{self.method}.order{self.round_id}.{self.task}"


def cells(method: str, final: bool) -> list[Cell]:
    if not final:
        return [
            Cell(method, i, i + 1, TASKS[i])
            for i in range(7)
        ]
    return [
        Cell(method, 7 + i, 8, task)
        for i, task in enumerate(TASKS)
    ]


def slora_layout(method: str) -> tuple[Path, str, Path]:
    if method == "slora_pre_released":
        return (
            ROOT / "implementations/SLoRA-upstream-port/scripts/repro/eval_trace.sh",
            "pre",
            ROOT / "results/full_runs_upstream_code/llama31/pre",
        )
    mode = "post" if method == "slora_post" else "seq"
    return (
        ROOT / "implementations/SLoRA-repro/scripts/repro/eval_trace.sh",
        mode,
        ROOT / f"results/full_runs/llama31/{mode}",
    )


def expected_count(task: str) -> int:
    with (DATA / task / "test.json").open(encoding="utf-8") as handle:
        return len(json.load(handle))


def is_complete(cell: Cell) -> bool:
    expected = expected_count(cell.task)
    if cell.method.startswith("ours_"):
        run_dir = (OURS_RUN_DIR or
                   ROOT / f"results/full_runs/llama31/{cell.method}")
        path = (run_dir / "evaluation" / f"order{cell.round_id}"
                / f"results-{cell.task}.json")
        if not path.is_file():
            return False
        try:
            with path.open(encoding="utf-8") as handle:
                result = json.load(handle)
            return len(result.get("results", [])) == expected
        except (OSError, ValueError, TypeError):
            return False
    _, _, run_dir = slora_layout(cell.method)
    path = run_dir / "evaluation" / f"order{cell.round_id}" / cell.task / "infer.jsonl"
    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if json.loads(line)) == expected
    except (OSError, ValueError):
        return False


def command(cell: Cell, gpu: int) -> tuple[list[str], dict[str, str]]:
    env = os.environ.copy()
    env["PATH"] = f"{ROOT}/.venv-runtime/bin:{env.get('PATH', '')}"
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "EVAL_SPARSE_15": "1",
        "EVAL_SHARD_COUNT": "15",
        "EVAL_SHARD_INDEX": str(cell.cell_index),
        "EVAL_SKIP_COLLECT": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    if cell.method.startswith("ours_"):
        version = cell.method.removeprefix("ours_lora_moe_")
        env.update({
            "OURS_LORAMOE_GPUS": str(gpu),
            "OURS_EVAL_SPARSE_15": "1",
            "OURS_LORAMOE_EVAL_BATCH": str(8 if cell.task == "MeetingBank" else BATCH),
        })
        if OURS_RUN_DIR:
            env["OURS_LORAMOE_OUTPUT_ROOT"] = str(OURS_RUN_DIR)
        cmd = [
            "bash", str(ROOT / "scripts/baselines/_run_ours_lora_moe.sh"),
            "eval", "llama31", version,
        ]
    else:
        script, mode, _ = slora_layout(cell.method)
        env["SLORA_EVAL_BATCH"] = str(8 if cell.task == "MeetingBank" else BATCH)
        if cell.method == "slora_pre_released":
            env["SLORA_OUTPUT_ROOT"] = str(
                ROOT / "results/full_runs_upstream_code")
        cmd = ["bash", str(script), mode, "llama31"]
    return cmd, env


def run_group(method: str) -> None:
    phase = "sparse15"
    method_cells = cells(method, final=False) + cells(method, final=True)
    method_cells = [
        cell for cell in method_cells
        if cell.task not in {"MeetingBank", "Py150"}
    ]
    pending = [cell for cell in method_cells if not is_complete(cell)]
    skipped = len(method_cells) - len(pending)
    print(
        f"[GROUP] method={method} phase={phase} "
        f"pending={len(pending)} complete={skipped}",
        flush=True,
    )
    active: dict[int, tuple[subprocess.Popen[bytes], Cell, object]] = {}
    free = list(GPUS)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    while pending or active:
        while pending and free:
            gpu = free.pop(0)
            cell = pending.pop(0)
            cmd, env = command(cell, gpu)
            log_path = LOG_ROOT / f"{phase}.{cell.label}.gpu{gpu}.log"
            log = log_path.open("ab", buffering=0)
            header = (
                f"\n[START] {time.strftime('%F %T')} gpu={gpu} "
                f"batch={BATCH} command={' '.join(cmd)}\n"
            )
            log.write(header.encode())
            proc = subprocess.Popen(
                cmd, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            active[gpu] = (proc, cell, log)
            print(f"[START] gpu={gpu} {cell.label} pid={proc.pid}", flush=True)

        finished_gpu = None
        while finished_gpu is None:
            for gpu, (proc, _, _) in active.items():
                if proc.poll() is not None:
                    finished_gpu = gpu
                    break
            if finished_gpu is None:
                time.sleep(2)

        proc, cell, log = active.pop(finished_gpu)
        rc = proc.returncode
        log.write(
            f"[END] {time.strftime('%F %T')} rc={rc}\n".encode())
        log.close()
        if rc != 0 or not is_complete(cell):
            for other, _, other_log in active.values():
                os.killpg(other.pid, signal.SIGTERM)
                other_log.close()
            raise RuntimeError(
                f"{cell.label} failed or incomplete (rc={rc}); "
                f"see {LOG_ROOT}")
        print(f"[DONE] gpu={finished_gpu} {cell.label}", flush=True)
        free.append(finished_gpu)
        free.sort()


def run_long_cells(method: str) -> None:
    long_cells = [
        Cell(method, 2, 3, "MeetingBank"),
        Cell(method, 9, 8, "MeetingBank"),
        Cell(method, 3, 4, "Py150"),
        Cell(method, 10, 8, "Py150"),
    ]
    pending = [cell for cell in long_cells if not is_complete(cell)]
    for cell in long_cells:
        if cell not in pending:
            print(f"[LONG SKIP] {cell.label}", flush=True)

    # Each long cell is split over four GPUs.  With eight GPUs, run two cells
    # concurrently in each wave; on a four-GPU host retain the serial behavior.
    group_count = max(1, len(GPUS) // 4)
    gpu_groups = [GPUS[i:i + 4] for i in range(0, group_count * 4, 4)]
    if any(len(group) != 4 for group in gpu_groups):
        raise RuntimeError("long-cell evaluation requires GPU groups of four")
    while pending:
        wave = pending[:len(gpu_groups)]
        pending = pending[len(gpu_groups):]
        active = []
        for cell, gpu_group in zip(wave, gpu_groups):
            task_batch = 4 if cell.task == "MeetingBank" else min(BATCH, 8)
            print(f"[LONG 4-WAY] {cell.label} batch={task_batch} "
                  f"gpus={gpu_group}", flush=True)
            if method.startswith("ours_"):
                long_command = [
                    sys.executable,
                    str(ROOT / "scripts/run_ours_py150_4way.py"),
                    "--round", str(cell.round_id), "--task", cell.task,
                    "--batch", str(task_batch), "--gpus",
                    ",".join(map(str, gpu_group)),
                ]
                if OURS_RUN_DIR:
                    long_command.extend(["--run-dir", str(OURS_RUN_DIR)])
                else:
                    long_command.extend(["--method", method])
            else:
                if gpu_group != [0, 1, 2, 3]:
                    raise RuntimeError(
                        "SLoRA long-cell helper currently requires GPUs 0,1,2,3")
                long_command = [
                    "bash", str(ROOT / "scripts/run_current_pre_py150_4way.sh"),
                    method, str(cell.round_id), cell.task, str(task_batch),
                ]
            active.append((
                cell,
                subprocess.Popen(long_command, cwd=ROOT, start_new_session=True),
            ))
        failures = [(cell, process.wait()) for cell, process in active]
        failures = [(cell, code) for cell, code in failures
                    if code != 0 or not is_complete(cell)]
        if failures:
            raise RuntimeError(f"long-cell failures: {failures}")


def prepare_slora_post() -> None:
    seq = ROOT / "results/full_runs/llama31/seq"
    jobs = [
        (round_id, seq / f"order{round_id}/adapter_model.safetensors",
         seq / f"order{round_id}/max.safetensors")
        for round_id in range(1, 9)
        if not (seq / f"order{round_id}/max.safetensors").is_file()
    ]
    if not jobs:
        return
    print(f"[POST-PREP] pending={len(jobs)}", flush=True)
    active: dict[int, tuple[subprocess.Popen[bytes], int, object]] = {}
    free = list(GPUS)
    impl = ROOT / "implementations/SLoRA-repro"
    while jobs or active:
        while jobs and free:
            gpu = free.pop(0)
            round_id, adapter, output = jobs.pop(0)
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "PYTHONPATH": str(impl),
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            })
            log = (LOG_ROOT / f"post_prep.order{round_id}.gpu{gpu}.log").open(
                "ab", buffering=0)
            cmd = [
                str(ROOT / ".venv-runtime/bin/python"),
                str(ROOT / "scripts/precompute_slora_post.py"),
                "--base-model", str(ROOT / "models/Llama-3.1-8B-Instruct"),
                "--adapter", str(adapter), "--output", str(output),
            ]
            proc = subprocess.Popen(
                cmd, cwd=impl, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            active[gpu] = (proc, round_id, log)
            print(f"[POST-PREP START] gpu={gpu} order={round_id}", flush=True)
        done = None
        while done is None:
            for gpu, (proc, _, _) in active.items():
                if proc.poll() is not None:
                    done = gpu
                    break
            if done is None:
                time.sleep(2)
        proc, round_id, log = active.pop(done)
        log.close()
        output = seq / f"order{round_id}/max.safetensors"
        if proc.returncode != 0 or not output.is_file():
            for other, _, other_log in active.values():
                os.killpg(other.pid, signal.SIGTERM)
                other_log.close()
            raise RuntimeError(f"post denoising failed for order{round_id}")
        print(f"[POST-PREP DONE] gpu={done} order={round_id}", flush=True)
        free.append(done)
        free.sort()


def collect() -> None:
    for method in METHODS:
        collect_command = [
                sys.executable, str(ROOT / "scripts/collect_results.py"),
                "--method", method, "--model", "llama31", "--sparse-15",
            ]
        if method.startswith("ours_") and OURS_RUN_DIR:
            collect_command.extend([
                "--run-dir", str(OURS_RUN_DIR),
                "--family", "paper_baseline",
                "--output", str(OURS_RUN_DIR / "sparse15_summary.json"),
            ])
        subprocess.run(collect_command, cwd=ROOT, check=True)


def main() -> None:
    print(f"[QUEUE] GPUs={GPUS} batch={BATCH}", flush=True)
    for method in METHODS:
        if method == "slora_post":
            prepare_slora_post()
        run_group(method)
        run_long_cells(method)
    collect()
    print("[QUEUE] all five sparse-15 evaluations complete", flush=True)


if __name__ == "__main__":
    main()
