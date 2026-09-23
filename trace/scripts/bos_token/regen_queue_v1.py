#!/usr/bin/env python3
"""Schedule the original v1 gen_doc + answer_pass shard jobs across eight GPUs.

Each shard keeps v1's prompt, answer, guard, bias and seed arguments.  A GPU
claims another shard as soon as it finishes, so long-output tails do not leave
other cards idle.  Completed records.part*.jsonl files are resumable.
"""
import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]
CAP = {"C-STANCE": 256, "FOMC": 256, "MeetingBank": 1024, "Py150": 1024, "ScienceQA": 512,
       "NumGLUE-cm": 256, "NumGLUE-ds": 256}
CUE = {"C-STANCE": "\n态度：", "FOMC": "\nStance:", "MeetingBank": "\nSummary:", "Py150": "",
       "ScienceQA": "\nAnswer:", "NumGLUE-cm": "\nAnswer:", "NumGLUE-ds": "\nAnswer:"}
COST = {"MeetingBank": 4, "Py150": 3, "ScienceQA": 2, "C-STANCE": 1, "FOMC": 1,
        "NumGLUE-cm": 1, "NumGLUE-ds": 1}
SHARDS = {"MeetingBank": 10, "Py150": 10, "ScienceQA": 10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--cond-dir", required=True)
    ap.add_argument("--num-experts", type=int, required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--num-tasks", type=int, required=True)
    ap.add_argument("--default-per-task", type=int, required=True)
    ap.add_argument("--meetingbank-per-task", type=int, default=0)
    ap.add_argument("--scienceqa-per-task", type=int, default=0)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--guard-decision", default="none")
    a = ap.parse_args()
    dest = Path(a.dest)
    jobs = []
    for j in range(a.num_tasks):
        task = TASKS[j]
        total = (a.meetingbank_per_task if task == "MeetingBank" and a.meetingbank_per_task else
                 a.scienceqa_per_task if task == "ScienceQA" and a.scienceqa_per_task else
                 a.default_per_task)
        bias = Path(a.cond_dir) / f"force_E{j}_of{a.num_experts}.pt"
        if not bias.is_file():
            raise FileNotFoundError(bias)
        nshards = SHARDS.get(task, 8)
        for k in range(nshards):
            n = total // nshards + (k < total % nshards)
            if n == 0:
                continue
            out = dest / task / f"records.part{k}.jsonl"
            if out.is_file() and out.stat().st_size > 0:
                continue
            jobs.append((task, j, k, n, bias))
    jobs.sort(key=lambda job: (-COST.get(job[0], 1), -CAP[job[0]], job[1], job[2]))
    work = queue.Queue()
    for job in jobs:
        work.put(job)
    errors = []
    lock = threading.Lock()
    stop = threading.Event()
    start = time.time()
    print(f"[queue] round {a.round}: {len(jobs)} pending v1 shards across {a.gpus}", flush=True)

    def worker(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        while not stop.is_set():
            try:
                task, j, k, n, bias = work.get_nowait()
            except queue.Empty:
                return
            task_dir = dest / task
            task_dir.mkdir(parents=True, exist_ok=True)
            stage_a = task_dir / f"stageA.s{k}"
            out = task_dir / f"records.part{k}.jsonl"
            seed = 1000 + a.round * 100 + j * 10 + k
            stage_a_cmd = [sys.executable, str(REPO / "scripts/bos_token/gen_doc.py"),
                           "--checkpoint", a.checkpoint, "--all-layer-bias-file", str(bias),
                           "--bias-positions", "decision", "--prompt-mode", "chat_header",
                           "--bos-guard", "--guard-header", "--guard-decision", a.guard_decision,
                           "--task-index", str(j), "--num-seqs", str(n), "--batch", str(n),
                           "--max-new-tokens", str(CAP[task]), "--seed", str(seed),
                           "--out-dir", str(stage_a)]
            stage_b_cmd = [sys.executable, str(REPO / "scripts/analysis/answer_pass_v3_fix.py"),
                           "--checkpoint", a.checkpoint, "--stage-a", str(stage_a),
                           "--out", str(out)]
            if CUE[task]:
                stage_b_cmd += ["--prompt-cue", CUE[task]]
            stage_b_cmd += ["--bos-guard", "--guard-header", "--guard-decision", a.guard_decision,
                            "--max-answer-tokens", str(CAP[task]), "--batch", "32"]
            try:
                with (task_dir / f"genA_s{k}.log").open("w") as log:
                    subprocess.run(stage_a_cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                with (task_dir / f"genB_s{k}.log").open("w") as log:
                    subprocess.run(stage_b_cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                if not out.is_file():
                    raise RuntimeError(f"missing {out}")
                with lock:
                    print(f"[queue] gpu{gpu} {task} shard{k}: {n} requested, {sum(1 for _ in out.open())} records", flush=True)
            except Exception as exc:
                with lock:
                    errors.append(f"gpu{gpu} {task} shard{k}: {exc}")
                    print(f"[queue] ERROR {errors[-1]}", flush=True)
                stop.set()
            finally:
                work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False)
               for gpu in a.gpus.split(",")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(f"[queue] done in {time.time()-start:.0f}s, errors={len(errors)}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
