#!/usr/bin/env python3
"""Work-queue regeneration: (task, shard) jobs are handed to whichever GPU is free, longest task first.

One worker process per GPU loads the checkpoint ONCE and then pulls jobs from a shared queue, so a GPU
that finishes a short task immediately picks up the next piece of work instead of idling until the GPU
with the long task catches up.  Each job is one (task, shard) pair and writes exactly what
regen_multitask.py would: <dest>/<task>/stageA.s<k>/{docs,text,stats}.json[l] and records.part<k>.jsonl.

  python regen_pool.py --checkpoint CKPT --dest DIR --round T --num-tasks T --gpus 0,1,..,7 \
      [--per-task 1000] [--shards 8] [--overrides gen/regen_overrides.json]
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]
CAP = {"C-STANCE": 256, "FOMC": 256, "MeetingBank": 1024, "Py150": 1024, "ScienceQA": 512,
       "NumGLUE-cm": 256, "NumGLUE-ds": 256}
CUE = {"C-STANCE": "\n态度：", "FOMC": "\nStance:", "MeetingBank": "\nSummary:", "Py150": "",
       "ScienceQA": "\nAnswer:", "NumGLUE-cm": "\nAnswer:", "NumGLUE-ds": "\nAnswer:"}
# rough cost per (task, shard): long-output tasks first so they start while short ones fill the gaps
COST = {"MeetingBank": 4.0, "Py150": 3.0, "ScienceQA": 1.5, "C-STANCE": 1.0, "FOMC": 1.0,
        "NumGLUE-cm": 0.8, "NumGLUE-ds": 0.8}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--num-tasks", type=int, required=True, help="regenerate tasks 0..num_tasks-1")
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--per-task", type=int, default=1000)
    p.add_argument("--shards", type=int, default=8, help="shards for short tasks")
    p.add_argument("--long-shards", type=int, default=32, help="finer shards for long-output tasks (MeetingBank/Py150)")
    p.add_argument("--procs-per-gpu", type=int, default=2)
    p.add_argument("--overrides", default="")
    p.add_argument("--guard-decision", default="none")
    return p.parse_args()


def main():
    a = parse_args()
    dest = Path(a.dest); dest.mkdir(parents=True, exist_ok=True)
    ov = json.load(open(a.overrides)) if a.overrides and Path(a.overrides).exists() else {}
    jobs = []
    for j in range(a.num_tasks):
        task = TASKS[j]; long = CAP[task] >= 1024
        # long-output tasks are split much finer so no single piece becomes the tail of the round
        shards = a.long_shards if long else a.shards
        per_shard = a.per_task // shards
        for k in range(shards):
            out = dest / task / f"records.part{k}.jsonl"
            if out.exists() and out.stat().st_size > 0:
                continue
            n = per_shard + (a.per_task - per_shard * shards if k == shards - 1 else 0)
            batch_a = int(ov.get("long_batch_a", 96)) if long else int(ov.get("short_batch_a", per_shard))
            batch_b = int(ov.get("long_batch_b", 48)) if long else int(ov.get("short_batch_b", per_shard))
            jobs.append({"cost": COST[task], "task_index": j, "task": task, "shard": k,
                         "num_seqs": n, "max_new_tokens": CAP[task], "cue": CUE[task],
                         "batch_a": min(batch_a, n), "batch_b": min(batch_b, n),
                         "seed": 1000 + a.round * 100 + j * 10 + k})
    jobs.sort(key=lambda x: -x["cost"])
    print(f"[pool] round {a.round}: {len(jobs)} jobs over {a.gpus} x{a.procs_per_gpu}", flush=True)
    if not jobs:
        return 0
    queue_dir = dest / "_pool"; queue_dir.mkdir(exist_ok=True)
    # one job file per worker slot, refilled as workers finish: simplest portable queue = a lock dir
    (queue_dir / "jobs.json").write_text(json.dumps(jobs, ensure_ascii=False))
    workers, t0 = [], time.time()
    slots = [(g, p) for g in a.gpus.split(",") for p in range(a.procs_per_gpu)]
    for g, p in slots:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        cmd = [sys.executable, str(REPO / "scripts/bos_token/regen_worker.py"),
               "--checkpoint", a.checkpoint, "--dest", str(dest), "--queue", str(queue_dir),
               "--worker", f"{g}.{p}", "--guard-decision", a.guard_decision]
        log = open(dest / f"regen.pool.g{g}.p{p}.log", "w")
        workers.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))
        time.sleep(1)
    rc = 0
    for w in workers:
        rc |= w.wait()
    print(f"[pool] done rc={rc} in {time.time()-t0:.0f}s", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
