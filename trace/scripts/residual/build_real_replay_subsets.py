#!/usr/bin/env python3
"""Pre-build nested real-data replay subsets (1/5/10/15/20% per task).

The 20% pool is drawn first, per task, from a seed-2025 permutation of that
task's train.json; the smaller ratios are prefixes of that pool, so
1% subset 5% subset 10% subset 15% subset 20%.  Each ratio is written as a
``fixed_replay_memory`` directory in the exact schema the trainer validates
(``Ours_LoRA_MoE._validate_v2_new_saved_memory``): task/task_index/
source_samples/unique_samples/selection_mode/indices/indices_sha256/
resolved_seed, with the digest over the indices in list order.

  python build_real_replay_subsets.py --out-root <dir> [--seed 2025]

A run then points at one ratio, e.g.
  cp -a <out-root>/pct10/fixed_replay_memory <RUN_DIR>/model/
  ... --v2_new_persistent_samples_per_task 500
(the per-task count must match the ratio, see manifest.json).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
         "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]
RATIOS = [20, 15, 10, 5, 1]          # 20 first: the others are prefixes of it


def digest(indices):
    return hashlib.sha256(",".join(map(str, indices)).encode("utf-8")).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace")
    p.add_argument("--out-root", required=True)
    p.add_argument("--seed", type=int, default=2025)
    a = p.parse_args()
    out_root = Path(a.out_root)
    manifest = {"seed": a.seed, "data_root": a.data_root, "ratios_pct": RATIOS,
                "nesting": "smaller ratios are prefixes of the 20% pool", "tasks": {}}

    for task_index, task in enumerate(TASKS):
        n_source = len(json.loads((Path(a.data_root) / task / "train.json").read_text()))
        rng = np.random.default_rng([a.seed, task_index])
        pool = rng.permutation(n_source)                       # per-task stream
        counts = {pct: int(round(n_source * pct / 100)) for pct in RATIOS}
        entry = {"source_samples": n_source, "counts": counts, "sha256": {}}
        for pct in RATIOS:
            n = counts[pct]
            indices = [int(i) for i in pool[:n]]
            d = digest(indices)
            entry["sha256"][pct] = d
            record = {
                "schema_version": 3,
                "task_index": task_index,
                "task": task,
                "source_samples": n_source,
                "persistent_samples_per_task": n,
                "unique_samples": n,
                "resolved_seed": a.seed,
                "seed": a.seed,
                "selection_mode": "random",
                "manifest_path": None,
                "indices_sha256": d,
                "indices": indices,
                "available_from_next_task_for_v2_past_replay": True,
            }
            dest = out_root / f"pct{pct}" / "fixed_replay_memory"
            dest.mkdir(parents=True, exist_ok=True)
            safe = task.replace("/", "_")
            (dest / f"task_{task_index}_{safe}.json").write_text(json.dumps(record, indent=2) + "\n")
        manifest["tasks"][task] = entry

        # nesting + validity assertions
        sets = {pct: set(int(i) for i in pool[:counts[pct]]) for pct in RATIOS}
        for small, large in ((1, 5), (5, 10), (10, 15), (15, 20)):
            assert sets[small] < sets[large], f"{task}: {small}% not nested in {large}%"
        for pct in RATIOS:
            assert len(sets[pct]) == counts[pct]
            assert all(0 <= i < n_source for i in sets[pct])

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"seed={a.seed}  tasks={len(TASKS)}  ratios={RATIOS}")
    for task, e in manifest["tasks"].items():
        print(f"  {task:12s} source={e['source_samples']} " +
              " ".join(f"{pct}%={e['counts'][pct]}" for pct in RATIOS))
    print(f"written under {out_root}/pct{{{','.join(map(str, RATIOS))}}}/fixed_replay_memory")


if __name__ == "__main__":
    main()
