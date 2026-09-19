#!/usr/bin/env python3
"""Prove that an HP-sensitivity cell trained with the knobs it claims.

Reads ``<run>/cell.json`` (written by run_hp_cell.sh) and checks every round's
``model/<r>/lora_moe_meta.json`` against it:

* per-task expert granularity: ``experts_per_task``, ``r``, ``attention_rank``,
  ``alpha``, ``top_k``, and the cumulative ``num_experts == (r + 1) * E`` --
  this is what makes the "2 x rank32 / 4 x rank16 / 8 x rank8 at fixed rank
  sum" axis verifiable rather than assumed;
* training order: ``dataset_order`` must equal the cell's order and round r's
  ``stop_after_task`` must be that order's r-th task -- the only record of a
  reversed run, since checkpoints are named by round index alone;
* the switches that must NOT move across the study: phase mode, KD-init,
  replay source.

usage: verify_hp_run.py <run-dir> [--expect-rounds 8]
exit 0 = every check passed.
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expect-rounds", type=int, default=8)
    parser.add_argument(
        "--single-process", action="store_true",
        help="all rounds ran in one process (smoke), so every round's "
             "stop_after_task is the last expected round's task")
    args = parser.parse_args()

    run = args.run_dir
    cell_path = run / "cell.json"
    if not cell_path.is_file():
        print(f"FAIL: no cell.json at {cell_path}")
        return 1
    cell = json.loads(cell_path.read_text())
    order = [t for t in cell["task_order"].split(",") if t]
    E = int(cell["experts_per_task"])
    rank = int(cell["rank"])
    failures = []
    checked = 0

    print(f"cell {cell['name']}: order={cell['order_mode']} "
          f"E={E} rank={rank} top_k={cell['top_k']} alpha={cell['alpha']} "
          f"rank_sum={cell['rank_sum_per_task']} phase={cell['phase']} "
          f"kd={cell['kd_init']} replay={cell['replay_source']}")

    if len(order) != 8 or len(set(order)) != 8:
        failures.append(f"task_order is not 8 distinct tasks: {order}")

    for r in range(args.expect_rounds):
        meta_path = run / "model" / str(r) / "lora_moe_meta.json"
        if not meta_path.is_file():
            failures.append(f"round {r}: missing {meta_path}")
            continue
        meta = json.loads(meta_path.read_text())
        want = {
            "experts_per_task": E,
            "r": rank,
            "attention_rank": rank,
            "alpha": int(cell["alpha"]),
            "top_k": int(cell["top_k"]),
            "num_experts": (r + 1) * E,
            "expert_dispatch": cell.get("expert_dispatch", "loop"),
        }
        for key, expected in want.items():
            actual = meta.get(key)
            if actual != expected:
                failures.append(
                    f"round {r}: {key} = {actual!r}, expected {expected!r}")
            checked += 1

        got_order = meta.get("dataset_order") or []
        if got_order != order:
            failures.append(
                f"round {r}: dataset_order = {got_order} != cell order {order}"
                + ("  [checkpoint predates the dataset_order field]"
                   if not got_order else ""))
        stop = meta.get("stop_after_task")
        want_stop = order[args.expect_rounds - 1] if args.single_process else order[r]
        if stop != want_stop:
            failures.append(
                f"round {r}: stop_after_task = {stop!r}, expected {want_stop!r}")
        checked += 2

        want_fraction = float(cell.get("kd_init_step_fraction", 1.0))
        got_fraction = (meta.get("v2") or {}).get("kd_init_step_fraction")
        if got_fraction is None or abs(float(got_fraction) - want_fraction) > 1e-9:
            failures.append(
                f"round {r}: v2.kd_init_step_fraction = {got_fraction!r}, "
                f"expected {want_fraction}"
                + ("  [checkpoint predates the KD-fraction field]"
                   if got_fraction is None else ""))
        checked += 1

        ablation = meta.get("ablation") or {}
        for key, expected in (("phase_mode", cell["phase"]),
                              ("kd_init", cell["kd_init"]),
                              ("replay_source", cell["replay_source"])):
            actual = ablation.get(key)
            if actual != expected:
                failures.append(
                    f"round {r}: ablation.{key} = {actual!r}, expected {expected!r}")
            checked += 1

        if r == 0:
            print(f"  round 0 meta: num_experts={meta.get('num_experts')} "
                  f"r={meta.get('r')} alpha={meta.get('alpha')} "
                  f"top_k={meta.get('top_k')} order[0]={stop}")

    if failures:
        print(f"\n{len(failures)} FAILURE(S) of {checked} checks:")
        for line in failures:
            print(f"  FAIL {line}")
        return 1
    print(f"\nALL CHECKS PASSED ({checked} assertions over "
          f"{args.expect_rounds} rounds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
