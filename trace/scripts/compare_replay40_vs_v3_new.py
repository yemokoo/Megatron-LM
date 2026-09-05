#!/usr/bin/env python3
"""Live sparse-15 comparison: v3_new (LM baseline) versus v3_new_replay40.

The baseline is complete, so its acquisition (diagonal, round r) and final
(round 8) columns are fixed.  The replay40 columns fill in cell by cell as the
running eval writes results-<task>.json, so this can be re-run at any time.

Delta columns are replay40 minus baseline; for the drop column a positive
number means replay40 forgot less.
"""
import json
import os
import sys

BASE = ("/data2/seonghyeonnoh/LLM-continual-learning-runs/"
        "instruct_priority_fourway_20260812/v3_new/sparse15_summary.json")
RUN = ("/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/v3_replay40/"
       "v3_new_replay40_st_top1_cap2000")

# The baseline summary names each task's metric; the per-cell result file
# stores every metric it computed under "eval".  Map one to the other rather
# than picking the first numeric key, so a task is never scored by the wrong
# metric.  Scores are stored as fractions and reported as percentages.
METRIC_KEY = {
    "accuracy": "accuracy",
    "rouge-l": "rouge-L",
    "similarity": "similarity",
    "sari": "sari",
}


def cell_score(round_id, task, metric):
    path = os.path.join(RUN, "evaluation", f"order{round_id}",
                        f"results-{task}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        payload = json.load(handle)
    scores = payload.get("eval")
    if not isinstance(scores, dict):
        return None
    key = METRIC_KEY.get(metric.lower())
    if key is None or key not in scores:
        raise KeyError(
            f"{task} round {round_id}: metric {metric!r} -> {key!r} missing "
            f"from {sorted(scores)}")
    value = float(scores[key])
    return value * 100.0 if value <= 1.0 else value


def fmt(value, width=7):
    return f"{value:>{width}.2f}" if value is not None else " " * (width - 1) + "-"


def main():
    with open(BASE) as handle:
        base = json.load(handle)
    tasks = base["tasks"]
    diag = base["diagonal_scores_rounds_1_to_7"]
    final = base["final_scores_round_8"]

    print(f"baseline: v3_new  OP {base['OP']:.2f}  BWT {base['BWT']:.2f}")
    print(f"run     : v3_new_replay40 (replay 2x; eval in progress)\n")
    header = (f"{'task':<13}{'metric':<11}"
              f"{'v3 acq':>8}{'v3 fin':>8}{'v3 drop':>9}   "
              f"{'r40 acq':>8}{'r40 fin':>8}{'r40 drop':>9}   "
              f"{'d fin':>7}{'d drop':>8}")
    print(header)
    print("-" * len(header))

    rows, done = [], 0
    for index, task in enumerate(tasks):
        round_id = index + 1
        b_acq = diag[index] if index < len(diag) else None
        b_fin = final[index]
        b_drop = (b_acq - b_fin) if b_acq is not None else None

        metric = base['metric'][task]
        r_acq = (cell_score(round_id, task, metric)
                 if round_id <= 7 else None)
        r_fin = cell_score(8, task, metric)
        r_drop = (r_acq - r_fin) if (r_acq is not None and r_fin is not None) else None
        done += sum(x is not None for x in
                    ((r_acq if round_id <= 7 else None), r_fin))

        d_fin = (r_fin - b_fin) if r_fin is not None else None
        d_drop = (b_drop - r_drop) if (b_drop is not None and r_drop is not None) else None
        rows.append((r_fin, b_fin))
        print(f"{task:<13}{metric:<11}"
              f"{fmt(b_acq,8)}{fmt(b_fin,8)}{fmt(b_drop,9)}   "
              f"{fmt(r_acq,8)}{fmt(r_fin,8)}{fmt(r_drop,9)}   "
              f"{fmt(d_fin,7)}{fmt(d_drop,8)}")

    print("-" * len(header))
    print(f"cells done: {done}/15")
    have = [(r, b) for r, b in rows if r is not None]
    if len(have) == len(tasks):
        op = sum(r for r, _ in have) / len(have)
        print(f"\nOP  v3_new {base['OP']:.2f}   replay40 {op:.2f}   "
              f"delta {op - base['OP']:+.2f}")
    elif have:
        # Partial OP is not comparable to the full-run OP, so compare only the
        # subset that both runs have -- otherwise an easy task finishing first
        # would look like an improvement.
        partial_r = sum(r for r, _ in have) / len(have)
        partial_b = sum(b for _, b in have) / len(have)
        print(f"\npartial mean over the {len(have)} finished FINAL cells: "
              f"v3_new {partial_b:.2f}  replay40 {partial_r:.2f}  "
              f"delta {partial_r - partial_b:+.2f}")
        print("(not the OP -- only the final-row cells that have landed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
