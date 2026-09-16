#!/usr/bin/env python
"""Build sparse15_summary.json for an S-LoRA rank-sweep run from the raw
eval_trace_pertask.sh per-cell logs (which, unlike run_tab1_sparse15.py, do not
write a summary file themselves).

  usage: python summarize_slora_sparse15.py <run_dir>   # run_dir/llama31/pre must exist
Writes <run_dir>/sparse15_summary.json with the same final_average/BWT keys the
rest of the tooling (progress-log AA/F prints, comparison scripts) expects.
"""
import json
import re
import sys

TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
         "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]


def score(path, task):
    try:
        text = open(path, errors="ignore").read()
    except FileNotFoundError:
        return None
    m = re.findall(r"In %s: \{[^}]*\}" % re.escape(task), text)
    if not m:
        return None
    d = dict(re.findall(r"'(\w[\w-]*)': ([0-9.]+)", m[-1]))
    if task == "MeetingBank":
        return float(d["rouge-l"]) * 100
    if task == "Py150":
        return float(d["similarity"])
    if task == "20Minuten":
        return float(d["sari"])
    return float(d["accuracy"]) * 100


def main():
    run_dir = sys.argv[1]
    ev = f"{run_dir}/llama31/pre/evaluation"
    diag = [score(f"{ev}/order{i+1}/{t}/eval.log", t) for i, t in enumerate(TASKS)]
    final = [score(f"{ev}/order8/{t}/eval.log", t) for t in TASKS]
    missing_diag = [t for t, v in zip(TASKS, diag) if v is None]
    missing_final = [t for t, v in zip(TASKS, final) if v is None]
    if missing_final:
        print(f"[ERROR] missing order8 scores: {missing_final}", file=sys.stderr)
        sys.exit(1)
    aa = sum(final) / 8
    if missing_diag:
        print(f"[WARN] missing diagonal scores (F skipped): {missing_diag}", file=sys.stderr)
        bwt = None
    else:
        bwt = -(sum(diag[i] - final[i] for i in range(7)) / 7)
    out = {
        "schema_version": 1,
        "evaluation_mode": "sparse_15",
        "final_average": aa,
        "BWT": bwt,
        "diag": dict(zip(TASKS, diag)),
        "final": dict(zip(TASKS, final)),
        "complete": bwt is not None,
    }
    with open(f"{run_dir}/sparse15_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"AA {aa:.2f}" + (f"  F {-bwt:.2f}" if bwt is not None else "  F <incomplete>"))


if __name__ == "__main__":
    main()
