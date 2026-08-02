#!/usr/bin/env python
"""Merge probe parts and choose common, conservative per-task batches."""
import argparse
import json
import math
import os

TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
         "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]
METHODS = ["seqlora", "loramoe", "ewc", "gem", "olora", "track1"]


def conservative(maximum):
    # Keep 25% headroom for real-data variation and long-running allocator state.
    value = max(1, math.floor(maximum * 0.75))
    return value if value < 8 else (value // 4) * 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    merged = {method: {} for method in METHODS}
    for method in METHODS:
        for task in TASKS:
            path = os.path.join(args.root, "parts", f"{method}__{task}.json")
            with open(path) as handle:
                merged[method][task] = json.load(handle)["results"][task]

    per_method = {
        method: {task: conservative(merged[method][task]["max_batch"])
                 for task in TASKS}
        for method in METHODS
    }
    # Fair comparison: identical effective batch for a given task across all six.
    common = {
        task: min(per_method[method][task] for method in METHODS)
        for task in TASKS
    }
    payload = {"methods": merged, "safe_batch_per_method": per_method,
               "recommended_common_batch": common,
               "recommended_common_csv": ",".join(str(common[t]) for t in TASKS),
               "policy": "75% of measured limit, rounded down to a multiple of 4; "
                         "common minimum across six methods"}
    path = os.path.join(args.root, "summary.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"{'task':12s} " + " ".join(f"{m:>8s}" for m in METHODS) + "  common")
    for task in TASKS:
        maxima = " ".join(f"{merged[m][task]['max_batch']:8d}" for m in METHODS)
        print(f"{task:12s} {maxima}  {common[task]:6d}")
    print("BATCH=" + payload["recommended_common_csv"])
    print("saved " + path)


if __name__ == "__main__":
    main()
