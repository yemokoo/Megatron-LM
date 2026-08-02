#!/usr/bin/env python
"""Build TRACE OP/BWT tables from the efficient 15-evaluation layout."""

import argparse
import csv
import json
from pathlib import Path


TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
         "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]
DEFAULT_METHODS = ["seqlora", "loramoe", "ewc", "gem", "olora"]
PRIMARY_METRIC = {
    "C-STANCE": "accuracy", "FOMC": "accuracy",
    "MeetingBank": "rouge-L", "Py150": "similarity",
    "ScienceQA": "accuracy", "NumGLUE-cm": "accuracy",
    "NumGLUE-ds": "accuracy", "20Minuten": "sari",
}


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                        help="Comma-separated methods to include in the table.")
    return parser.parse_args()


def load_payload(path: Path, task: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    lengths = [len(payload.get(key, []))
               for key in ("prompts", "results", "results_scored", "labels")]
    if not lengths[0] or len(set(lengths)) != 1:
        raise ValueError(f"incomplete result {path}: lengths={lengths}")
    metric = PRIMARY_METRIC[task]
    value = payload.get("eval", {}).get(metric)
    if not isinstance(value, (int, float)):
        raise ValueError(f"missing numeric {metric} in {path}")
    # TRACE task scores are reported consistently on a 0..100 scale. Py150's
    # fuzzy similarity is already 0..100; accuracy/ROUGE are stored as 0..1.
    return float(value if task in ("Py150", "20Minuten") else value * 100.0), payload["eval"]


def main():
    args = args_parser()
    methods = [item for item in args.methods.split(",") if item]
    root = Path(args.eval_root)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "protocol": {
            "benchmark": "TRACE",
            "task_order": TASKS,
            "score_scale": "0..100",
            "op": "mean final score over all 8 tasks",
            "bwt": "mean(final score - score immediately after learning) over tasks 0..6",
            "primary_metric": PRIMARY_METRIC,
        },
        "methods": {},
    }
    csv_rows = []
    for method in methods:
        final_scores, learned_scores, full_metrics = {}, {}, {}
        for index, task in enumerate(TASKS):
            final_path = root / method / "final" / f"results-{task}.json"
            final_scores[task], full_metrics[task] = load_payload(final_path, task)
            if index < len(TASKS) - 1:
                diagonal_path = (root / method / "diagonal" / str(index) /
                                 f"results-{task}.json")
                learned_scores[task], _ = load_payload(diagonal_path, task)
        bwt_per_task = {
            task: final_scores[task] - learned_scores[task]
            for task in TASKS[:-1]
        }
        op = sum(final_scores.values()) / len(TASKS)
        bwt = sum(bwt_per_task.values()) / len(bwt_per_task)
        result["methods"][method] = {
            "OP": op, "BWT": bwt,
            "final_scores": final_scores,
            "learned_scores": learned_scores,
            "bwt_per_task": bwt_per_task,
            "full_final_metrics": full_metrics,
        }
        row = {"method": method, "OP": op, "BWT": bwt}
        row.update({task: final_scores[task] for task in TASKS})
        csv_rows.append(row)

    Path(f"{prefix}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["method", "OP", "BWT", *TASKS]
    with open(f"{prefix}.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        "# TRACE five-baseline results",
        "",
        "All values are on a 0–100 scale; BWT is in percentage points.",
        "",
        "| Method | OP ↑ | BWT ↑ | " + " | ".join(TASKS) + " |",
        "|---|---:|---:|" + "---:|" * len(TASKS),
    ]
    for row in csv_rows:
        lines.append(
            f"| {row['method']} | {row['OP']:.2f} | {row['BWT']:.2f} | " +
            " | ".join(f"{row[task]:.2f}" for task in TASKS) + " |")
    lines += ["", "## Per-task backward transfer", ""]
    for method in methods:
        values = result["methods"][method]["bwt_per_task"]
        lines.append(f"- {method}: " + ", ".join(
            f"{task} {values[task]:+.2f}" for task in TASKS[:-1]))
    Path(f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({method: {key: result["methods"][method][key]
                                     for key in ("OP", "BWT")}
                      for method in methods}, indent=2))
    print(f"wrote {prefix}.json/.csv/.md")


if __name__ == "__main__":
    main()
