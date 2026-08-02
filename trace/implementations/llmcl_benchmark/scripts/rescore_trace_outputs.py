#!/usr/bin/env python
"""Re-score saved TRACE generations without loading a model or using a GPU."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_eval import (ALL_TASKS, TRACE_SCORING_PROTOCOL,
                       normalize_predictions, score)

PRIMARY_METRIC = {
    "C-STANCE": "accuracy",
    "FOMC": "accuracy",
    "MeetingBank": "rouge-L",
    "Py150": "similarity",
    "ScienceQA": "accuracy",
    "NumGLUE-cm": "accuracy",
    "NumGLUE-ds": "accuracy",
    "20Minuten": "rouge-L",
}
PRIMARY_SCALE = {task: (1.0 if task == "Py150" else 100.0)
                 for task in PRIMARY_METRIC}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument(
        "--write", action="store_true",
        help="Update results-*.json and rebuild summaries in place.")
    parser.add_argument(
        "--with_sari", action="store_true",
        help="Also compute 20Minuten SARI (may require a cached HF metric).")
    return parser.parse_args()


def result_files(root):
    for current, _, files in os.walk(root):
        for task in ALL_TASKS:
            name = f"results-{task}.json"
            if name in files:
                yield current, task, os.path.join(current, name)


def primary_scalar(task, result):
    value = result.get(PRIMARY_METRIC[task]) if isinstance(result, dict) else None
    return (value * PRIMARY_SCALE[task]
            if isinstance(value, (int, float)) else None)


def rebuild_cl_summary(root, summaries):
    rounds = {}
    for directory, summary in summaries.items():
        name = os.path.basename(directory)
        if name.startswith("round_") and name[6:].isdigit():
            rounds[name[6:]] = {
                task: primary_scalar(task, result)
                for task, result in summary.items()
            }
    if not rounds:
        return

    round_ids = sorted(rounds, key=int)
    payload = {
        "matrix": {round_id: rounds[round_id] for round_id in round_ids},
        "primary_metric": PRIMARY_METRIC,
        "scoring_protocol": TRACE_SCORING_PROTOCOL,
    }
    existing_path = os.path.join(root, "cl_summary.json")
    task_order = None
    if os.path.isfile(existing_path):
        with open(existing_path, "r", encoding="utf-8") as handle:
            task_order = json.load(handle).get("task_per_round")
    if task_order is None and len(round_ids) == len(ALL_TASKS):
        task_order = ALL_TASKS
    if task_order and len(task_order) == len(round_ids):
        final_round = round_ids[-1]
        terms = {}
        for index, round_id in enumerate(round_ids[:-1]):
            task = task_order[index]
            learned = rounds[round_id].get(task)
            final = rounds[final_round].get(task)
            if learned is not None and final is not None:
                terms[task] = final - learned
        final_scores = {
            task: rounds[final_round].get(task) for task in task_order
        }
        numeric_final = [value for value in final_scores.values()
                         if isinstance(value, (int, float))]
        payload.update({
            "task_per_round": task_order,
            "bwt": sum(terms.values()) / len(terms) if terms else None,
            "bwt_per_task": terms,
            "final_scores": final_scores,
            "final_avg": (
                sum(numeric_final) / len(numeric_final)
                if numeric_final else None
            ),
        })
    with open(existing_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"wrote {existing_path}")


def main():
    args = parse_args()
    summaries = {}
    found = 0
    for directory, task, path in result_files(args.input_dir):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        prompts = payload["prompts"]
        raw_predictions = payload["results"]
        labels = payload["labels"]
        if not (len(prompts) == len(raw_predictions) == len(labels)):
            raise ValueError(
                f"length mismatch in {path}: prompts={len(prompts)} "
                f"results={len(raw_predictions)} labels={len(labels)}")

        predictions = normalize_predictions(task, raw_predictions)
        result = score(task, prompts, predictions, labels, args.with_sari)
        old_result = payload.get("eval")
        summaries.setdefault(directory, {})[task] = result
        found += 1
        print(f"{path}: {old_result} -> {result}")

        if args.write:
            payload["eval"] = result
            payload["results_scored"] = predictions
            payload["scoring_protocol"] = TRACE_SCORING_PROTOCOL
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)

    if found == 0:
        raise SystemExit(
            f"no results-<task>.json files under {args.input_dir}")

    if args.write:
        for directory, summary in summaries.items():
            summary_path = os.path.join(directory, "summary.json")
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)
            print(f"wrote {summary_path}")
        rebuild_cl_summary(args.input_dir, summaries)
    else:
        print("dry run only; pass --write to update result and summary files")


if __name__ == "__main__":
    main()
