#!/usr/bin/env python3
"""Run a long Ours LoRA-MoE cell as four sample shards."""

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLMCL = ROOT / "implementations/llmcl_benchmark"
PYTHON = Path(os.environ.get("TRACE_PYTHON", ROOT / ".venv-runtime/bin/python"))
MODEL = ROOT / "models/Llama-3.1-8B-Instruct"
DATA = Path(os.environ.get("TRACE_DATA_ROOT", ROOT / "data/trace"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["ours_lora_moe_v1", "ours_lora_moe_v2", "ours_lora_moe_v2_5"],
        required=True,
    )
    parser.add_argument("--round", type=int, choices=[3, 4, 8], required=True)
    parser.add_argument("--task", choices=["MeetingBank", "Py150"], required=True)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    task = args.task
    run_dir = ROOT / "results/full_runs/llama31" / args.method
    out_dir = run_dir / "evaluation" / f"order{args.round}"
    out_dir.mkdir(parents=True, exist_ok=True)

    processes, logs = [], []
    for gpu in range(4):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log = (out_dir / f"{task}.shard{gpu}.log").open("wb")
        command = [
            str(PYTHON), "evaluate_Ours_LoRA_MoE.py",
            "--checkpoint_dir", str(run_dir / str(args.round - 1)),
            "--base_model_name_or_path", str(MODEL),
            "--data_path", str(DATA), "--inference_tasks", task,
            "--inference_output_path", str(out_dir),
            "--result_suffix", f".shard{gpu}",
            "--summary_filename", f"{task}.shard{gpu}.summary.json",
            "--num_sample_shards", "4", "--sample_shard_id", str(gpu),
            "--max_prompt_len", "0", "--max_ans_len", "1024",
            "--no-task_generation_limits", "--slora_conv_mode", "llama3",
            "--per_device_eval_batch_size", str(args.batch),
            "--temperature", "0",
        ]
        process = subprocess.Popen(
            command, cwd=LLMCL, env=env, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        logs.append(log)
        print(f"[OURS {task} SHARD START] gpu={gpu} pid={process.pid}", flush=True)
    codes = [process.wait() for process in processes]
    for log in logs:
        log.close()
    if any(codes):
        raise SystemExit(f"{task} shard failure: {codes}")

    rows, protocol = [], None
    for gpu in range(4):
        part = json.loads(
            (out_dir / f"results-{task}.shard{gpu}.json").read_text())
        protocol = part["scoring_protocol"]
        rows.extend(zip(
            part["sample_indices"], part["prompts"], part["results"],
            part["results_scored"], part["labels"]))
    rows.sort(key=lambda row: row[0])
    expected_count = len(json.loads((DATA / task / "test.json").read_text()))
    if [row[0] for row in rows] != list(range(expected_count)):
        raise SystemExit(f"{task} shard indices are incomplete or duplicated")
    sys.path.insert(0, str(LLMCL))
    evaluator = importlib.import_module("evaluate_Ours_LoRA_MoE")
    prompts = [row[1] for row in rows]
    raw = [row[2] for row in rows]
    scored = [row[3] for row in rows]
    labels = [row[4] for row in rows]
    result = evaluator.score(task, prompts, scored, labels, True)
    merged = {
        "eval": result, "prompts": prompts, "results": raw,
        "results_scored": scored, "labels": labels,
        "scoring_protocol": protocol, "sample_indices": list(range(expected_count)),
    }
    (out_dir / f"results-{task}.json").write_text(
        json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"{task}.summary.json").write_text(
        json.dumps({task: result}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[OURS {task} MERGED] {result}", flush=True)


if __name__ == "__main__":
    main()
