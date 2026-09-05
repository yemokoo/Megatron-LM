#!/usr/bin/env python3
"""Run a long Ours LoRA-MoE cell as parallel sample shards."""

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
MODEL = Path(os.environ.get(
    "SLORA_LLAMA31_PATH", ROOT / "models/Llama-3.1-8B-Instruct"))
DATA = Path(os.environ.get("TRACE_DATA_ROOT", ROOT / "data/trace"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["ours_lora_moe_v1", "ours_lora_moe_v2",
                 "ours_lora_moe_v2_new", "ours_lora_moe_v2_new_top4",
                 "ours_lora_moe_v2_5",
                 "ours_lora_moe_v3", "ours_lora_moe_v3_new",
                 "ours_lora_moe_v3_new_top4"],
    )
    parser.add_argument("--run-dir")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--round", type=int, choices=[3, 4, 6, 8], required=True)
    parser.add_argument("--task", choices=["MeetingBank", "Py150"], required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--output-subdir", default=None,
        help="Override evaluation/orderN with this run-relative output path.")
    parser.add_argument(
        "--force-expert-index", type=int, default=None,
        help="Force every token through one FFN LoRA expert at weight 1.")
    args = parser.parse_args()
    task = args.task
    if not args.run_dir and not args.method:
        parser.error("one of --run-dir or --method is required")
    run_dir = (Path(args.run_dir).resolve() if args.run_dir else
               ROOT / "results/full_runs/llama31" / args.method)
    gpus = args.gpus.split(",")
    if not gpus or any(not gpu.strip() for gpu in gpus):
        parser.error("--gpus must contain one or more GPU ids")
    num_shards = len(gpus)
    out_dir = (
        run_dir / args.output_subdir
        if args.output_subdir else
        run_dir / "evaluation" / f"order{args.round}")
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_count = len(json.loads((DATA / task / "test.json").read_text()))

    def shard_complete(shard):
        result_path = out_dir / f"results-{task}.shard{shard}.json"
        if not result_path.is_file():
            return False
        try:
            payload = json.loads(result_path.read_text())
        except (OSError, ValueError):
            return False
        expected_indices = list(range(shard, expected_count, num_shards))
        return payload.get("sample_indices") == expected_indices

    processes, logs = [], []
    for shard, gpu in enumerate(gpus):
        if shard_complete(shard):
            print(f"[OURS {task} SHARD SKIP] shard={shard} complete", flush=True)
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        log = (out_dir / f"{task}.shard{shard}.log").open("ab")
        command = [
            str(PYTHON), "evaluate_Ours_LoRA_MoE.py",
            "--checkpoint_dir", str(run_dir / str(args.round - 1)),
            "--base_model_name_or_path", str(MODEL),
            "--data_path", str(DATA), "--inference_tasks", task,
            "--inference_output_path", str(out_dir),
            "--result_suffix", f".shard{shard}",
            "--summary_filename", f"{task}.shard{shard}.summary.json",
            "--num_sample_shards", str(num_shards),
            "--sample_shard_id", str(shard),
            "--max_prompt_len", "0", "--max_ans_len", "1024",
            "--no-task_generation_limits", "--slora_conv_mode", "llama3",
            "--per_device_eval_batch_size", str(args.batch),
            "--temperature", "0",
        ]
        if args.force_expert_index is not None:
            command.extend([
                "--force_expert_index", str(args.force_expert_index)])
        process = subprocess.Popen(
            command, cwd=LLMCL, env=env, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        logs.append(log)
        print(f"[OURS {task} SHARD START] gpu={gpu} shard={shard} "
              f"pid={process.pid}", flush=True)
    codes = [process.wait() for process in processes]
    for log in logs:
        log.close()
    if any(codes):
        raise SystemExit(f"{task} shard failure: {codes}")

    rows, protocol = [], None
    for shard in range(num_shards):
        part = json.loads(
            (out_dir / f"results-{task}.shard{shard}.json").read_text())
        protocol = part["scoring_protocol"]
        rows.extend(zip(
            part["sample_indices"], part["prompts"], part["results"],
            part["results_scored"], part["labels"]))
    rows.sort(key=lambda row: row[0])
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
    if args.force_expert_index is not None:
        merged["forced_expert_index"] = args.force_expert_index
    (out_dir / f"results-{task}.json").write_text(
        json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"{task}.summary.json").write_text(
        json.dumps({task: result}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[OURS {task} MERGED] {result}", flush=True)


if __name__ == "__main__":
    main()
