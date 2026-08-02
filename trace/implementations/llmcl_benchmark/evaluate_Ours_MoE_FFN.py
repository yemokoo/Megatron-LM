#!/usr/bin/env python
"""HF-generate evaluation for growing FFN-expert MoE (OLMoE) continual-learning checkpoints.

vllm_eval.py can't load our checkpoints: each FFN is a custom GrowingOlmoeMoE that
vLLM / a plain from_pretrained doesn't understand. This script keeps vllm_eval.py's
*scoring* untouched (load_task / normalize_predictions / score -> identical
metrics, numbers stay comparable to the paper protocol) and only swaps the
generation backend to a HF `model.generate` loop over a model rebuilt by
model.Ours_MoE_FFN.load_moe_ffn_checkpoint.

Two modes:
  * single checkpoint  : --checkpoint_dir OUTPUT/3          (one grown model)
  * whole CL sequence  : --all_rounds --output_dir OUTPUT   (rounds 0..N-1)
                         -> builds the task x round accuracy matrix + BWT.

Offline note (same as vllm_eval): export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1,
and 20Minuten's SARI is skipped unless --with_sari (needs the HF 'sari' metric).
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import load_hf_tokenizer
from utils.eval_generation import generate_predictions as generate_hf_predictions
from model.Ours_MoE_FFN import load_moe_ffn_checkpoint
# Reuse the benchmark scoring verbatim -- importing vllm_eval does NOT pull vLLM in
# (its `from vllm import ...` lives inside main(), not at module top level).
from vllm_eval import (load_task, normalize_predictions, score, ALL_TASKS,
                       TASK_MAX_NEW_TOKENS, TRACE_SCORING_PROTOCOL)

# Single headline scalar per task, used only for the BWT / forgetting matrix.
# The full metric dict is always kept in the per-task result json regardless.
PRIMARY_METRIC = {
    "C-STANCE": "accuracy", "FOMC": "accuracy", "ScienceQA": "accuracy",
    "NumGLUE-cm": "accuracy", "NumGLUE-ds": "accuracy",
    "Py150": "similarity", "MeetingBank": "rouge-L", "20Minuten": "rouge-L",
}
PRIMARY_SCALE = {task: (1.0 if task == "Py150" else 100.0)
                 for task in PRIMARY_METRIC}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir",
                   help="A single grown-model round dir (has pytorch_model.bin + "
                        "moe_ffn_meta.json). Mutually exclusive with --all_rounds.")
    p.add_argument("--all_rounds", action="store_true",
                   help="Evaluate every numeric round subfolder under --output_dir "
                        "and build the CL accuracy matrix + BWT.")
    p.add_argument("--output_dir",
                   help="Training output root containing round subfolders 0,1,... "
                        "(required with --all_rounds).")
    p.add_argument("--base_model_name_or_path", default=None,
                   help="Original pretrained model. Required for delta checkpoints "
                        "unless the recorded training-time path is still valid.")
    p.add_argument("--data_path", required=True,
                   help="Root dir with one subfolder per task holding test.json.")
    p.add_argument("--inference_tasks", default=",".join(ALL_TASKS),
                   help="Comma-separated task names. Default: all 8.")
    p.add_argument("--inference_output_path", required=True,
                   help="Where to write per-task result json + summary.json.")
    p.add_argument("--max_prompt_len", type=int, default=1024)
    p.add_argument("--max_ans_len", type=int, default=512)
    p.add_argument(
        "--task_generation_limits", action=argparse.BooleanOptionalAction,
        default=True,
        help="Use TRACE task-specific max-new-token ceilings instead of 512 for "
             "every task. The uniform --max_ans_len remains an upper bound.")
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 = greedy (reproducible). Paper used 0.1 w/ sampling.")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_implementation", default="auto",
                   choices=["auto", "flash_attention_2", "sdpa", "eager"],
                   help="auto uses FlashAttention-2 when installed, otherwise SDPA.")
    p.add_argument(
        "--device_map", default="none",
        choices=["none", "balanced", "balanced_low_0", "auto", "sequential"],
        help="Shard the model over all visible GPUs through Accelerate. Use "
             "balanced for two-GPU evaluation; none keeps the legacy one-GPU path.")
    p.add_argument("--with_sari", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Compute offline local SARI for 20Minuten (default: on).")
    p.add_argument("--limit", type=int, default=None,
                   help="Debug: only evaluate the first N samples per task.")
    p.add_argument(
        "--length_bucketing", action=argparse.BooleanOptionalAction, default=True,
        help="Group similar token lengths into generation batches while restoring "
             "the original prediction order (default: enabled).")
    p.add_argument(
        "--summary_filename", default="summary.json",
        help="Summary filename inside each output directory. Parallel workers can "
             "use distinct names to avoid concurrent writes.")
    return p.parse_args()


def generate_predictions(model, tokenizer, prompts, args, device):
    task = getattr(args, "_cur_task", "")
    max_new_tokens = args.max_ans_len
    if args.task_generation_limits:
        max_new_tokens = min(max_new_tokens, TASK_MAX_NEW_TOKENS.get(task, max_new_tokens))
    return generate_hf_predictions(
        model,
        tokenizer,
        prompts,
        device=device,
        batch_size=args.per_device_eval_batch_size,
        max_prompt_len=args.max_prompt_len,
        max_new_tokens=max_new_tokens,
        temperature=args.temperature,
        length_bucketing=args.length_bucketing,
        description=f"gen[{task},max_new={max_new_tokens}]",
    )


def evaluate_checkpoint(model, tokenizer, args, tasks, device, out_subdir):
    os.makedirs(out_subdir, exist_ok=True)
    summary = {}
    for task in tasks:
        prompts, gts = load_task(args.data_path, task)
        if args.limit:
            prompts, gts = prompts[:args.limit], gts[:args.limit]

        args._cur_task = task
        task_max_new = min(
            args.max_ans_len,
            TASK_MAX_NEW_TOKENS.get(task, args.max_ans_len),
        ) if args.task_generation_limits else args.max_ans_len
        print(f"[{task}] evaluating {len(prompts)} samples "
              f"(batch {args.per_device_eval_batch_size}, "
              f"max_new_tokens {task_max_new})...", flush=True)
        raw_preds = generate_predictions(model, tokenizer, prompts, args, device)
        preds = normalize_predictions(task, raw_preds)
        result = score(task, prompts, preds, gts, args.with_sari)
        summary[task] = result
        print(f"[{task}] n={len(preds)} -> {result}", flush=True)

        out = {"eval": result, "prompts": prompts, "results": raw_preds,
               "results_scored": preds, "labels": gts,
               "scoring_protocol": TRACE_SCORING_PROTOCOL}
        with open(os.path.join(out_subdir, f"results-{task}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)

    with open(os.path.join(out_subdir, args.summary_filename), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def primary_scalar(task, result):
    if not isinstance(result, dict):
        return None
    key = PRIMARY_METRIC.get(task)
    val = result.get(key) if key else None
    if isinstance(val, (int, float)):
        return val * PRIMARY_SCALE.get(task, 1.0)
    # Fallback: first numeric value in the dict.
    for v in result.values():
        if isinstance(v, (int, float)):
            return v * PRIMARY_SCALE.get(task, 1.0)
    return None


def compute_bwt(matrix, round_ids, task_per_round, tasks):
    """matrix[round_id][task] = primary scalar. task_per_round[i] is the task
    trained at round round_ids[i]. BWT = mean over every task trained before the
    final round of (final_score - score_right_after_that_task_was_trained)."""
    final_rid = round_ids[-1]
    bwt_terms = {}
    for i, rid in enumerate(round_ids[:-1]):
        task = task_per_round[i]
        if task not in tasks:
            continue
        after_learn = matrix.get(rid, {}).get(task)
        final = matrix.get(final_rid, {}).get(task)
        if after_learn is not None and final is not None:
            bwt_terms[task] = final - after_learn
    bwt = sum(bwt_terms.values()) / len(bwt_terms) if bwt_terms else None
    return bwt, bwt_terms


def main():
    args = parse_args()
    if bool(args.checkpoint_dir) == bool(args.all_rounds):
        raise SystemExit("Pass exactly one of --checkpoint_dir or --all_rounds.")
    if args.all_rounds and not args.output_dir:
        raise SystemExit("--all_rounds requires --output_dir.")

    tasks = [t for t in args.inference_tasks.split(",") if t]
    os.makedirs(args.inference_output_path, exist_ok=True)
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)

    def load(ckpt_dir):
        tok = load_hf_tokenizer(args.base_model_name_or_path or ckpt_dir, fast_tokenizer=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model, meta = load_moe_ffn_checkpoint(
            ckpt_dir, tok, base_model_name_or_path=args.base_model_name_or_path,
            device=device, dtype=dtype,
            attn_implementation=args.attn_implementation,
            device_map=None if args.device_map == "none" else args.device_map)
        print(f"Loaded {ckpt_dir}: num_new_experts={meta['num_new_experts']} "
              f"(base={meta.get('num_base_experts', '?')})", flush=True)
        input_device = model.get_input_embeddings().weight.device
        print(f"Model input device={input_device}; device_map="
              f"{getattr(model, 'hf_device_map', None)}", flush=True)
        return model, tok, input_device

    if args.checkpoint_dir:
        model, tok, input_device = load(args.checkpoint_dir)
        evaluate_checkpoint(model, tok, args, tasks, input_device, args.inference_output_path)
        return

    # --all_rounds: evaluate each grown round on all tasks -> CL matrix.
    round_dirs = sorted(
        (d for d in os.listdir(args.output_dir)
         if d.isdigit() and os.path.isdir(os.path.join(args.output_dir, d))),
        key=int)
    if not round_dirs:
        raise SystemExit(f"No numeric round subfolders under {args.output_dir}.")

    matrix = {}
    for rd in round_dirs:
        ckpt_dir = os.path.join(args.output_dir, rd)
        model, tok, input_device = load(ckpt_dir)
        out_subdir = os.path.join(args.inference_output_path, f"round_{rd}")
        summary = evaluate_checkpoint(model, tok, args, tasks, input_device, out_subdir)
        matrix[rd] = {t: primary_scalar(t, r) for t, r in summary.items()}
        del model
        torch.cuda.empty_cache()

    # round i trained the i-th task of the run; recover that ordering from the data
    # folders present, defaulting to the tasks list order if it matches the count.
    cl = {"matrix": matrix, "primary_metric": PRIMARY_METRIC}
    # BWT needs the task-per-round mapping. We can only infer it when the number of
    # tasks being evaluated equals the number of rounds (i.e. --inference_tasks is
    # the full training sequence, in training order). Otherwise the matrix is still
    # written; BWT is just skipped.
    ordered = tasks if len(tasks) == len(round_dirs) else None
    if ordered:
        bwt, bwt_terms = compute_bwt(matrix, round_dirs, ordered, tasks)
        final_scores = {t: matrix[round_dirs[-1]].get(t) for t in tasks}
        cl.update({"task_per_round": ordered, "bwt": bwt, "bwt_per_task": bwt_terms,
                   "final_scores": final_scores,
                   "final_avg": _mean([v for v in final_scores.values() if v is not None])})

    with open(os.path.join(args.inference_output_path, "cl_summary.json"), "w", encoding="utf-8") as f:
        json.dump(cl, f, ensure_ascii=False, indent=2)
    print("\n================ CL SUMMARY ================")
    print(json.dumps(cl, ensure_ascii=False, indent=2))
    print(f"\nSaved to {args.inference_output_path}/cl_summary.json")


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


if __name__ == "__main__":
    main()
