#!/usr/bin/env python
"""Evaluate Table-1 baseline checkpoints on TRACE.

Generation, prompting, scoring, the CL matrix and BWT all come from
``evaluate_Ours_LoRA_MoE`` unchanged -- importing them is the point, since a
second copy of the prompt formatter or the stop-condition logic is exactly how
two rows of the same table stop being comparable.  Only model construction is
different, and that lives in ``model/tab1_checkpoint.py``.

  # one checkpoint
  python evaluate_tab1.py --checkpoint_dir RUN/7 \
      --base_model_name_or_path .../Llama-3.1-8B-Instruct \
      --data_path data/trace --inference_output_path OUT

  # every round -> CL matrix, BWT, final average
  python evaluate_tab1.py --all_rounds --output_dir RUN \
      --base_model_name_or_path .../Llama-3.1-8B-Instruct \
      --data_path data/trace --inference_output_path OUT
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_Ours_LoRA_MoE import (PRIMARY_METRIC, compute_bwt,
                                    evaluate_checkpoint, load_eval_tokenizer,
                                    parse_args, primary_scalar, _mean)
from model.tab1_checkpoint import load_tab1_checkpoint, read_tab1_meta


def load(ckpt_dir, args, dtype):
    tok = load_eval_tokenizer(
        args.base_model_name_or_path or ckpt_dir, args.slora_conv_mode)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    device_map = None if args.device_map == "none" else args.device_map
    if not args.base_model_name_or_path:
        raise SystemExit(
            "Table-1 checkpoints hold adapters only; pass "
            "--base_model_name_or_path")
    model, meta = load_tab1_checkpoint(
        ckpt_dir, tok, args.base_model_name_or_path,
        device=torch.device("cuda"), dtype=dtype, device_map=device_map)
    described = {key: meta.get(key) for key in
                 ("method", "targets", "moe_scope", "num_experts", "r",
                  "alpha", "top_k", "merged") if meta.get(key) is not None}
    print(f"Loaded {ckpt_dir}: {described}", flush=True)
    input_device = model.get_input_embeddings().weight.device
    return model, tok, input_device


def main():
    args = parse_args()
    if bool(args.checkpoint_dir) == bool(args.all_rounds):
        raise SystemExit("Pass exactly one of --checkpoint_dir or --all_rounds.")
    if args.all_rounds and not args.output_dir:
        raise SystemExit("--all_rounds requires --output_dir.")
    tasks = [task for task in args.inference_tasks.split(",") if task]
    os.makedirs(args.inference_output_path, exist_ok=True)
    dtype = getattr(torch, args.dtype)

    if args.checkpoint_dir:
        model, tok, input_device = load(args.checkpoint_dir, args, dtype)
        evaluate_checkpoint(model, tok, args, tasks, input_device,
                            args.inference_output_path)
        return

    round_dirs = sorted(
        (name for name in os.listdir(args.output_dir)
         if name.isdigit()
         and os.path.isdir(os.path.join(args.output_dir, name))
         and os.path.isfile(os.path.join(args.output_dir, name,
                                         "tab1_meta.json"))),
        key=int)
    if not round_dirs:
        raise SystemExit(
            f"No Table-1 round subfolders under {args.output_dir}. "
            "An MTL run saves a single round at the last index.")

    matrix = {}
    for round_name in round_dirs:
        ckpt_dir = os.path.join(args.output_dir, round_name)
        model, tok, input_device = load(ckpt_dir, args, dtype)
        out_subdir = os.path.join(args.inference_output_path,
                                  f"round_{round_name}")
        summary = evaluate_checkpoint(model, tok, args, tasks, input_device,
                                      out_subdir)
        matrix[round_name] = {task: primary_scalar(task, result)
                              for task, result in summary.items()}
        del model
        torch.cuda.empty_cache()

    cl = {"matrix": matrix, "primary_metric": PRIMARY_METRIC,
          "method": read_tab1_meta(
              os.path.join(args.output_dir, round_dirs[-1])).get("method")}
    ordered = tasks if len(tasks) == len(round_dirs) else None
    if ordered:
        bwt, bwt_terms = compute_bwt(matrix, round_dirs, ordered, tasks)
        final_scores = {task: matrix[round_dirs[-1]].get(task) for task in tasks}
        cl.update({"task_per_round": ordered, "bwt": bwt,
                   "bwt_per_task": bwt_terms, "final_scores": final_scores,
                   "final_avg": _mean([value for value in final_scores.values()
                                       if value is not None])})
    with open(os.path.join(args.inference_output_path, "cl_summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump(cl, handle, ensure_ascii=False, indent=2)
    print("\n================ CL SUMMARY ================")
    print(json.dumps(cl, ensure_ascii=False, indent=2))
    print(f"\nSaved to {args.inference_output_path}/cl_summary.json")


if __name__ == "__main__":
    main()
