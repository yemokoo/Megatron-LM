#!/usr/bin/env python
"""TRACE eval of a router-tuned (optionally residual-expert) V3 checkpoint.

Same generation/scoring as evaluate_Ours_LoRA_MoE.py (its parse_args and
evaluate_checkpoint are reused verbatim); only model construction differs.

  python eval_trace_router_tuned.py --checkpoint_dir <arm dir> \
     --base_model_name_or_path ... --data_path ... --inference_tasks C-STANCE,FOMC \
     --inference_output_path OUT --max_prompt_len 0 --max_ans_len 1024 \
     --no-task_generation_limits --slora_conv_mode llama3 --per_device_eval_batch_size 16 --temperature 0
"""
import sys

TRACE = "/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace"
sys.path.insert(0, f"{TRACE}/implementations/llmcl_benchmark")
sys.path.insert(0, f"{TRACE}/scripts/residual")

import torch  # noqa: E402

import evaluate_Ours_LoRA_MoE as E  # noqa: E402
import residual_expert as RE  # noqa: E402


def main():
    args = E.parse_args()
    tok = E.load_eval_tokenizer(args.base_model_name_or_path, args.slora_conv_mode)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    import json, os
    dtype = getattr(torch, args.dtype)
    v3meta = os.path.join(args.checkpoint_dir, "lora_moe_meta.json")
    if os.path.exists(os.path.join(args.checkpoint_dir, RE.RESIDUAL_META)):
        model, meta = RE.load_router_tuned(args.checkpoint_dir, tok, args.base_model_name_or_path,
                                           device="cuda", dtype=dtype)
    elif os.path.exists(v3meta) and "residual_expert" in json.load(open(v3meta)):
        model, meta = RE.load_v3_residual_checkpoint(args.checkpoint_dir, tok,
                                                     args.base_model_name_or_path, dtype=dtype)
        meta["n_residual"] = 1
        meta["source_checkpoint"] = args.checkpoint_dir
    else:
        model, meta = E.load_v3_checkpoint(args.checkpoint_dir, tok,
                                           base_model_name_or_path=args.base_model_name_or_path,
                                           device="cuda", dtype=dtype, device_map=None)
        meta["n_residual"] = 0
        meta["source_checkpoint"] = args.checkpoint_dir
    print(f"Loaded router-tuned {args.checkpoint_dir}: n_residual={meta.get('n_residual')} "
          f"source={meta.get('source_checkpoint')}", flush=True)
    tasks = [t for t in args.inference_tasks.split(",") if t]
    E.evaluate_checkpoint(model, tok, args, tasks, model.get_input_embeddings().weight.device,
                          args.inference_output_path)


if __name__ == "__main__":
    main()
