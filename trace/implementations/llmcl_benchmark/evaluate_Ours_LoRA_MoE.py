#!/usr/bin/env python
"""HF-generate evaluation for growing LoRA-MoE continual-learning checkpoints.

vllm_eval.py can't load our checkpoints: each FFN is a custom LoRAMoEMLP that
vLLM / a plain from_pretrained doesn't understand. This script keeps vllm_eval.py's
*scoring* untouched (load_task / normalize_predictions / score -> identical
metrics, numbers stay comparable to the paper protocol) and only swaps the
generation backend to a HF `model.generate` loop over a model rebuilt by
model.Ours_LoRA_MoE.load_lora_moe_checkpoint.

Two modes:
  * single checkpoint  : --checkpoint_dir OUTPUT/3          (one grown model)
  * whole CL sequence  : --all_rounds --output_dir OUTPUT   (rounds 0..N-1)
                         -> builds the task x round accuracy matrix + BWT.

Offline note (same as vllm_eval): export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1,
and 20Minuten's SARI is skipped unless --with_sari (needs the HF 'sari' metric).
"""
import argparse
from contextlib import nullcontext
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# rouge's summary-level LCS implementation is recursive. Uniform 1024-token
# generation can produce more segments than Python's default recursion limit.
sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))

from utils.utils import load_hf_tokenizer
from utils.model.model_utils import create_hf_model, resolve_attention_implementation
from utils.eval_generation import generate_predictions as generate_hf_predictions
from model.Ours_LoRA_MoE import (
    force_lora_moe_expert, load_lora_moe_checkpoint)
from model.Ours_LoRA_MoE_V3 import (
    V3_ARCHITECTURE, load_v3_checkpoint)
from model.continual_lora import (
    PAPER_BASELINE_META, load_paper_baseline_checkpoint)
from utils.my_peft import PeftModel as OriginalOLoraPeftModel
# Reuse the benchmark scoring verbatim -- importing vllm_eval does NOT pull vLLM in
# (its `from vllm import ...` lives inside main(), not at module top level).
from vllm_eval import (load_task, normalize_predictions, score, ALL_TASKS,
                       TASK_MAX_NEW_TOKENS, TRACE_SCORING_PROTOCOL,
                       TRACE_STOP_MARKERS)

# Single headline scalar per task, used only for the BWT / forgetting matrix.
# The full metric dict is always kept in the per-task result json regardless.
PRIMARY_METRIC = {
    "C-STANCE": "accuracy", "FOMC": "accuracy", "ScienceQA": "accuracy",
    "NumGLUE-cm": "accuracy", "NumGLUE-ds": "accuracy",
    "Py150": "similarity", "MeetingBank": "rouge-L", "20Minuten": "sari",
}
PRIMARY_SCALE = {task: (1.0 if task in ("Py150", "20Minuten") else 100.0)
                 for task in PRIMARY_METRIC}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir",
                   help="A single grown-model round dir (has pytorch_model.bin + "
                        "lora_moe_meta.json). Mutually exclusive with --all_rounds.")
    p.add_argument("--base_only", action="store_true",
                   help="Evaluate the untouched pretrained base model (zero-shot). "
                        "Requires --base_model_name_or_path and no checkpoint.")
    p.add_argument("--all_rounds", action="store_true",
                   help="Evaluate every numeric round subfolder under --output_dir "
                        "and build the CL accuracy matrix + BWT.")
    p.add_argument("--output_dir",
                   help="Training output root containing round subfolders 0,1,... "
                        "(required with --all_rounds).")
    p.add_argument("--base_model_name_or_path", default=None,
                   help="Original pretrained model. Recommended: gives the correct "
                        "base weights + a strict load. If omitted, the skeleton is "
                        "built from the checkpoint's own config.json.")
    p.add_argument("--data_path", required=True,
                   help="Root dir with one subfolder per task holding test.json.")
    p.add_argument("--inference_tasks", default=",".join(ALL_TASKS),
                   help="Comma-separated task names. Default: all 8.")
    p.add_argument("--inference_output_path", required=True,
                   help="Where to write per-task result json + summary.json.")
    p.add_argument("--max_prompt_len", type=int, default=1024,
                   help="Prompt cutoff; 0 disables truncation like released SLoRA eval.")
    p.add_argument(
        "--slora_conv_mode", choices=["none", "llama3", "llama3_template", "qwen", "qwen3"],
        default="none",
        help="Format evaluation inputs exactly like released SLoRA TRACE eval.")
    p.add_argument("--max_ans_len", type=int, default=512)
    p.add_argument(
        "--task_generation_limits", action=argparse.BooleanOptionalAction,
        default=True,
        help="Use TRACE task-specific max-new-token ceilings instead of 512 for "
             "every task. The uniform --max_ans_len remains an upper bound.")
    p.add_argument("--per_device_eval_batch_size", type=int, default=64,
                   help="Bigger is much faster for HF generate (we can't use vLLM -- the "
                        "custom LoRA-MoE FFN isn't a vLLM-known arch). GPU has headroom.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 = greedy (reproducible). Paper used 0.1 w/ sampling.")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument(
        "--device_map", default="none",
        choices=["none", "balanced", "balanced_low_0", "auto", "sequential"],
        help="Shard the model over all visible GPUs through Accelerate. Use "
             "balanced for two-GPU evaluation; none keeps the legacy one-GPU path.")
    p.add_argument("--with_sari", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Compute offline local SARI for 20Minuten (default: on).")
    p.add_argument("--bos_guard", action="store_true",
                   help="install scripts/residual/bos_guard on the loaded V3 model (must match training)")
    p.add_argument("--guard_header", action="store_true",
                   help="with --bos_guard: also guard the fixed chat-template header")
    p.add_argument("--guard_decision", choices=("nn", "end_header", "none"), default="nn",
                   help="header guard decision token (see scripts/bos_token/train_bos_token.py)")
    p.add_argument("--limit", type=int, default=None,
                   help="Debug: only evaluate the first N samples per task.")
    p.add_argument("--limit_frac", type=float, default=None,
                   help="Evaluate a strided fraction (0<f<=1) of each task's test set "
                        "for a fast approximate score, e.g. 0.25. Strided (every k-th "
                        "sample) not head-N, to avoid bias if the set is label-ordered.")
    p.add_argument("--num_sample_shards", type=int, default=1,
                   help="Split each task across this many independent workers.")
    p.add_argument("--sample_shard_id", type=int, default=0,
                   help="Zero-based sample shard assigned to this worker.")
    p.add_argument("--result_suffix", default="",
                   help="Suffix before .json for collision-free shard outputs.")
    p.add_argument(
        "--length_bucketing", action=argparse.BooleanOptionalAction, default=True,
        help="Group similar token lengths into generation batches while restoring "
             "the original prediction order (default: enabled).")
    p.add_argument(
        "--trace_generation_stops",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop each generated row as soon as its TRACE scoring marker is "
             "emitted. Opt-in and score-preserving: post-hoc normalization, "
             "EOS/EOT tokens, and max-new-token ceilings remain unchanged.")
    p.add_argument(
        "--summary_filename", default="summary.json",
        help="Summary filename inside each output directory. Parallel workers can "
        "use distinct names to avoid concurrent writes.")
    p.add_argument(
        "--force_expert_index", type=int, default=None,
        help="Diagnostic for FFN-only LoRA-MoE: route every token through this "
             "single expert at weight 1, bypassing learned router selection.")
    return p.parse_args()


_TEMPLATE_TOK = None


def load_eval_tokenizer(model_name_or_path, slora_conv_mode):
    global _TEMPLATE_TOK
    if slora_conv_mode == "none":
        return load_hf_tokenizer(model_name_or_path, fast_tokenizer=True)
    # Released SLoRA builder uses the slow AutoTokenizer, preserves an existing
    # Qwen pad token, falls back to EOS only when no pad token is configured,
    # and then switches to left padding for batched generation.
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=False, trust_remote_code=True,
        local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    _TEMPLATE_TOK = tokenizer
    return tokenizer


def format_slora_trace_eval_prompt(prompt, task, conv_mode):
    suffix = ""
    if task == "Py150":
        suffix = ("\n\nPlease output only the next line of code, without adding "
                  "anything else. Do not include explanations or comments. "
                  "Treat <EOL> as a line break in the original code.")
    elif task in ("NumGLUE-cm", "NumGLUE-ds"):
        suffix = ("\n\nSolve the math problem and output only the final answer. "
                  "Do not include any explanation, reasoning, or extra words.")
    user_text = prompt + suffix
    if conv_mode == "llama3_template":
        # exactly the training-time framing (SLoRATraceDataCollator: apply_chat_template with the
        # default system prompt -> includes the Cutting Knowledge/Today Date lines) so a header
        # guard trained on that header matches at evaluation time.  The template text starts with
        # one <|begin_of_text|>; the collator tokenizes it with add_special_tokens=True which adds a
        # second one, and generate_predictions below does the same for this mode.
        return _TEMPLATE_TOK.apply_chat_template(
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": user_text}],
            tokenize=False, add_generation_prompt=True)
    if conv_mode == "llama3":
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            + user_text
            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    if conv_mode in ("qwen", "qwen3"):
        text = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n" + user_text + "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        # Qwen3 think-off: the chat template renders every trained assistant
        # turn with an empty think block, so generation must start after it.
        if conv_mode == "qwen3":
            text += "<think>\n\n</think>\n\n"
        return text
    return prompt


def generate_predictions(model, tokenizer, prompts, args, device):
    task = getattr(args, "_cur_task", "")
    max_new_tokens = args.max_ans_len
    if args.task_generation_limits:
        max_new_tokens = min(max_new_tokens, TASK_MAX_NEW_TOKENS.get(task, max_new_tokens))
    stop_token_ids = None
    if args.slora_conv_mode == "llama3":
        stop_token_ids = [tokenizer.eos_token_id]
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_id, int) and eot_id >= 0:
            stop_token_ids.append(eot_id)
    elif args.slora_conv_mode in ("qwen", "qwen3"):
        stop_token_ids = [tokenizer.eos_token_id]
    stop_strings = None
    if args.trace_generation_stops:
        marker = TRACE_STOP_MARKERS.get(task)
        if marker is not None:
            stop_strings = [marker]
    return generate_hf_predictions(
        model,
        tokenizer,
        prompts,
        device=device,
        batch_size=args.per_device_eval_batch_size,
        max_prompt_len=args.max_prompt_len,
        max_new_tokens=max_new_tokens,
        temperature=args.temperature,
        eos_token_ids=stop_token_ids,
        stop_strings=stop_strings,
        length_bucketing=args.length_bucketing,
        description=f"gen[{task},max_new={max_new_tokens}]",
    )


def evaluate_checkpoint(model, tokenizer, args, tasks, device, out_subdir):
    os.makedirs(out_subdir, exist_ok=True)
    summary = {}
    for task in tasks:
        prompts, gts = load_task(args.data_path, task)
        sample_indices = list(range(len(prompts)))
        if args.limit_frac and 0 < args.limit_frac < 1:
            k = max(1, round(1 / args.limit_frac))  # keep every k-th sample
            n0 = len(prompts)
            prompts, gts = prompts[::k], gts[::k]
            sample_indices = sample_indices[::k]
            print(f"[{task}] limit_frac={args.limit_frac} -> strided every {k}: "
                  f"{n0} -> {len(prompts)} samples", flush=True)
        if args.limit:
            prompts, gts = prompts[:args.limit], gts[:args.limit]
            sample_indices = sample_indices[:args.limit]
        if args.num_sample_shards > 1:
            shard = slice(args.sample_shard_id, None, args.num_sample_shards)
            prompts, gts = prompts[shard], gts[shard]
            sample_indices = sample_indices[shard]
            print(
                f"[{task}] sample shard {args.sample_shard_id + 1}/"
                f"{args.num_sample_shards}: {len(prompts)} samples",
                flush=True)

        args._cur_task = task  # label the tqdm bar
        if task == "ScienceQA":
            args.per_device_eval_batch_size = max(
                args.per_device_eval_batch_size, 64)
        task_max_new = min(
            args.max_ans_len,
            TASK_MAX_NEW_TOKENS.get(task, args.max_ans_len),
        ) if args.task_generation_limits else args.max_ans_len
        active_marker = (
            TRACE_STOP_MARKERS.get(task)
            if args.trace_generation_stops else None
        )
        print(f"[{task}] evaluating {len(prompts)} samples "
              f"(batch {args.per_device_eval_batch_size}, "
              f"max_new_tokens {task_max_new}, "
              f"generation_stop={active_marker!r})...", flush=True)
        generation_prompts = [
            format_slora_trace_eval_prompt(prompt, task, args.slora_conv_mode)
            for prompt in prompts
        ]
        raw_preds = generate_predictions(
            model, tokenizer, generation_prompts, args, device)
        preds = normalize_predictions(task, raw_preds)
        result = score(task, prompts, preds, gts, args.with_sari)
        summary[task] = result
        print(f"[{task}] n={len(preds)} -> {result}", flush=True)

        out = {"eval": result, "prompts": prompts, "results": raw_preds,
               "results_scored": preds, "labels": gts,
               "scoring_protocol": TRACE_SCORING_PROTOCOL,
               "sample_indices": sample_indices}
        result_name = f"results-{task}{args.result_suffix}.json"
        with open(os.path.join(out_subdir, result_name), "w", encoding="utf-8") as f:
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
    if args.num_sample_shards < 1:
        raise SystemExit("--num_sample_shards must be positive")
    if not 0 <= args.sample_shard_id < args.num_sample_shards:
        raise SystemExit(
            "--sample_shard_id must be in [0, num_sample_shards)")
    selected_modes = sum((bool(args.checkpoint_dir), bool(args.all_rounds),
                          bool(args.base_only)))
    if selected_modes != 1:
        raise SystemExit(
            "Pass exactly one of --checkpoint_dir, --all_rounds, or --base_only.")
    if args.all_rounds and not args.output_dir:
        raise SystemExit("--all_rounds requires --output_dir.")
    if args.base_only and not args.base_model_name_or_path:
        raise SystemExit("--base_only requires --base_model_name_or_path.")

    tasks = [t for t in args.inference_tasks.split(",") if t]
    os.makedirs(args.inference_output_path, exist_ok=True)
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)

    if args.base_only:
        tok = load_eval_tokenizer(args.base_model_name_or_path, args.slora_conv_mode)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        device_map = None if args.device_map == "none" else args.device_map
        model = create_hf_model(
            AutoModelForCausalLM,
            args.base_model_name_or_path,
            tok,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation=resolve_attention_implementation("auto"),
            device_map=device_map,
        )
        if device_map is None:
            model.to(device)
        model.eval()
        input_device = model.get_input_embeddings().weight.device
        print(f"Loaded zero-shot base model {args.base_model_name_or_path}; "
              f"input device={input_device}", flush=True)
        evaluate_checkpoint(
            model, tok, args, tasks, input_device, args.inference_output_path)
        return

    def load(ckpt_dir):
        tok = load_eval_tokenizer(
            args.base_model_name_or_path or ckpt_dir, args.slora_conv_mode)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        device_map = None if args.device_map == "none" else args.device_map
        if (os.path.exists(os.path.join(ckpt_dir, "adapter_config.json")) and
                os.path.exists(os.path.join(ckpt_dir, "adapter_model.bin"))):
            if not args.base_model_name_or_path:
                raise SystemExit(
                    "O-LoRA adapter checkpoints require "
                    "--base_model_name_or_path")
            base = create_hf_model(
                AutoModelForCausalLM,
                args.base_model_name_or_path,
                tok,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                attn_implementation=resolve_attention_implementation("auto"),
            )
            model = OriginalOLoraPeftModel.from_pretrained(base, ckpt_dir)
            if device_map is not None:
                raise SystemExit(
                    "Original O-LoRA evaluation currently supports one visible "
                    "GPU per worker; use --device_map none.")
            model.to(device)
            with open(os.path.join(ckpt_dir, "adapter_config.json"),
                      encoding="utf-8") as f:
                meta = json.load(f)
            print(f"Loaded original O-LoRA {ckpt_dir}: r={meta.get('r')} "
                  f"r_sum={meta.get('r_sum')} "
                  f"targets={meta.get('target_modules')}", flush=True)
        elif os.path.exists(os.path.join(ckpt_dir, PAPER_BASELINE_META)):
            if not args.base_model_name_or_path:
                raise SystemExit(
                    "paper baseline checkpoints are partial; pass "
                    "--base_model_name_or_path")
            model, meta = load_paper_baseline_checkpoint(
                ckpt_dir, tok, args.base_model_name_or_path,
                device=device, dtype=dtype, device_map=device_map)
            print(f"Loaded {ckpt_dir}: method={meta['method']} "
                  f"r={meta['r']} alpha={meta['alpha']}", flush=True)
        else:
            meta_path = os.path.join(ckpt_dir, "lora_moe_meta.json")
            with open(meta_path, encoding="utf-8") as handle:
                checkpoint_meta = json.load(handle)
            if checkpoint_meta.get("architecture") == V3_ARCHITECTURE:
                # picks the residual-aware loader for checkpoints trained
                # with a residual row from task 0 (scripts/residual/)
                sys.path.insert(0, os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "scripts", "residual"))
                from residual_expert import load_v3_any_checkpoint
                model, meta = load_v3_any_checkpoint(
                    ckpt_dir, tok, args.base_model_name_or_path,
                    device=device, dtype=dtype, device_map=device_map)
                if args.bos_guard:
                    import bos_guard
                    if args.guard_header:
                        sys.path.insert(0, os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "scripts", "bos_token"))
                        from train_bos_token import install_header_guard
                        install_header_guard(model, tok, args.guard_decision)
                    else:
                        bos_guard.install_bos_guard(model)
                    print(f"bos_guard ON header={args.guard_header} decision={args.guard_decision}", flush=True)
            else:
                model, meta = load_lora_moe_checkpoint(
                    ckpt_dir, tok,
                    base_model_name_or_path=args.base_model_name_or_path,
                    device=device, dtype=dtype, device_map=device_map)
            print(f"Loaded {ckpt_dir}: num_experts={meta['num_experts']} "
                  f"r={meta['r']} alpha={meta['alpha']} "
                  f"top_k={meta['top_k']} "
                  f"architecture={meta.get('architecture', 'ffn_only')}",
                  flush=True)
        input_device = model.get_input_embeddings().weight.device
        print(f"Model input device={input_device}; device_map="
              f"{getattr(model, 'hf_device_map', None)}", flush=True)
        return model, tok, input_device

    if args.checkpoint_dir:
        model, tok, input_device = load(args.checkpoint_dir)
        routing_context = (
            force_lora_moe_expert(model, args.force_expert_index)
            if args.force_expert_index is not None else nullcontext())
        if args.force_expert_index is not None:
            print(f"Forcing every token to expert {args.force_expert_index} "
                  "with dispatch weight 1", flush=True)
        with routing_context:
            evaluate_checkpoint(
                model, tok, args, tasks, input_device,
                args.inference_output_path)
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
