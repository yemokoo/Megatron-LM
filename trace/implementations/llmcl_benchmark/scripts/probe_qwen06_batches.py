#!/usr/bin/env python
"""Measure worst-case single-GPU train batches for the five baselines + Track1."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.continual_lora import (attach_loramoe, attach_olora, attach_seq_lora,
                                  collect_loramoe_loss, set_olora_task,
                                  set_router_token_mask as set_paper_mask,
                                  trainable_named_parameters)
from model.Ours_LoRA_MoE import (add_experts_to_all_layers, attach_lora_moe,
                                 collect_moe_losses, freeze_lora_moe_experts,
                                 freeze_lora_moe_routers,
                                 set_router_token_mask as set_track1_mask)
from utils.data.data_collator import DataCollator
from utils.data.data_utils import create_prompt_dataset
from utils.model.model_utils import create_hf_model
from utils.utils import (get_optimizer_grouped_parameters, load_hf_tokenizer,
                         set_random_seed)

TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
         "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]
METHODS = ["seqlora", "loramoe", "ewc", "gem", "olora", "track1"]
MIB = 1024 * 1024


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--model_name_or_path",
                   default="/home/work/Agent_HJ/00_models/Qwen3-0.6B")
    p.add_argument("--data_path", default="data/LLM-CL-Benchmark_5000")
    p.add_argument("--data_output_path", default="/tmp/qwen06_batch_probe")
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--out", required=True)
    p.add_argument("--max_prompt_len", type=int, default=1536)
    p.add_argument("--max_ans_len", type=int, default=512)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=32)
    p.add_argument("--loramoe_experts", type=int, default=4)
    p.add_argument("--track1_experts_per_task", type=int, default=1)
    p.add_argument("--max_batch", type=int, default=512)
    p.add_argument("--budget_mib", type=float, default=74000)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--gradient_checkpointing", action="store_true")
    return p.parse_args()


def longest_examples(args, tokenizer, tasks):
    found = {}
    cap = args.max_prompt_len + args.max_ans_len
    for task in tasks:
        train, _, _ = create_prompt_dataset(
            0, os.path.join(args.data_path, task), args.data_output_path,
            args.seed, distributed=False)
        best, best_len = None, -1
        for item in train:
            length = len(tokenizer(
                item["prompt"] + item["answer"], truncation=True,
                max_length=cap, add_special_tokens=False)["input_ids"])
            if length > best_len:
                best, best_len = item, length
        found[task] = (best, min(cap, best_len + 2), len(train))
        print(f"{task:12s} n={len(train):5d} longest={found[task][1]}", flush=True)
    return found


def build(args, device, tokenizer):
    model = create_hf_model(
        AutoModelForCausalLM, args.model_name_or_path, tokenizer,
        disable_dropout=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.to(device)
    if args.method in {"seqlora", "ewc", "gem"}:
        attach_seq_lora(model, args.rank, args.alpha, 0.0)
    elif args.method == "loramoe":
        attach_loramoe(model, args.rank, args.alpha, args.loramoe_experts, 1,
                       0.0, "full_softmax", 0.0, 0.0)
    elif args.method == "olora":
        attach_olora(model, args.rank, args.alpha, len(TASKS), 0.0)
        set_olora_task(model, len(TASKS) - 1)
    else:
        attach_lora_moe(model, r=args.rank, alpha=args.alpha, top_k=1,
                        aux_loss_coeff=0.01, z_loss_coeff=0.001,
                        routing_weight_mode="full_softmax")
        count = len(TASKS) * args.track1_experts_per_task
        add_experts_to_all_layers(model, count)
        freeze_lora_moe_experts(model, {count - 1})
        freeze_lora_moe_routers(model, True)
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        get_optimizer_grouped_parameters(model, 0.01), lr=1e-4,
        betas=(0.9, 0.95), eps=1e-6)
    model.train()
    return model, parameters, optimizer


def attempt(args, model, parameters, optimizer, collator, example, batch_size,
            device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for _ in range(args.steps):
            batch = collator([example] * batch_size)
            batch.pop("sources", None)
            batch = {k: v.to(device) for k, v in batch.items()
                     if isinstance(v, torch.Tensor)}
            optimizer.zero_grad(set_to_none=True)
            if args.method == "loramoe":
                set_paper_mask(model, batch.get("attention_mask"))
            elif args.method == "track1":
                set_track1_mask(model, batch.get("attention_mask"))
            output = model(**batch, use_cache=False)
            loss = output.loss
            if args.method == "loramoe":
                extra = collect_loramoe_loss(model)
                loss = loss if extra is None else loss + extra
            elif args.method == "track1":
                extra = collect_moe_losses(model)
                loss = loss if extra is None else loss + extra
            loss.backward()
            optimizer.step()
        peak = torch.cuda.max_memory_reserved(device) / MIB
        del output, loss, batch
        return peak
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return None
    finally:
        if args.method == "loramoe":
            set_paper_mask(model, None)
        elif args.method == "track1":
            set_track1_mask(model, None)


def search(args, model, parameters, optimizer, collator, example, device):
    ok, ok_peak, candidate = 0, None, 1
    while candidate <= args.max_batch:
        peak = attempt(args, model, parameters, optimizer, collator, example,
                       candidate, device)
        if peak is None or peak > args.budget_mib:
            reason = "OOM" if peak is None else f"{peak:.0f} MiB > budget"
            print(f"  B={candidate:<4d} {reason}; stop", flush=True)
            break
        ok, ok_peak = candidate, peak
        print(f"  B={candidate:<4d} OK peak={peak:.0f} MiB", flush=True)
        if candidate == args.max_batch:
            break
        # Fast climb, then fine-grained +2 near the memory budget.
        candidate = min(args.max_batch,
                        candidate * 2 if peak < args.budget_mib * 0.55
                        else candidate + 2)
    return ok, ok_peak, ok == args.max_batch


def main():
    args = args_parser()
    set_random_seed(args.seed)
    device = torch.device("cuda", 0)
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    tasks = [x for x in args.tasks.split(",") if x]
    examples = longest_examples(args, tokenizer, tasks)
    collator = DataCollator(tokenizer, padding="longest",
                            max_prompt_len=args.max_prompt_len,
                            max_ans_len=args.max_ans_len,
                            pad_to_multiple_of=8, inference=False)
    model, parameters, optimizer = build(args, device, tokenizer)
    results = {}
    for task in tasks:
        example, length, count = examples[task]
        print(f"[{args.method}:{task}]", flush=True)
        maximum, peak, capped = search(
            args, model, parameters, optimizer, collator, example, device)
        results[task] = {"max_batch": maximum, "peak_mib": peak,
                         "capped": capped, "max_tok_len": length,
                         "n_train": count}
    payload = {"method": args.method, "config": vars(args),
               "gpu_total_mib": torch.cuda.get_device_properties(device).total_memory / MIB,
               "results": results}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
