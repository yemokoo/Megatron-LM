#!/usr/bin/env python
"""Per-task train-batch-size survey for Track-1 Ours_LoRA_MoE.

Goal: for each TRACE task, find the LARGEST per-device train batch that does NOT
OOM, so training can use a per-task `--per_device_train_batch_size` comma-list
instead of a single conservative value. Short tasks (C-STANCE/FOMC) can afford a
much bigger batch than long ones (MeetingBank/20Minuten).

Why "longest sample x B":
  DataCollator pads a batch to the LONGEST sequence IN that batch (capped at
  max_prompt_len + max_ans_len = 2048). Training uses a RandomSampler, so some
  batch WILL eventually contain the globally-longest sample alongside B-1 others
  -> that batch is padded to the global max and is the true worst case for size
  B. We reproduce exactly that upper bound with B replicas of the longest sample.
  If B replicas of the longest fit, no real batch of size B can OOM.

Memory model matches training (model/Ours_LoRA_MoE.py):
  frozen bf16 backbone + attach_lora_moe(top_k=1) + 1 new expert (trainable) +
  router (trainable) + AdamW states, full forward(use_cache=False) + backward +
  optimizer.step(). Single GPU == per-GPU footprint under DDP (each rank holds its
  own full replica; DDP's extra gradient-bucket buffers scale with the tiny
  trainable set, so single-proc is a faithful proxy for a dedicated 80GB GPU).

Default: gradient checkpointing OFF (faster steps). Any task whose OFF max batch
is <= --ckpt_threshold is ALSO probed with checkpointing ON (long tasks), since
checkpointing trades recompute for a much larger feasible batch.

Run on an EMPTY, dedicated GPU:
  CUDA_VISIBLE_DEVICES=0 python scripts/batch_survey.py \
      --model_name_or_path /home/work/Agent_HJ/00_models/Qwen3-8B \
      --data_path data/LLM-CL-Benchmark_5000 \
      --out eval_out/batch_survey_8b.json
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import DataCollator
from utils.utils import load_hf_tokenizer, get_optimizer_grouped_parameters
from utils.model.model_utils import create_hf_model
from model.Ours_LoRA_MoE import (
    attach_lora_moe, add_experts_to_all_layers,
    freeze_lora_moe_experts, freeze_lora_moe_routers, collect_moe_losses,
    set_router_token_mask)

ALL_TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
             "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]

MiB = 1024 * 1024


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--data_output_path", default="/tmp/data_files_survey/")
    p.add_argument("--tasks", default=",".join(ALL_TASKS),
                   help="Comma-separated subset, default all 8.")
    p.add_argument("--out", required=True, help="Where to write the survey JSON.")
    p.add_argument("--max_prompt_len", type=int, default=1536)
    p.add_argument("--max_ans_len", type=int, default=512)
    p.add_argument("--lora_moe_rank", type=int, default=8)
    p.add_argument("--lora_moe_alpha", type=int, default=32)
    p.add_argument("--top_k", type=int, default=1)
    p.add_argument("--max_batch", type=int, default=256,
                   help="Search cap. A task that fits this is reported as '>=cap'.")
    p.add_argument("--budget_mib", type=float, default=72000,
                   help="Per-GPU memory budget. A batch whose measured peak exceeds "
                        "this is rejected. Set well below the physical 81151 MiB: under "
                        "DDP each rank sees only ~79.25 GiB (NCCL/CUDA context) and adds "
                        "gradient-bucket/comm buffers the single-proc probe omits, so a "
                        "~7 GiB margin keeps surveyed batches OOM-safe in real training.")
    p.add_argument("--num_experts", type=int, default=8,
                   help="Number of experts present during the probe. Memory grows with "
                        "expert count (see build_model); default 8 = the final round, so "
                        "the surveyed batch is safe at EVERY round (conservative).")
    p.add_argument("--ckpt_threshold", type=int, default=3,
                   help="If a task's checkpointing-OFF max batch is <= this, also "
                        "probe it with checkpointing ON.")
    p.add_argument("--steps_per_probe", type=int, default=3,
                   help="fwd+bwd+step iterations per candidate B (peak stabilizes "
                        "after the first optimizer step allocates AdamW states).")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def build_model(args, device, grad_ckpt):
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    model = create_hf_model(AutoModelForCausalLM, args.model_name_or_path,
                            tokenizer, disable_dropout=True,
                            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model = model.to(device=device)
    attach_lora_moe(model, r=args.lora_moe_rank, alpha=args.lora_moe_alpha,
                    top_k=args.top_k, aux_loss_coeff=0.01, z_loss_coeff=0.001,
                    routing_weight_mode="full_softmax")
    # Grow to num_experts experts. CRITICAL for a faithful memory measurement: the
    # LoRA-MoE forward computes a gate/up/down delta for EVERY expert that any token
    # routes to (per-token top-k over a random router -> with a long sample's many
    # tokens, effectively ALL experts fire), and with checkpointing OFF those
    # per-expert [batch, seq, intermediate] activations are all saved for backward.
    # So peak memory grows with the expert count -- a 1-expert probe underestimates
    # every later round. Survey at the worst-case count (the final round's 8) to get
    # batch sizes safe for the whole run. Only the newest expert + router are trained,
    # matching phase-1 (trainable set is tiny; the cost is the forward over all experts).
    ne = args.num_experts
    add_experts_to_all_layers(model, ne)
    freeze_lora_moe_experts(model, trainable_expert_indices={ne - 1})
    freeze_lora_moe_routers(model, trainable=True)
    if grad_ckpt:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()
    optimizer = torch.optim.AdamW(
        get_optimizer_grouped_parameters(model, 0.0),
        lr=1e-4, betas=(0.9, 0.95))
    model.train()
    return model, tokenizer, optimizer


def task_longest_instance(args, task):
    """Return (longest_instance, max_tok_len, n_samples) for a task's train set.
    Length = tokenized(prompt+answer) capped at max_prompt_len+max_ans_len."""
    tok = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    path = os.path.join(args.data_path, task)
    train_dataset, _, _ = create_prompt_dataset(
        0, path, args.data_output_path, args.seed, distributed=False)
    cap = args.max_prompt_len + args.max_ans_len
    best, best_len = None, -1
    for i in range(len(train_dataset)):
        ex = train_dataset[i]
        n = len(tok(ex["prompt"] + ex["answer"], truncation=True,
                    max_length=cap, add_special_tokens=False)["input_ids"])
        if n > best_len:
            best_len, best = n, ex
    # +2 for bos/eos the collator appends (capped at cap).
    return best, min(cap, best_len + 2), len(train_dataset)


def try_batch(model, optimizer, collator, longest, B, device, steps):
    """Run `steps` fwd+bwd+step on a batch of B replicas of `longest`.
    Return peak reserved MiB on success, or None on CUDA OOM."""
    batch_instances = [longest] * B
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for _ in range(steps):
            model_inputs = collator(batch_instances)
            model_inputs.pop("sources", None)
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()
                            if isinstance(v, torch.Tensor)}
            optimizer.zero_grad(set_to_none=True)
            set_router_token_mask(model, model_inputs.get("attention_mask"))
            try:
                outputs = model(**model_inputs, use_cache=False)
                moe_loss = collect_moe_losses(model)
                loss = outputs.loss if moe_loss is None else outputs.loss + moe_loss
                loss.backward()
            finally:
                set_router_token_mask(model, None)
            optimizer.step()
        peak = torch.cuda.max_memory_reserved(device) / MiB
        del outputs, loss, model_inputs
        return peak
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        # Recover: drop refs + free cache. OOM caught before any state corruption.
        optimizer.zero_grad(set_to_none=True)
        for v in ("outputs", "loss", "model_inputs"):
            if v in locals():
                del locals()[v]
        torch.cuda.empty_cache()
        return None


def search_max_batch(model, optimizer, collator, longest, device, steps, cap, budget_mib):
    """Largest B whose peak stays within budget, found by a MONOTONE-INCREASING
    climb that stops at the first OOM / budget-exceed. Returns (max_ok, peak, capped).

    Why not exponential-bracket + binary-search: throwing a way-too-big B (e.g. 2x
    over) and catching its OOM leaves the caching allocator in a state where the
    next (smaller, physically-fitting) B *also* OOMs -- so the binary search that
    descends AFTER a big OOM systematically underestimates (C-STANCE fit B=16 at
    52GB, yet B=17 "OOM"d). expandable_segments did not fix it. The robust cure is
    to never test a smaller B after a larger one: climb up only, and the single
    OOM we ever hit is terminal, so it can't pollute a later measurement.

    To stay fast we double B while far below budget, then step by +2 once within
    half the budget (memory is linear in B, so the last double lands < budget and
    the fine steps pin the limit to +/-2). `budget_mib` carries a safety margin
    below the physical 81GB for DDP/NCCL overhead the single-proc probe omits."""
    ok, peak_ok = 0, None
    B = 1
    while B <= cap:
        peak = try_batch(model, optimizer, collator, longest, B, device, steps)
        if peak is None:
            print(f"    B={B:<4d} OOM (stop)", flush=True)
            break
        if peak > budget_mib:
            print(f"    B={B:<4d} peak={peak:8.0f} MiB > budget {budget_mib:.0f} (stop)", flush=True)
            break
        ok, peak_ok = B, peak
        print(f"    B={B:<4d} OK   peak={peak:8.0f} MiB", flush=True)
        if B >= cap:
            break
        # Double while comfortably below budget; fine +2 steps once past halfway.
        nxt = B + 2 if peak > 0.5 * budget_mib else B * 2
        B = min(nxt, cap)
    capped = ok >= cap
    return ok, peak_ok, capped


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda", 0)
    tasks = [t for t in args.tasks.split(",") if t]

    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    collator = DataCollator(tokenizer, padding="longest",
                            max_prompt_len=args.max_prompt_len,
                            max_ans_len=args.max_ans_len,
                            pad_to_multiple_of=8, inference=False)

    # Precompute each task's worst-case (longest) sample once.
    print("== tokenizing task train sets for longest-sample lengths ==", flush=True)
    longest = {}
    for t in tasks:
        inst, ln, n = task_longest_instance(args, t)
        longest[t] = (inst, ln, n)
        print(f"  {t:12s} n={n:5d}  max_tok_len={ln}", flush=True)

    results = {}
    total_mib = torch.cuda.get_device_properties(device).total_memory / MiB
    print(f"\nGPU total = {total_mib:.0f} MiB\n", flush=True)

    # --- Pass 1: checkpointing OFF for every task ---
    print("=========== PASS 1: gradient checkpointing OFF ===========", flush=True)
    model, _, optimizer = build_model(args, device, grad_ckpt=False)
    for t in tasks:
        inst, ln, n = longest[t]
        print(f"[{t}] OFF  (longest={ln} tok)", flush=True)
        mx, peak, capped = search_max_batch(model, optimizer, collator, inst, device,
                                            args.steps_per_probe, args.max_batch, args.budget_mib)
        results[t] = {"max_tok_len": ln, "n_train": n,
                      "off": {"max_batch": mx, "peak_mib": peak, "capped": capped}}
        print(f"  -> OFF max_batch = {mx}{'+ (capped)' if capped else ''}"
              f"  peak={peak:.0f} MiB\n", flush=True)
    del model, optimizer
    torch.cuda.empty_cache()

    # --- Pass 2: checkpointing ON for tasks whose OFF batch was tiny ---
    on_tasks = [t for t in tasks
                if not results[t]["off"]["capped"]
                and results[t]["off"]["max_batch"] <= args.ckpt_threshold]
    if on_tasks:
        print(f"=========== PASS 2: checkpointing ON for {on_tasks} ===========", flush=True)
        model, _, optimizer = build_model(args, device, grad_ckpt=True)
        for t in on_tasks:
            inst, ln, n = longest[t]
            print(f"[{t}] ON  (longest={ln} tok)", flush=True)
            mx, peak, capped = search_max_batch(model, optimizer, collator, inst, device,
                                                args.steps_per_probe, args.max_batch, args.budget_mib)
            results[t]["on"] = {"max_batch": mx, "peak_mib": peak, "capped": capped}
            print(f"  -> ON max_batch = {mx}{'+ (capped)' if capped else ''}"
                  f"  peak={peak:.0f} MiB\n", flush=True)
        del model, optimizer
        torch.cuda.empty_cache()

    # --- Recommendation per task: prefer OFF; use ON if it meaningfully helps ---
    for t in tasks:
        off = results[t]["off"]
        on = results[t].get("on")
        if on and on["max_batch"] > off["max_batch"]:
            results[t]["recommend"] = {"grad_ckpt": True, "batch": on["max_batch"]}
        else:
            results[t]["recommend"] = {"grad_ckpt": False, "batch": off["max_batch"]}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"gpu_total_mib": total_mib, "config": vars(args),
                   "results": results}, f, indent=2)

    # --- Pretty summary + ready-to-paste comma list (in ALL_TASKS order) ---
    print("\n================= BATCH SURVEY SUMMARY =================", flush=True)
    print(f"{'task':12s} {'maxlen':>6s} {'OFF':>10s} {'ON':>10s}  recommend")
    for t in tasks:
        r = results[t]
        off = r["off"]
        offs = f"{off['max_batch']}{'+' if off['capped'] else ''}"
        on = r.get("on")
        ons = f"{on['max_batch']}{'+' if on['capped'] else ''}" if on else "-"
        rec = r["recommend"]
        print(f"{t:12s} {r['max_tok_len']:6d} {offs:>10s} {ons:>10s}  "
              f"batch={rec['batch']} ckpt={'ON' if rec['grad_ckpt'] else 'off'}")
    order = [t for t in ALL_TASKS if t in results]
    comma = ",".join(str(results[t]["recommend"]["batch"]) for t in order)
    print(f"\n--per_device_train_batch_size {comma}"
          f"   (order: {','.join(order)})")
    print(f"\nsaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
