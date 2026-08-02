#!/usr/bin/env python
"""Track-2 OLMoE batch-size survey.

This reuses the longest-sample search and monotone batch search from
scripts/batch_survey.py, but replaces its Track-1 LoRA model probe with the
actual growing OLMoE phase-1 path.  The probe grows all 16 MoE layers to the
requested final expert count, trains only the newest full expert and router
row, and forces that expert into top-k.  Forced routing is the conservative
memory case and is therefore also safe for the natural-router ablation.

Run this only on an empty, dedicated GPU.  It performs short optimizer probes,
not continual training, and writes a JSON report selected by --out.
"""
import json
import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import batch_survey as survey
from model.Ours_MoE_FFN import (
    add_experts_to_all_layers,
    attach_growing_moe,
    collect_moe_losses,
    freeze_moe_experts,
    freeze_moe_routers,
    set_phase1_forced_experts,
    set_router_token_mask,
)
from utils.data.data_collator import DataCollator
from utils.model.model_utils import create_hf_model, resolve_attention_implementation
from utils.utils import get_optimizer_grouped_parameters, load_hf_tokenizer


def build_growing_base(args, device, grad_ckpt):
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    attention_backend = resolve_attention_implementation("auto")
    model = create_hf_model(
        AutoModelForCausalLM,
        args.model_name_or_path,
        tokenizer,
        disable_dropout=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=attention_backend,
        forbid_vocab_growth=True,
    )
    model = model.to(device=device)
    attach_growing_moe(model, aux_loss_coeff=0.01, z_loss_coeff=0.001)

    if grad_ckpt:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()
    model.train()
    return model, tokenizer, attention_backend


def configure_phase1(model, num_new_experts):
    newest = {num_new_experts - 1}
    freeze_moe_experts(model, trainable_new_indices=newest)
    freeze_moe_routers(
        model, trainable_new_indices=newest, trainable_original=False)
    set_phase1_forced_experts(model, newest)

    return torch.optim.AdamW(
        get_optimizer_grouped_parameters(model, 0.0),
        lr=1e-4,
        betas=(0.9, 0.95),
    )


def build_model(args, device, grad_ckpt):
    model, tokenizer, _ = build_growing_base(args, device, grad_ckpt)
    if args.num_experts < 1:
        raise ValueError("--num_experts must be at least 1.")
    add_experts_to_all_layers(model, args.num_experts)
    optimizer = configure_phase1(model, args.num_experts)
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(
            "Track-2 probe: attention_backend=%s new_experts_per_layer=%d "
            "phase1_routing=force grad_ckpt=%s"
            % (model.config._attn_implementation, args.num_experts, grad_ckpt),
            flush=True,
        )
    return model, tokenizer, optimizer


def try_batch(model, optimizer, collator, longest, batch_size, device, steps):
    batch_instances = [longest] * batch_size
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for _ in range(steps):
            model_inputs = collator(batch_instances)
            model_inputs.pop("sources", None)
            model_inputs = {
                key: value.to(device)
                for key, value in model_inputs.items()
                if isinstance(value, torch.Tensor)
            }
            optimizer.zero_grad(set_to_none=True)
            set_router_token_mask(model, model_inputs.get("attention_mask"))
            try:
                outputs = model(**model_inputs, use_cache=False)
                moe_loss = collect_moe_losses(model)
                loss = outputs.loss if moe_loss is None else outputs.loss + moe_loss
                loss.backward()
            finally:
                set_router_token_mask(model, None)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters()
                 if parameter.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
        peak = torch.cuda.max_memory_reserved(device) / survey.MiB
        del outputs, loss, model_inputs
        return peak
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        optimizer.zero_grad(set_to_none=True)
        set_router_token_mask(model, None)
        torch.cuda.empty_cache()
        return None


def distributed_search(model, optimizer, collator, longest, device, steps,
                       cap, budget_mib):
    """Run the monotone batch search on every DDP rank.

    The returned peak is the maximum reserved memory over all ranks. Batch size
    is per device; the corresponding global batch is batch_size * world_size.
    """
    rank = dist.get_rank()
    ok, peak_ok = 0, None
    batch_size = 1
    while batch_size <= cap:
        local_peak = try_batch(
            model, optimizer, collator, longest, batch_size, device, steps)
        success = torch.tensor(
            0 if local_peak is None else 1, device=device, dtype=torch.int32)
        dist.all_reduce(success, op=dist.ReduceOp.MIN)
        if not success.item():
            if rank == 0:
                print("    per_device_B=%-4d OOM on at least one rank (stop)"
                      % batch_size, flush=True)
            break

        peak = torch.tensor(local_peak, device=device, dtype=torch.float64)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        max_peak = peak.item()
        if max_peak > budget_mib:
            if rank == 0:
                print(
                    "    per_device_B=%-4d max_rank_peak=%8.0f MiB "
                    "> budget %.0f (stop)"
                    % (batch_size, max_peak, budget_mib),
                    flush=True,
                )
            break

        ok, peak_ok = batch_size, max_peak
        if rank == 0:
            print(
                "    per_device_B=%-4d global_B=%-4d OK "
                "max_rank_peak=%8.0f MiB"
                % (batch_size, batch_size * dist.get_world_size(), max_peak),
                flush=True,
            )
        if batch_size >= cap:
            break
        next_batch = (
            batch_size + 2 if max_peak > 0.5 * budget_mib
            else batch_size * 2
        )
        batch_size = min(next_batch, cap)
    return ok, peak_ok, ok >= cap


def wrap_ddp(raw_model, local_rank):
    return DDP(
        raw_model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )


def distributed_main():
    args = survey.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed)

    if world_size != 4 and rank == 0:
        print(
            "WARNING: requested Track-2 setting uses 4 GPUs, but torchrun "
            "started world_size=%d." % world_size,
            flush=True,
        )

    tasks = [task for task in args.tasks.split(",") if task]
    if len(tasks) > args.num_experts:
        raise ValueError(
            "--num_experts is the final appended-expert count and must be at "
            "least the number of surveyed tasks.")
    tokenizer = load_hf_tokenizer(
        args.model_name_or_path, fast_tokenizer=True)
    collator = DataCollator(
        tokenizer,
        padding="longest",
        max_prompt_len=args.max_prompt_len,
        max_ans_len=args.max_ans_len,
        pad_to_multiple_of=8,
        inference=False,
    )

    # Only rank 0 scans the datasets/cache; the selected raw example and its
    # metadata are then broadcast to every rank for identical DDP probes.
    payload = [None]
    if rank == 0:
        print(
            "== tokenizing task train sets for longest-sample lengths ==",
            flush=True,
        )
        longest = {}
        for task in tasks:
            instance, length, sample_count = survey.task_longest_instance(
                args, task)
            longest[task] = (instance, length, sample_count)
            print(
                "  %-12s n=%5d max_tok_len=%d"
                % (task, sample_count, length),
                flush=True,
            )
        payload[0] = longest
    dist.broadcast_object_list(payload, src=0)
    longest = payload[0]

    total_mib = (
        torch.cuda.get_device_properties(device).total_memory / survey.MiB)
    if rank == 0:
        print(
            "\nDDP world_size=%d, per-rank GPU total=%.0f MiB\n"
            % (world_size, total_mib),
            flush=True,
        )

    results = {}
    if rank == 0:
        print(
            "=========== PASS 1: gradient checkpointing OFF ===========",
            flush=True,
        )
    raw_model, _, _ = build_growing_base(args, device, grad_ckpt=False)
    for task_index, task in enumerate(tasks):
        num_new_experts = task_index + 1
        add_experts_to_all_layers(raw_model, 1)
        optimizer = configure_phase1(raw_model, num_new_experts)
        model = wrap_ddp(raw_model, local_rank)
        instance, length, sample_count = longest[task]
        if rank == 0:
            print(
                "[%s] OFF (longest=%d tok, experts=64+%d)"
                % (task, length, num_new_experts),
                flush=True,
            )
        maximum, peak, capped = distributed_search(
            model, optimizer, collator, instance, device,
            args.steps_per_probe, args.max_batch, args.budget_mib)
        results[task] = {
            "max_tok_len": length,
            "n_train": sample_count,
            "num_new_experts": num_new_experts,
            "off": {
                "max_batch": maximum,
                "global_batch": maximum * world_size,
                "peak_mib": peak,
                "capped": capped,
            },
        }
        for parameter in raw_model.parameters():
            parameter.grad = None
        del model, optimizer
        torch.cuda.empty_cache()
        dist.barrier()
    del raw_model
    torch.cuda.empty_cache()
    dist.barrier()

    on_tasks = [
        task for task in tasks
        if not results[task]["off"]["capped"]
        and results[task]["off"]["max_batch"] <= args.ckpt_threshold
    ]
    if on_tasks:
        if rank == 0:
            print(
                "=========== PASS 2: gradient checkpointing ON for %s "
                "===========" % on_tasks,
                flush=True,
            )
        raw_model, _, _ = build_growing_base(args, device, grad_ckpt=True)
        for task_index, task in enumerate(tasks):
            num_new_experts = task_index + 1
            add_experts_to_all_layers(raw_model, 1)
            if task not in on_tasks:
                continue
            optimizer = configure_phase1(raw_model, num_new_experts)
            model = wrap_ddp(raw_model, local_rank)
            instance, length, _ = longest[task]
            if rank == 0:
                print(
                    "[%s] ON (longest=%d tok, experts=64+%d)"
                    % (task, length, num_new_experts),
                    flush=True,
                )
            maximum, peak, capped = distributed_search(
                model, optimizer, collator, instance, device,
                args.steps_per_probe, args.max_batch, args.budget_mib)
            results[task]["on"] = {
                "max_batch": maximum,
                "global_batch": maximum * world_size,
                "peak_mib": peak,
                "capped": capped,
            }
            for parameter in raw_model.parameters():
                parameter.grad = None
            del model, optimizer
            torch.cuda.empty_cache()
            dist.barrier()
        del raw_model
        torch.cuda.empty_cache()
        dist.barrier()

    for task in tasks:
        off = results[task]["off"]
        on = results[task].get("on")
        selected = on if on and on["max_batch"] > off["max_batch"] else off
        results[task]["recommend"] = {
            "grad_ckpt": selected is on,
            "per_device_batch": selected["max_batch"],
            "global_batch": selected["global_batch"],
        }

    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(
                {
                    "world_size": world_size,
                    "batch_semantics": "per_device",
                    "gpu_total_mib": total_mib,
                    "config": vars(args),
                    "results": results,
                },
                handle,
                indent=2,
            )

        print("\n============= 4-GPU DDP BATCH SURVEY SUMMARY =============")
        print("%-12s %6s %10s %10s  %s"
              % ("task", "maxlen", "OFF/dev", "ON/dev", "memory result"))
        for task in tasks:
            result = results[task]
            off = result["off"]
            off_text = "%d%s" % (
                off["max_batch"], "+" if off["capped"] else "")
            on = result.get("on")
            on_text = (
                "%d%s" % (on["max_batch"], "+" if on["capped"] else "")
                if on else "-"
            )
            rec = result["recommend"]
            print(
                "%-12s %6d %10s %10s  per_device=%d global=%d ckpt=%s"
                % (
                    task,
                    result["max_tok_len"],
                    off_text,
                    on_text,
                    rec["per_device_batch"],
                    rec["global_batch"],
                    "ON" if rec["grad_ckpt"] else "off",
                )
            )
        print("\nsaved -> %s" % args.out, flush=True)
        print(
            "NOTE: this recommendation is memory-capacity based. Compare "
            "checkpointing ON/OFF throughput before adopting final settings.",
            flush=True,
        )

    dist.destroy_process_group()


def main():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        distributed_main()
    else:
        survey.build_model = build_model
        survey.try_batch = try_batch
        survey.main()
        print(
            "NOTE: this was a single-GPU proxy. Use torchrun with four ranks "
            "for the actual 4-GPU DDP batch survey.",
            flush=True,
        )


if __name__ == "__main__":
    main()
