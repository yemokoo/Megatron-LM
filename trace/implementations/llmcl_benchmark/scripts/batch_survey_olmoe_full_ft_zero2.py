#!/usr/bin/env python
"""One ZeRO-2 OLMoE full-finetune micro-batch probe.

This program intentionally tests exactly one TRACE task and one per-device
micro-batch size.  The shell orchestrator starts a fresh torchrun for every
candidate, because a CUDA OOM on one rank can leave NCCL/ZeRO collectives in an
unusable state for subsequent candidates.

The batch is made from copies of the longest tokenized training example in the
task.  Since the real collator pads to the longest sequence in a batch, this is
the conservative activation-memory case for that task.  Two or more optimizer
steps are required so AdamW's lazily-created state is included in the peak.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import deepspeed
from transformers import AutoModelForCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from scripts.batch_survey import task_longest_instance
from utils.data.data_collator import DataCollator
from utils.model.model_utils import create_hf_model, resolve_attention_implementation
from utils.utils import get_optimizer_grouped_parameters, load_hf_tokenizer, set_random_seed

MiB = 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single OLMoE full-FT ZeRO-2 micro-batch probe")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--data_output_path", default="/tmp/data_files_olmoe_full_ft_survey/")
    parser.add_argument("--task", required=True)
    parser.add_argument("--micro_batch", type=int, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max_prompt_len", type=int, default=1536)
    parser.add_argument("--max_ans_len", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--attn_implementation", default="auto",
                        choices=["auto", "flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--result", required=True)
    parser.add_argument("--offload_optimizer", action="store_true",
                        help="Survey fallback only: move Adam states to CPU.")
    args = parser.parse_args()
    if args.micro_batch < 1 or args.steps < 2:
        parser.error("--micro_batch must be >=1 and --steps must be >=2")
    return args


def zero2_config(args, world_size):
    optimizer_device = "cpu" if args.offload_optimizer else "none"
    return {
        "train_micro_batch_size_per_gpu": args.micro_batch,
        "gradient_accumulation_steps": 1,
        "train_batch_size": args.micro_batch * world_size,
        "steps_per_print": 1_000_000,
        "gradient_clipping": 1.0,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": optimizer_device,
                "pin_memory": bool(args.offload_optimizer),
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_scatter": True,
        },
        "wall_clock_breakdown": False,
    }


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed(dist_backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    set_random_seed(args.seed)

    # Only rank 0 scans/tokenizes the dataset. All ranks receive the exact same
    # raw example, avoiding cache races and making the memory probe identical.
    payload = [None]
    if rank == 0:
        longest, token_length, sample_count = task_longest_instance(args, args.task)
        payload[0] = (longest, token_length, sample_count)
    dist.broadcast_object_list(payload, src=0)
    longest, token_length, sample_count = payload[0]

    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    attention_backend = resolve_attention_implementation(args.attn_implementation)
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
    # Full fine-tuning: do not freeze experts, gate, attention, embeddings, or head.
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

    optimizer = torch.optim.AdamW(
        get_optimizer_grouped_parameters(model, args.weight_decay),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
    )
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=zero2_config(args, world_size),
        dist_init_required=False,
    )

    collator = DataCollator(
        tokenizer,
        padding="longest",
        max_prompt_len=args.max_prompt_len,
        max_ans_len=args.max_ans_len,
        pad_to_multiple_of=8,
        inference=False,
    )
    instances = [longest] * args.micro_batch
    dist.barrier()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()

    last_loss = None
    for _ in range(args.steps):
        batch = collator(instances)
        batch.pop("sources", None)
        batch = {
            key: value.to(device)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        outputs = engine(**batch, use_cache=False)
        loss = outputs.loss
        engine.backward(loss)
        engine.step()
        last_loss = loss.detach().float()

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak_allocated = torch.tensor(
        torch.cuda.max_memory_allocated(device) / MiB,
        device=device, dtype=torch.float64)
    peak_reserved = torch.tensor(
        torch.cuda.max_memory_reserved(device) / MiB,
        device=device, dtype=torch.float64)
    dist.all_reduce(peak_allocated, op=dist.ReduceOp.MAX)
    dist.all_reduce(peak_reserved, op=dist.ReduceOp.MAX)
    elapsed_tensor = torch.tensor(elapsed, device=device, dtype=torch.float64)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)

    if rank == 0:
        result = {
            "status": "ok",
            "task": args.task,
            "micro_batch_per_gpu": args.micro_batch,
            "global_batch": args.micro_batch * world_size,
            "world_size": world_size,
            "steps": args.steps,
            "max_token_length": token_length,
            "train_sample_count": sample_count,
            "gradient_checkpointing": True,
            "zero_stage": 2,
            "optimizer_offload": args.offload_optimizer,
            "attention_backend": getattr(model.config, "_attn_implementation", attention_backend),
            "peak_allocated_mib_max_rank": peak_allocated.item(),
            "peak_reserved_mib_max_rank": peak_reserved.item(),
            "seconds_per_step_max_rank": elapsed_tensor.item() / args.steps,
            "last_loss": last_loss.item(),
            "torch_version": torch.__version__,
            "deepspeed_version": deepspeed.__version__,
        }
        result_path = Path(args.result)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        print("SURVEY_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
