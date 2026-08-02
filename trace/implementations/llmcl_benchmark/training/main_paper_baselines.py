#!/usr/bin/env python
"""Train the five non-MH-MoE baselines from arXiv:2602.12587 on TRACE."""
import argparse
import json
import os
import sys

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, SchedulerType

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.continual_lora import (
    attach_loramoe, attach_olora, attach_seq_lora, parameter_report)
from model.paper_baselines import EWCLoRA, GEMLoRA, LoRAMoE, OLoRA, SeqLoRA
from utils.data.data_collator import DataCollator
from utils.data.data_utils import create_prompt_dataset
from utils.model.model_utils import create_hf_model
from utils.utils import load_hf_tokenizer, print_rank_0, set_random_seed


TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
         "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]


def csv_strings(value):
    return value.split(",")


def csv_ints(value):
    return [int(item) for item in value.split(",")]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=["seqlora", "loramoe", "ewc", "gem", "olora"])
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--dataset_name", type=csv_strings, default=["all"])
    parser.add_argument("--data_output_path", default="/tmp/paper_baseline_data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_train_epochs", type=csv_ints, default=[2])
    parser.add_argument("--per_device_train_batch_size", type=csv_ints, default=[3])
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_prompt_len", type=int, default=1536)
    parser.add_argument("--max_ans_len", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--lr_scheduler_type", type=SchedulerType,
                        default="constant_with_warmup")
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--start_task", type=int, default=0)
    parser.add_argument("--resume_checkpoint", default="",
                        help="Completed task checkpoint to resume after. The matching "
                             "output_dir/result.json restores prior metrics.")
    parser.add_argument("--loss_log_interval", type=int, default=10)
    parser.add_argument("--max_train_steps_per_task", type=int, default=0,
                        help="Smoke-test cap; zero uses the complete task loader.")
    parser.add_argument("--gradient_checkpointing_tasks", default="")
    parser.add_argument("--disable_dropout", action="store_true")

    # Shared FFN-LoRA parameterization. Keep rank/alpha equal to MH-MoE.
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    # LoRAMoE: official residual-mixture equation + target-paper sparse top-k.
    parser.add_argument("--loramoe_num_experts", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--routing_weight_mode", default="full_softmax",
                        choices=["full_softmax", "topk_softmax"])
    # Official LoRAMoE uses the routed residual directly. Its optional BLC loss
    # defaults to zero; the target paper does not report an added aux/z loss.
    parser.add_argument("--moe_aux_loss_coeff", type=float, default=0.0)
    parser.add_argument("--moe_z_loss_coeff", type=float, default=0.0)

    # TRACE EWC accumulates squared training gradients online.
    parser.add_argument("--ewc_lambda", type=float, default=400.0)

    # Canonical GEM recomputes past-task gradients from episodic examples at the
    # current parameters. Memory size is the number of examples retained per task.
    parser.add_argument("--gem_memory_size", type=int, default=32,
                        help="Episodic examples retained per completed task.")
    parser.add_argument("--gem_memory_batch_size", type=int, default=4,
                        help="Global examples sampled per past task and GEM step; "
                             "samples are sharded across distributed ranks.")
    parser.add_argument("--gem_margin", type=float, default=0.0)
    parser.add_argument("--gem_qp_eps", type=float, default=1e-3)

    # Official O-LoRA defaults (|A_old A_new^T| and optional new-adapter L2).
    parser.add_argument("--olora_lambda_orthogonal", type=float, default=0.5)
    parser.add_argument("--olora_lambda_l2", type=float, default=0.0)
    return parser.parse_args()


def expand_per_task(values, tasks, name):
    if len(values) == 1:
        return values * len(tasks)
    if len(values) != len(tasks):
        raise ValueError(f"{name} has {len(values)} values for {len(tasks)} tasks")
    return values


def main():
    args = parse_args()
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group("nccl", device_id=device)
    args.global_rank = dist.get_rank() if dist.is_initialized() else 0
    set_random_seed(args.seed)
    if dist.is_initialized():
        dist.barrier()

    tasks = TASKS if args.dataset_name[0] == "all" else args.dataset_name
    args.num_train_epochs = expand_per_task(
        args.num_train_epochs, tasks, "num_train_epochs")
    batches = expand_per_task(
        args.per_device_train_batch_size, tasks, "per_device_train_batch_size")
    args.batch_by_task = dict(zip(tasks, batches))
    args.ckpt_tasks = {item for item in
                       args.gradient_checkpointing_tasks.split(",") if item}
    unknown = args.ckpt_tasks - set(tasks)
    if unknown:
        raise ValueError(f"unknown checkpointing tasks: {sorted(unknown)}")

    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    model = create_hf_model(
        AutoModelForCausalLM, args.model_name_or_path, tokenizer,
        disable_dropout=args.disable_dropout, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True)
    model.to(device)

    if args.method in {"seqlora", "ewc", "gem"}:
        attach_seq_lora(model, args.lora_rank, args.lora_alpha,
                        args.lora_dropout)
    elif args.method == "loramoe":
        attach_loramoe(
            model, args.lora_rank, args.lora_alpha,
            args.loramoe_num_experts, args.top_k, args.lora_dropout,
            args.routing_weight_mode, args.moe_aux_loss_coeff,
            args.moe_z_loss_coeff)
    else:
        attach_olora(model, args.lora_rank, args.lora_alpha, len(tasks),
                     args.lora_dropout)

    if args.resume_checkpoint:
        meta_path = os.path.join(args.resume_checkpoint, "paper_baseline_meta.json")
        state_path = os.path.join(args.resume_checkpoint, "pytorch_model.bin")
        with open(meta_path) as handle:
            resume_meta = json.load(handle)
        if resume_meta.get("method") != args.method:
            raise ValueError(
                f"resume method={resume_meta.get('method')} != requested {args.method}")
        state = torch.load(state_path, map_location="cpu")
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"unexpected resume keys: {incompatible.unexpected_keys[:8]}")
        checkpoint_task = int(os.path.basename(os.path.normpath(args.resume_checkpoint)))
        args.start_task = checkpoint_task + 1
        print_rank_0(
            f"Resuming {args.method} after task {checkpoint_task} from "
            f"{args.resume_checkpoint}", args.global_rank)

    train_loaders, eval_loaders, test_loaders = {}, {}, {}
    for task in tasks:
        train_data, eval_data, test_data = create_prompt_dataset(
            args.local_rank, os.path.join(args.data_path, task),
            args.data_output_path, args.seed)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
            eval_sampler = SequentialSampler(eval_data)
            test_sampler = SequentialSampler(test_data)
        else:
            train_sampler = DistributedSampler(train_data, shuffle=True, seed=args.seed)
            eval_sampler = DistributedSampler(eval_data, shuffle=False)
            test_sampler = DistributedSampler(test_data, shuffle=False)
        collator = DataCollator(
            tokenizer, padding="longest", max_prompt_len=args.max_prompt_len,
            max_ans_len=args.max_ans_len, pad_to_multiple_of=8, inference=False)
        inference_collator = DataCollator(
            tokenizer, model=model, padding="longest",
            max_prompt_len=args.max_prompt_len, max_ans_len=args.max_ans_len,
            pad_to_multiple_of=8, inference=True)
        train_loaders[task] = DataLoader(
            train_data, sampler=train_sampler, collate_fn=collator,
            batch_size=args.batch_by_task[task], num_workers=4, pin_memory=True)
        eval_loaders[task] = DataLoader(
            eval_data, sampler=eval_sampler, collate_fn=collator,
            batch_size=args.per_device_eval_batch_size)
        test_loaders[task] = DataLoader(
            test_data, sampler=test_sampler, collate_fn=inference_collator,
            batch_size=args.per_device_eval_batch_size)

    trainer_class = {
        "seqlora": SeqLoRA,
        "loramoe": LoRAMoE,
        "ewc": EWCLoRA,
        "gem": GEMLoRA,
        "olora": OLoRA,
    }[args.method]
    print_rank_0(
        f"paper baseline={args.method} epochs={args.num_train_epochs} "
        f"rank={args.lora_rank} alpha={args.lora_alpha} "
        f"parameters={parameter_report(model, args.method)}",
        args.global_rank)
    trainer = trainer_class(
        model, tokenizer, None, train_loaders, eval_loaders, test_loaders, args)
    if args.resume_checkpoint:
        trainer.load_resume_state(args.resume_checkpoint)
    try:
        trainer.train_continual()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
