#!/usr/bin/env python
# Track-2 entry point: growing FFN experts on a native MoE backbone (OLMoE).
# Mirror of training/main_Ours_LoRA_MoE.py, but the backbone is already a sparse
# MoE (OlmoeSparseMoeBlock: 64 experts + 64-way gate, top-8). We wrap each block
# with GrowingOlmoeMoE and append full OlmoeMLP experts per task, growing the
# router 64 -> 64+n (see model/Ours_MoE_FFN.py). No LoRA rank/alpha/top_k args --
# top_k comes from the OLMoE config; only the added-expert count and router-loss
# coefficients are ours to set.
#
# Distributed via plain torchrun + DistributedDataParallel, NOT DeepSpeed -- see
# main_Ours_LoRA_MoE.py (repeated deepspeed.initialize() on the same model leaks
# GPU memory without bound).
import sys
sys.dont_write_bytecode = True

import argparse
import json
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, SchedulerType

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import DataCollator
from utils.utils import print_rank_0, set_random_seed, load_hf_tokenizer
from utils.model.model_utils import create_hf_model, resolve_attention_implementation
from model.Ours_MoE_FFN import Ours_MoE_FFN, attach_growing_moe, load_moe_ffn_checkpoint

AllDatasetName = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
                  "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]


class OlmoePretrainReplayDataset(Dataset):
    """Raw-text OLMoE replay, exposed as causal-LM prompt/answer examples."""

    def __init__(self, path):
        self.path = path
        self.records = []
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record.get("text")
                if not isinstance(text, str) or not text:
                    raise ValueError(f"{path}:{line_number} has no non-empty text field.")
                self.records.append({"text": text, "domain": record.get("domain")})
        if not self.records:
            raise ValueError(f"No OLMoE pretraining records found in {path}.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        # Empty prompt means the complete pretraining text receives causal-LM loss.
        return {"prompt": "", "answer": self.records[index]["text"]}


def list_of_strings(arg):
    return arg.split(',')


def list_of_ints(arg):
    return [int(value) for value in arg.split(',')]


def parse_args():
    parser = argparse.ArgumentParser(description="Growing FFN-expert continual learning on a MoE backbone (OLMoE)")
    parser.add_argument('--data_path', type=str, required=True,
                        help='Root dir with one subfolder per task (train/eval/test.json).')
    parser.add_argument('--dataset_name', type=list_of_strings, default='all',
                        help='Comma-separated task names, in training order. "all" = AllDatasetName order.')
    parser.add_argument('--data_output_path', type=str, default='/tmp/data_files/')
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help='A native MoE causal LM, e.g. OLMoE-1B-7B-0125.')
    parser.add_argument(
        "--per_device_train_batch_size", type=list_of_ints, default=[4],
        help="One per-device batch for every task, or a comma-separated list "
             "matching --dataset_name order.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--max_prompt_len", type=int, default=1024)
    parser.add_argument("--max_ans_len", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--num_train_epochs", type=list_of_strings, required=True,
                        help='Comma-separated epoch count per task, matching --dataset_name order.')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="constant_with_warmup")
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Completed task checkpoint to restore before continuing training.")
    parser.add_argument("--start_task", type=int, default=0,
                        help="Global zero-based TRACE task index to train next.")
    parser.add_argument("--resume_phase2_checkpoint", type=str, default=None,
                        help="Phase-1 checkpoint for start_task; skip phase 1 and run phase 2.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help='Overridden by the LOCAL_RANK env var when launched via torchrun.')
    parser.add_argument(
        '--gradient_checkpointing', action='store_true',
        help='Enable gradient checkpointing for every phase-1 task. '
             'Phase 2 always enables it.')
    parser.add_argument(
        '--gradient_checkpointing_tasks', type=str, default='',
        help='Comma-separated phase-1 tasks that use gradient checkpointing. '
             'Ignored when --gradient_checkpointing enables it for every task.')
    parser.add_argument('--disable_dropout', action='store_true')
    parser.add_argument('--print_loss', action='store_true')
    parser.add_argument('--loss_log_interval', type=int, default=10,
                        help='Read loss back to CPU every N micro-steps (avoids a per-step GPU sync).')

    # --- MoE-FFN specific ---
    parser.add_argument('--experts_per_task', type=int, default=1,
                        help='New full FFN (OlmoeMLP) experts appended at the start of each task.')
    parser.add_argument('--phase1_new_expert_routing', type=str, default='force',
                        choices=['force', 'router'],
                        help='force guarantees current new experts one top-k slot; router is an ablation.')
    parser.add_argument('--moe_aux_loss_coeff', type=float, default=0.01,
                        help='Switch-style load-balancing aux loss coefficient on the extended 64+n router.')
    parser.add_argument('--moe_z_loss_coeff', type=float, default=0.001,
                        help='Router logit z-loss coefficient.')
    parser.add_argument('--router_retune_epochs', type=int, default=1,
                        help='Phase-2 router retune epochs on replayed past-task data.')
    parser.add_argument(
        '--router_retune_batch_size', type=int, default=28,
        help='Per-device batch for every phase-2 router retune. Phase 2 always '
             'uses gradient checkpointing because OLMoE replay can reach 2048 tokens.')
    parser.add_argument('--past_task_ratio', type=float, default=1.0,
                        help='Fraction of every seen task used for router retuning (default: full data).')
    parser.add_argument('--olmoe_replay_path', type=str, required=True,
                        help='5K raw-text OLMoE replay JSONL used in every phase-2 router retune.')
    parser.add_argument('--attn_implementation', type=str, default='auto',
                        choices=['auto', 'flash_attention_2', 'sdpa', 'eager'],
                        help='OLMoE attention backend. auto uses FlashAttention-2 when installed, else SDPA.')

    return parser.parse_args()


def main():
    args = parse_args()
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))

    if not 0.0 < args.past_task_ratio <= 1.0:
        raise ValueError("--past_task_ratio must be in (0, 1].")
    if args.experts_per_task < 1:
        raise ValueError("--experts_per_task must be at least 1.")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be at least 1.")
    if args.router_retune_batch_size < 1:
        raise ValueError("--router_retune_batch_size must be at least 1.")

    if args.start_task < 0:
        raise ValueError("--start_task must be non-negative.")
    if args.resume_checkpoint and args.resume_phase2_checkpoint:
        raise ValueError("Use only one resume checkpoint option.")
    if bool(args.resume_checkpoint or args.resume_phase2_checkpoint) != (args.start_task > 0):
        raise ValueError("A resume checkpoint and positive --start_task must be provided together.")

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl")
    args.global_rank = torch.distributed.get_rank() if dist.is_initialized() else 0

    set_random_seed(args.seed)
    if dist.is_initialized():
        torch.distributed.barrier()

    router_pretrain_dataset = OlmoePretrainReplayDataset(args.olmoe_replay_path)
    if len(router_pretrain_dataset) != 5000:
        raise ValueError(
            "The upper-bound router replay requires exactly 5000 OLMoE pretraining "
            f"samples, but {args.olmoe_replay_path} contains {len(router_pretrain_dataset)}.")
    print_rank_0(
        f"Loaded {len(router_pretrain_dataset)} OLMoE pretraining replay samples "
        f"from {args.olmoe_replay_path}", args.global_rank)

    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    assert tokenizer.padding_side == 'left'
    assert tokenizer.truncation_side == "left"

    load_checkpoint = args.resume_phase2_checkpoint or args.resume_checkpoint
    if load_checkpoint:
        print_rank_0(
            f"Resuming from {load_checkpoint}; "
            f"next global task index={args.start_task}", args.global_rank)
        model, resume_meta = load_moe_ffn_checkpoint(
            load_checkpoint, tokenizer,
            base_model_name_or_path=args.model_name_or_path,
            device=device, dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation)
        expected_new = (args.start_task + int(bool(args.resume_phase2_checkpoint))) * args.experts_per_task
        if resume_meta.get("num_new_experts") != expected_new:
            raise ValueError(
                f"Resume checkpoint contains {resume_meta.get('num_new_experts')} "
                f"new experts per layer, but start_task={args.start_task} and "
                f"experts_per_task={args.experts_per_task} require {expected_new}.")
        saved_ept = resume_meta.get("experts_per_task")
        if saved_ept is not None and saved_ept != args.experts_per_task:
            raise ValueError(
                f"Resume checkpoint experts_per_task={saved_ept}, requested "
                f"{args.experts_per_task}.")
    else:
        resolved_attn = resolve_attention_implementation(args.attn_implementation)
        print_rank_0(
            f"Loading pretrained MoE directly as BF16 (attention={resolved_attn}, "
            f"requested={args.attn_implementation})", args.global_rank)
        model = create_hf_model(AutoModelForCausalLM, args.model_name_or_path, tokenizer,
                                disable_dropout=args.disable_dropout,
                                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                attn_implementation=resolved_attn,
                                forbid_vocab_growth=True)
        model = model.to(device=device)
        attach_growing_moe(model, aux_loss_coeff=args.moe_aux_loss_coeff,
                           z_loss_coeff=args.moe_z_loss_coeff)

    datasets = AllDatasetName if args.dataset_name[0] == "all" else args.dataset_name

    batches = args.per_device_train_batch_size
    if len(batches) == 1:
        args.batch_by_task = {task: batches[0] for task in datasets}
    elif len(batches) == len(datasets):
        args.batch_by_task = dict(zip(datasets, batches))
    else:
        raise ValueError(
            "--per_device_train_batch_size must contain one value or exactly "
            f"{len(datasets)} task values, but received {len(batches)}.")
    if any(batch < 1 for batch in args.batch_by_task.values()):
        raise ValueError("Every per-task train batch size must be at least 1.")
    print_rank_0(
        f"Phase-1 per-task train batch: {args.batch_by_task}",
        args.global_rank)

    if args.gradient_checkpointing:
        args.ckpt_tasks = set(datasets)
    else:
        args.ckpt_tasks = {
            task for task in args.gradient_checkpointing_tasks.split(',') if task}
    unknown_ckpt_tasks = args.ckpt_tasks - set(datasets)
    if unknown_ckpt_tasks:
        raise ValueError(
            "--gradient_checkpointing_tasks contains unknown tasks: "
            f"{sorted(unknown_ckpt_tasks)}")
    print_rank_0(
        f"Phase-1 gradient-checkpointing tasks: {sorted(args.ckpt_tasks)}; "
        f"phase-2 batch={args.router_retune_batch_size}, checkpointing=ON",
        args.global_rank)

    train_task_list, eval_task_list, test_task_list = {}, {}, {}
    for dataset in datasets:
        dataset_path = os.path.join(args.data_path, dataset)
        train_dataset, eval_dataset, test_dataset = create_prompt_dataset(
            args.local_rank, dataset_path, args.data_output_path, args.seed)

        if args.local_rank == -1:
            train_sampler, eval_sampler, test_sampler = (
                RandomSampler(train_dataset), SequentialSampler(eval_dataset), SequentialSampler(test_dataset))
        else:
            train_sampler, eval_sampler, test_sampler = (
                DistributedSampler(train_dataset, shuffle=True, seed=args.seed),
                DistributedSampler(eval_dataset, shuffle=False),
                DistributedSampler(test_dataset, shuffle=False))

        data_collator = DataCollator(tokenizer, padding="longest", max_prompt_len=args.max_prompt_len,
                                     max_ans_len=args.max_ans_len, pad_to_multiple_of=8, inference=False)
        inf_data_collator = DataCollator(tokenizer, model=model, padding="longest", max_prompt_len=args.max_prompt_len,
                                         max_ans_len=args.max_ans_len, pad_to_multiple_of=8, inference=True)

        train_task_list[dataset] = DataLoader(train_dataset, collate_fn=data_collator, sampler=train_sampler,
                                              batch_size=args.batch_by_task[dataset],
                                              num_workers=4, pin_memory=True)
        eval_task_list[dataset] = DataLoader(eval_dataset, collate_fn=data_collator, sampler=eval_sampler,
                                             batch_size=args.per_device_eval_batch_size)
        test_task_list[dataset] = DataLoader(test_dataset, collate_fn=inf_data_collator, sampler=test_sampler,
                                             batch_size=args.per_device_eval_batch_size)

    # NOTE: no engine is built here. Each task's add_experts_to_all_layers()
    # creates brand-new nn.Parameters (new OlmoeMLP experts + per-expert router rows) --
    # a DDP wrapper/optimizer built once up front would never see them.
    # Ours_MoE_FFN builds a fresh optimizer + DDP wrapper itself (_reinit_engine)
    # at the start of every phase.
    print_rank_0("***** Running MoE-FFN growing-expert continual training *****", args.global_rank)
    trainer = Ours_MoE_FFN(
        model, tokenizer, None, train_task_list, eval_task_list, test_task_list,
        args, router_pretrain_dataset=router_pretrain_dataset)
    trainer.train_continual()


if __name__ == "__main__":
    main()
