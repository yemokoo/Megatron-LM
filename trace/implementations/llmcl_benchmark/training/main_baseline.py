#!/usr/bin/env python
# Baseline entry point for the TRACE CL comparison. One entry covers all controls:
#   --baseline finetune        : full-finetune the whole model, sequential over 8 tasks
#   --baseline static_lora_moe  : pre-add N LoRA experts (frozen backbone), train them
#                                 (Qwen LoRAMoE = N=8*ept ; SeqLoRA = N=1, top_k 1)
#   --baseline static_moe_ffn   : pre-add N full FFN experts on OLMoE (frozen backbone)
# No task-wise growth, no phase-2 retune -- that is what Ours_* adds on top.
#
# Distributed via plain torchrun + DistributedDataParallel, NOT DeepSpeed -- see
# main_Ours_LoRA_MoE.py for why.
import sys
sys.dont_write_bytecode = True

import argparse
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, SchedulerType

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import DataCollator
from utils.utils import print_rank_0, set_random_seed, load_hf_tokenizer
from utils.model.model_utils import create_hf_model
from model.baselines import StaticBaseline

AllDatasetName = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
                  "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]


def list_of_strings(arg):
    return arg.split(',')


def parse_args():
    parser = argparse.ArgumentParser(description="TRACE CL baselines (finetune / static expert pool)")
    parser.add_argument('--baseline', type=str, required=True,
                        choices=['finetune', 'static_lora_moe', 'static_moe_ffn'])
    parser.add_argument('--static_experts', type=int, default=8,
                        help='Total experts pre-added ONCE (static_* baselines). Ignored by finetune.')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=list_of_strings, default='all')
    parser.add_argument('--data_output_path', type=str, default='/tmp/data_files/')
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--max_prompt_len", type=int, default=1024)
    parser.add_argument("--max_ans_len", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--num_train_epochs", type=list_of_strings, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--loss_log_interval", type=int, default=10)
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="constant_with_warmup")
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help='Overridden by the LOCAL_RANK env var when launched via torchrun.')
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--disable_dropout', action='store_true')

    # expert/router hyperparams for the static_* baselines (match Ours for a fair control)
    parser.add_argument('--lora_moe_rank', type=int, default=8)
    parser.add_argument('--lora_moe_alpha', type=int, default=32)
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--routing_weight_mode', type=str, default='full_softmax',
                        choices=['full_softmax', 'topk_softmax'])
    parser.add_argument('--moe_aux_loss_coeff', type=float, default=0.01)
    parser.add_argument('--moe_z_loss_coeff', type=float, default=0.001)

    return parser.parse_args()


def prepare_model(args, tokenizer):
    """Return model, MoE loss callback, metadata saver, and router-mask setter."""
    model = create_hf_model(AutoModelForCausalLM, args.model_name_or_path, tokenizer,
                            disable_dropout=args.disable_dropout,
                            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

    if args.baseline == 'finetune':
        return model, None, None, None  # whole model trainable, no experts

    if args.baseline == 'static_lora_moe':
        from model.Ours_LoRA_MoE import (attach_lora_moe, add_experts_to_all_layers,
                                          freeze_lora_moe_experts, freeze_lora_moe_routers,
                                          collect_moe_losses, save_lora_moe_meta,
                                          set_router_token_mask)
        attach_lora_moe(model, r=args.lora_moe_rank, alpha=args.lora_moe_alpha,
                        top_k=args.top_k, aux_loss_coeff=args.moe_aux_loss_coeff,
                        z_loss_coeff=args.moe_z_loss_coeff,
                        routing_weight_mode=args.routing_weight_mode)
        add_experts_to_all_layers(model, args.static_experts)          # all N at once
        freeze_lora_moe_experts(model, trainable_expert_indices=set(range(args.static_experts)))
        freeze_lora_moe_routers(model, trainable=True)
        return model, collect_moe_losses, save_lora_moe_meta, set_router_token_mask

    if args.baseline == 'static_moe_ffn':
        from model.Ours_MoE_FFN import (attach_growing_moe, add_experts_to_all_layers,
                                        freeze_moe_experts, freeze_moe_routers,
                                        collect_moe_losses, save_moe_ffn_meta,
                                        set_router_token_mask)
        attach_growing_moe(model, aux_loss_coeff=args.moe_aux_loss_coeff,
                           z_loss_coeff=args.moe_z_loss_coeff)
        add_experts_to_all_layers(model, args.static_experts)          # all N at once
        # train the N added experts + their router rows; backbone + original gate frozen
        freeze_moe_experts(model, trainable_new_indices=set(range(args.static_experts)))
        freeze_moe_routers(model,
                           trainable_new_indices=set(range(args.static_experts)),
                           trainable_original=False)
        return model, collect_moe_losses, save_moe_ffn_meta, set_router_token_mask

    raise ValueError(args.baseline)


def main():
    args = parse_args()
    if args.gradient_accumulation_steps < 1 or args.loss_log_interval < 1:
        raise ValueError("gradient accumulation and loss log interval must be positive")
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))

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

    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    assert tokenizer.padding_side == 'left'
    assert tokenizer.truncation_side == "left"

    model, moe_loss_fn, meta_saver, router_mask_fn = prepare_model(args, tokenizer)
    model = model.to(device=device)

    datasets = AllDatasetName if args.dataset_name[0] == "all" else args.dataset_name
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
                DistributedSampler(train_dataset), DistributedSampler(eval_dataset), DistributedSampler(test_dataset))
        data_collator = DataCollator(tokenizer, padding="longest", max_prompt_len=args.max_prompt_len,
                                     max_ans_len=args.max_ans_len, pad_to_multiple_of=8, inference=False)
        inf_data_collator = DataCollator(tokenizer, model=model, padding="longest", max_prompt_len=args.max_prompt_len,
                                         max_ans_len=args.max_ans_len, pad_to_multiple_of=8, inference=True)
        train_task_list[dataset] = DataLoader(train_dataset, collate_fn=data_collator, sampler=train_sampler,
                                              batch_size=args.per_device_train_batch_size,
                                              num_workers=4, pin_memory=True)
        eval_task_list[dataset] = DataLoader(eval_dataset, collate_fn=data_collator, sampler=eval_sampler,
                                             batch_size=args.per_device_eval_batch_size)
        test_task_list[dataset] = DataLoader(test_dataset, collate_fn=inf_data_collator, sampler=test_sampler,
                                             batch_size=args.per_device_eval_batch_size)

    if args.gradient_checkpointing:
        # Needed whenever the backbone is frozen (static_lora_moe / static_moe_ffn) --
        # see main_Ours_LoRA_MoE.py for why. Harmless no-op for `finetune`, where the
        # embeddings already require grad.
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    print_rank_0(f"***** Running baseline: {args.baseline} "
                 f"(static_experts={args.static_experts if args.baseline!='finetune' else '-'}) *****",
                 args.global_rank)
    trainer = StaticBaseline(model, tokenizer, train_task_list, eval_task_list, test_task_list,
                             args, moe_loss_fn=moe_loss_fn, meta_saver=meta_saver,
                             router_mask_fn=router_mask_fn)
    trainer.train_continual()


if __name__ == "__main__":
    main()
