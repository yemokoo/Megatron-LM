#!/usr/bin/env python
"""Entry point for the Table-1 continual-learning baselines on TRACE.

Six methods, one contract.  Everything that is not the method itself -- chat
template, token cache, task order, epochs, global batch, optimizer, scheduler,
seed, checkpoint layout -- is taken from the same SLoRA profile Ours runs
under, so a Table-1 difference is a method difference.

  seq_lora       shared LoRA carried across tasks (the control)
  ewc            + post-convergence diagonal-Fisher penalty
  olora          per-task rank block, orthogonal to the frozen ones, merged
  mtl            joint training on all eight tasks (the ceiling)
  lifelong_moe   grow experts, freeze old experts/router rows, distil
  moe_lpr        grow experts, then router-only review with routing CE
  dymoe          LLaVA-DyMoE: per-projection expert banks, TAG + RSR
                 (--dymoe_variant incmoelora = the paper's no-TAG/no-RSR baseline)

``--moe_scope`` selects the FFN-only or shared QKVO+FFN expansion for the two
MoE rows, so each can be reported at Ours' trainable-parameter count.

This file is a sibling of main_Ours_LoRA_MoE.py, not a modification of it: the
Ours runner stays byte-identical so every previously produced Ours checkpoint
keeps its provenance.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import math
import os

import torch
import torch.distributed as dist
from datasets import load_from_disk
from torch.utils.data import (ConcatDataset, DataLoader, RandomSampler,
                              SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, SchedulerType

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from training.main_Ours_LoRA_MoE import (
    tokenizer_source_fingerprint as _ours_tokenizer_source_fingerprint)
from utils.chat_templates import ensure_llama31_chat_template
from utils.data.data_collator import (DataCollator,
                                      PreTokenizedSLoRATraceDataCollator,
                                      SLoRATraceDataCollator)
from utils.data.data_utils import create_prompt_dataset
from utils.model.model_utils import create_hf_model
from utils.utils import print_rank_0, set_random_seed

from model.tab1_baselines import TAB1_TRAINERS
from model.tab1_lora import (attach_olora_targets, attach_seq_lora_targets,
                             adapter_parameter_count, resolve_targets)
from model.tab1_moe import TAB1_MOE_TRAINERS, attach_shared_path
from model.tab1_dymoe import (DyMoETab1, attach_dymoe_targets,
                              grow_dymoe_old_banks)

ALL_TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
             "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]
LORA_METHODS = {"seq_lora", "ewc", "olora", "mtl"}
MOE_METHODS = {"lifelong_moe", "moe_lpr"}
DYMOE_METHODS = {"dymoe"}


def csv_strings(value):
    return value.split(",")


def csv_ints(value):
    return [int(item) for item in value.split(",")]


def tokenizer_source_fingerprint(model_path):
    """Hash the local tokenizer assets that determine cached token IDs.

    Delegates to the Ours runner rather than restating the recipe.  A
    reimplementation here that merely looked equivalent -- same four files,
    same hash -- already rejected the shared cache once because it dropped the
    filename separator and the chat-template contribution.  The cache is shared
    with every Ours run, so the two functions have to be the same function.
    """
    return _ours_tokenizer_source_fingerprint(model_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=sorted(LORA_METHODS | MOE_METHODS
                                       | DYMOE_METHODS))
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--dataset_name", type=csv_strings, default=["all"])
    parser.add_argument("--data_output_path", default="/tmp/tab1_data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_train_epochs", type=csv_ints,
                        default=[5, 3, 7, 5, 3, 5, 5, 7])
    parser.add_argument("--per_device_train_batch_size", type=csv_ints,
                        default=[8])
    parser.add_argument("--gradient_accumulation_steps", type=csv_ints,
                        default=[2])
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--max_prompt_len", type=int, default=1024)
    parser.add_argument("--max_ans_len", type=int, default=512)
    parser.add_argument("--max_train_len", type=int, default=1024)
    parser.add_argument("--train_format", default="slora_chat_full",
                        choices=["slora_chat_full", "trace_legacy"])
    parser.add_argument("--tokenized_train_cache_dir", default="")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--lr_scheduler_type", type=SchedulerType,
                        default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--start_task", type=int, default=0)
    parser.add_argument("--resume_from", default="",
                        help="A previous run directory. Its highest completed "
                             "round is loaded and training continues from the "
                             "next task, so an interrupted sweep cell does not "
                             "repeat the tasks it already finished.")
    parser.add_argument("--loss_log_interval", type=int, default=10)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--gradient_checkpointing_tasks", default="")
    parser.add_argument("--disable_dropout", action="store_true")
    parser.add_argument("--max_train_steps_per_task", type=int, default=0,
                        help="Smoke-test cap on micro-batches per phase; "
                             "0 uses the complete loader.")

    # Shared LoRA parameterization (seq_lora / ewc / olora / mtl).
    parser.add_argument("--lora_targets", default=None,
                        choices=["ffn", "attn", "all7", "qv"],
                        help="Default depends on the method: all7 (the "
                             "SLoRA/Seq-LoRA contract) for every row except "
                             "O-LoRA, which defaults to qv because that is "
                             "what both the official repo and TRACE's port "
                             "actually adapt. See resolve_lora_targets.")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=float, default=128.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # EWC
    parser.add_argument("--ewc_lambda", type=float, default=400.0)
    parser.add_argument("--ewc_mode", default="online",
                        choices=["online", "per_task"])
    parser.add_argument("--ewc_fisher_samples", type=int, default=1000,
                        help="0 walks the whole task train split.")
    parser.add_argument(
        "--ewc_fisher_batch_size", type=int, default=1,
        help="Micro-batch for the Fisher pass. That pass runs the model in "
             "eval() mode, which disables HF's gradient-checkpointing gate "
             "(`self.training`), so it retains full activations and needs a "
             "much smaller batch than checkpointed training does at the "
             "same memory budget.")
    parser.add_argument("--ewc_state_device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Where the Fisher and anchors live between steps. "
                             "cpu moves ~1.3GB over PCIe per step (11 s/step "
                             "measured); use it only if memory forces it.")

    # O-LoRA.  One value applies to every task; a comma list of eight
    # reproduces the official per-position schedule from scripts/long.sh.
    parser.add_argument("--olora_lambda_orthogonal", default="0.5")
    parser.add_argument("--olora_lambda_l2", default="0.0")
    parser.add_argument("--olora_merge_at_end", type=int, default=1)

    # MTL.  Five epochs over the 40,000-record union equals the 200,000
    # samples the sequential schedule consumes.
    parser.add_argument("--mtl_epochs", type=int, default=5)

    # Knobs the inherited Ours base class reads directly. The MoE rows reuse
    # Ours_LoRA_MoE.train_continual for its per-task wall-time/FLOP accounting,
    # and that method reads these off args without a default.
    parser.add_argument("--disable_training_flop_counter", type=int, default=1)
    parser.add_argument("--router_retune_epochs", type=int, default=0,
                        help="Ours v1's post-task router phase. The Table-1 "
                             "MoE rows drive their own phases, so this stays 0.")
    # Ours_LoRA_MoE.save_model records these in its lora_moe_meta.json. The
    # Table-1 rows write their own metadata and never reach that method, but a
    # future edit that does would otherwise crash after a full training run
    # rather than at startup. Defined, unused, and deliberately inert.
    for _inert, _default in (("v2_memory_batch_size", 0),
                             ("v2_kd_loss_coeff", 0.0),
                             ("v2_kd_temperature", 1.0),
                             ("v2_kd_learning_rate", 0.0),
                             ("v2_kd_chunk_tokens", 256),
                             ("v2_joint_replay_loss_coeff", 0.0),
                             ("v2_max_replay_batches_per_step", 0)):
        parser.add_argument(f"--{_inert}", type=type(_default),
                            default=_default, help=argparse.SUPPRESS)
    parser.add_argument("--v2_kd_token_scope", default="nonpad",
                        help=argparse.SUPPRESS)

    # MoE expansion (lifelong_moe / moe_lpr)
    parser.add_argument("--moe_scope", default="ffn",
                        choices=["ffn", "ffn_attn"])
    parser.add_argument("--lora_moe_rank", type=int, default=64)
    parser.add_argument("--lora_moe_alpha", type=float, default=128.0)
    parser.add_argument("--lora_moe_dropout", type=float, default=0.05)
    parser.add_argument("--experts_per_task", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--routing_weight_mode", default="straight_through_topk",
                        choices=["full_softmax", "topk_softmax",
                                 "straight_through_topk"])
    parser.add_argument("--moe_aux_loss_coeff", type=float, default=0.01)
    parser.add_argument("--moe_z_loss_coeff", type=float, default=0.001)

    # Replay pool shared by MoE-LPR's review phase.
    # 10% of each task, i.e. 500 of TRACE's 5,000 records -- the same memory
    # Ours keeps (OURS_V2_NEW_PERSISTENT_SAMPLES_PER_TASK=500). The wiki study
    # used 0.1%, but wiki tasks hold 2.1B tokens against TRACE's 0.4-4.2M, so
    # every way of carrying that ratio over lands on 5 records. The memory
    # SIZE is therefore set by the TRACE contract, not transferred; what is
    # transferred is the exposure budget (20% of one epoch = 1,000 samples).
    parser.add_argument("--replay_subset_ratio", type=float, default=0.1)
    parser.add_argument("--replay_subset_seed", type=int, default=-1)
    parser.add_argument("--replay_manifest_path", default="")
    parser.add_argument("--replay_distribution", default="equal_task")
    parser.add_argument("--replay_recency_power", type=float, default=1.0)
    parser.add_argument("--replay_selection_mode", default="random")
    parser.add_argument("--router_replay_exposure_samples", type=int,
                        default=1000)

    # Lifelong-MoE
    parser.add_argument("--lifelong_shared_targets", default="attn",
                        choices=["none", "attn", "ffn", "all7", "qv"])
    parser.add_argument("--lifelong_train_shared", type=int, default=1)
    parser.add_argument("--lifelong_kd_coeff", type=float, default=1.0)
    parser.add_argument("--lifelong_kd_temperature", type=float, default=1.0)
    parser.add_argument("--lifelong_kd_chunk_tokens", type=int, default=256,
                        help="Tokens per chunk in the distillation term; "
                             "matches Ours --v2_kd_chunk_tokens. The one-shot "
                             "form needs ~40 GB at micro-batch 16.")
    # Online L2 anchor, paper Eq. 5.  The wiki fork carries the same knob and
    # runs with it at zero, using the KL term alone; kept at zero here so the
    # two studies match by default.
    parser.add_argument("--lifelong_l2_coeff", type=float, default=0.0)
    parser.add_argument("--lifelong_allow_no_shared_path", type=int, default=0,
                        help="Opt in to a Lifelong-MoE run with no shared "
                             "adapter. Such a run is an ablation, not the "
                             "published method; see model/tab1_moe.py.")

    # MoE-LPR
    parser.add_argument("--lpr_gamma", type=float, default=0.1)
    # old: every older task's tokens -> log-sum-exp over all old experts (paper's
    # single "original" group, applied cumulatively). task: legacy per-task expert.
    parser.add_argument("--lpr_label_mode", default="old", choices=["old", "task"])
    # The wiki study set the review phase to 360 updates against 1,800 training
    # updates. Expressed as a fraction it transfers to TRACE, where each task
    # has a different step count; a fixed step number would not.
    parser.add_argument("--lpr_review_fraction", type=float, default=0.2)

    # LLaVA-DyMoE. Defaults are the upstream scripts/Train/*.sh values; rank,
    # alpha and dropout come from --lora_rank/--lora_alpha/--lora_dropout, and
    # --lora_rank is split evenly over the experts (64 / 16 = rank-4 experts).
    parser.add_argument("--dymoe_variant", default="dymoe",
                        choices=["dymoe", "incmoelora"],
                        help="incmoelora is the paper's IncMoELoRA baseline: "
                             "the same layer with TAG off and RSR at zero.")
    parser.add_argument("--dymoe_experts_per_task", type=int, default=16)
    parser.add_argument("--dymoe_top_k", type=int, default=16)
    parser.add_argument("--dymoe_router_temperature", type=float, default=0.01)
    parser.add_argument("--dymoe_cosine_scale", type=float, default=1.0)
    parser.add_argument("--dymoe_tag", type=int, default=1)
    parser.add_argument("--dymoe_conflict_ratio", type=float, default=0.2)
    parser.add_argument("--dymoe_exc_coeff", type=float, default=1e-3)
    parser.add_argument("--dymoe_spe_coeff", type=float, default=1e-3)
    parser.add_argument("--dymoe_rsr_temperature", type=float, default=0.1)
    parser.add_argument("--dymoe_rsr_start_fraction", type=float, default=0.5)
    args = parser.parse_args()
    if args.method == "dymoe" and args.dymoe_variant == "incmoelora" and (
            args.dymoe_tag or args.dymoe_exc_coeff or args.dymoe_spe_coeff):
        parser.error("--dymoe_variant incmoelora requires --dymoe_tag 0 "
                     "--dymoe_exc_coeff 0 --dymoe_spe_coeff 0")
    return args


# Official O-LoRA passes no target_modules to LoraConfig, so PEFT's Llama
# default applies and only q_proj/v_proj are adapted; TRACE's own port agrees,
# its merge step handling exactly those two names. Every other Table-1 row
# follows the SLoRA/Seq-LoRA contract of all seven projections.
METHOD_DEFAULT_TARGETS = {"olora": "qv"}


def resolve_lora_targets(args):
    if args.lora_targets is not None:
        return args.lora_targets
    return METHOD_DEFAULT_TARGETS.get(args.method, "all7")


def resume_state(model, args):
    """Load the newest round of --resume_from and set the next task index.

    Adapter-only checkpoints carry no optimizer state, which is fine here: each
    task rebuilds its optimizer anyway (the expert pool and the freeze pattern
    change between tasks). What must be restored is the grown expert count --
    the state dict cannot be applied to a model that has not grown to the same
    shape yet.
    """
    if not args.resume_from:
        return
    rounds = sorted(
        (int(name) for name in os.listdir(args.resume_from)
         if name.isdigit()
         and os.path.isfile(os.path.join(args.resume_from, name,
                                         "tab1_meta.json"))),
        reverse=True)
    if not rounds:
        raise ValueError(f"--resume_from has no completed round: {args.resume_from}")
    last = rounds[0]
    directory = os.path.join(args.resume_from, str(last))
    with open(os.path.join(directory, "tab1_meta.json"), encoding="utf-8") as handle:
        meta = json.load(handle)

    if args.method in MOE_METHODS:
        from model.tab1_moe import build_scope
        build_scope(args.moe_scope).add_experts(model, meta["num_experts"])
    elif args.method in DYMOE_METHODS:
        grow_dymoe_old_banks(model, int(meta["total_expert_num"]))
    elif args.method == "olora":
        from model.tab1_lora import set_olora_task
        set_olora_task(model, int(meta.get("current_task", last)))

    state = torch.load(os.path.join(directory, "pytorch_model.bin"),
                       map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"resume state has unexpected keys: {unexpected[:5]}")
    args.start_task = last + 1
    print_rank_0(
        f"[resume] {directory}: round {last} loaded "
        f"({len(state)} tensors, experts={meta.get('num_experts', 'n/a')}); "
        f"continuing from task {args.start_task}", args.global_rank)


def build_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, use_fast=True,
        local_files_only=True)
    if args.train_format != "slora_chat_full":
        return tokenizer
    if tokenizer.pad_token is None:
        if "llama" not in args.model_name_or_path.lower():
            raise ValueError(
                "SLoRA training requires a configured non-Llama pad token")
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = 128004
    args.chat_template_source = ensure_llama31_chat_template(
        tokenizer, args.model_name_or_path)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return tokenizer


def load_token_cache_manifest(args, tokenizer):
    if not args.tokenized_train_cache_dir:
        return None
    if args.train_format != "slora_chat_full":
        raise ValueError(
            "--tokenized_train_cache_dir supports slora_chat_full only")
    with open(os.path.join(args.tokenized_train_cache_dir, "manifest.json"),
              encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected_length = args.max_train_len or (
        args.max_prompt_len + args.max_ans_len)
    checks = {
        "format": "slora_chat_full",
        "max_length": expected_length,
        "tokenizer_fingerprint": tokenizer_source_fingerprint(
            args.model_name_or_path),
    }
    mismatches = {key: (manifest.get(key), value)
                  for key, value in checks.items()
                  if manifest.get(key) != value}
    if mismatches:
        raise ValueError(f"incompatible token cache: {mismatches}")
    return manifest


def attach_layout(model, args):
    """Install the adapter layout the chosen method needs, and report it."""
    if args.method in LORA_METHODS:
        targets = resolve_targets(args.lora_targets)
        if args.method == "olora":
            # One rank-``lora_rank`` block per task keeps the trainable count
            # equal to Seq-LoRA's at every task, which is the comparison the
            # table makes; the blocks fold into the backbone at the end so the
            # inference cost stays equal too.
            wrapped = attach_olora_targets(
                model, targets, args.lora_rank, args.lora_alpha,
                num_tasks=len(args.task_names), dropout=args.lora_dropout)
        else:
            wrapped = attach_seq_lora_targets(
                model, targets, args.lora_rank, args.lora_alpha,
                args.lora_dropout)
        print_rank_0(f"[layout] {args.method}: {wrapped} projections on "
                     f"{args.lora_targets}", args.global_rank)
        return model

    if args.method in DYMOE_METHODS:
        wrapped = attach_dymoe_targets(
            model, resolve_targets(args.lora_targets), args.lora_rank,
            args.lora_alpha, args.lora_dropout, args.dymoe_experts_per_task,
            args.dymoe_top_k, args.dymoe_router_temperature,
            args.dymoe_cosine_scale)
        print_rank_0(
            f"[layout] {args.dymoe_variant}: {wrapped} projections on "
            f"{args.lora_targets}, {args.dymoe_experts_per_task} experts x "
            f"rank {args.lora_rank // args.dymoe_experts_per_task} per task",
            args.global_rank)
        return model

    # MoE rows: the shared path, if any, must be wrapped before the expert
    # layout takes ownership of the projections it routes.
    shared = 0
    if args.method == "lifelong_moe":
        shared = attach_shared_path(
            model, args.moe_scope, args.lifelong_shared_targets,
            args.lora_rank, args.lora_alpha, args.lora_dropout)
    from model.tab1_moe import build_scope
    build_scope(args.moe_scope).attach(model, args)
    print_rank_0(f"[layout] {args.method}: scope={args.moe_scope}, "
                 f"shared adapters={shared}", args.global_rank)
    return model


def build_loaders(args, tokenizer, model, manifest):
    train_task_list, eval_task_list, test_task_list = {}, {}, {}
    raw_train_datasets = []
    for dataset_name in args.task_names:
        dataset_path = os.path.join(args.data_path, dataset_name)
        train_dataset, eval_dataset, test_dataset = create_prompt_dataset(
            args.local_rank, dataset_path, args.data_output_path, args.seed)
        if manifest is not None:
            entry = manifest.get("tasks", {}).get(dataset_name)
            if entry is None:
                raise KeyError(f"token cache has no entry for {dataset_name}")
            cached = load_from_disk(
                os.path.join(args.tokenized_train_cache_dir, entry["path"]))
            if len(cached) != len(train_dataset):
                raise ValueError(
                    f"token cache size mismatch for {dataset_name}: "
                    f"{len(cached)} != {len(train_dataset)}")
            train_dataset = cached
        raw_train_datasets.append(train_dataset)

        if args.local_rank == -1:
            samplers = (RandomSampler(train_dataset),
                        SequentialSampler(eval_dataset),
                        SequentialSampler(test_dataset))
        else:
            samplers = (DistributedSampler(train_dataset),
                        DistributedSampler(eval_dataset),
                        DistributedSampler(test_dataset))
        train_sampler, eval_sampler, test_sampler = samplers

        if args.train_format == "slora_chat_full":
            max_length = args.max_train_len or (
                args.max_prompt_len + args.max_ans_len)
            eval_collator = SLoRATraceDataCollator(
                tokenizer, max_length=max_length, label_scope="answer")
            if manifest is not None:
                collator = PreTokenizedSLoRATraceDataCollator(tokenizer)
            else:
                collator = SLoRATraceDataCollator(
                    tokenizer, max_length=max_length, label_scope="full")
        else:
            collator = DataCollator(
                tokenizer, padding="longest",
                max_prompt_len=(args.max_train_len or args.max_prompt_len),
                max_ans_len=(0 if args.max_train_len else args.max_ans_len),
                pad_to_multiple_of=8, inference=False)
            eval_collator = collator
        inference_collator = DataCollator(
            tokenizer, model=model, padding="longest",
            max_prompt_len=args.max_prompt_len, max_ans_len=args.max_ans_len,
            pad_to_multiple_of=8, inference=True)

        train_task_list[dataset_name] = DataLoader(
            train_dataset, collate_fn=collator, sampler=train_sampler,
            batch_size=args.batch_by_task[dataset_name], num_workers=4,
            pin_memory=True)
        eval_task_list[dataset_name] = DataLoader(
            eval_dataset, collate_fn=eval_collator, sampler=eval_sampler,
            batch_size=args.per_device_eval_batch_size)
        test_task_list[dataset_name] = DataLoader(
            test_dataset, collate_fn=inference_collator, sampler=test_sampler,
            batch_size=args.per_device_eval_batch_size)

    joint_loader = None
    if args.method == "mtl":
        joint_dataset = ConcatDataset(raw_train_datasets)
        joint_sampler = (RandomSampler(joint_dataset)
                         if args.local_rank == -1
                         else DistributedSampler(joint_dataset))
        micro_batch = min(args.batch_by_task.values())
        joint_loader = DataLoader(
            joint_dataset,
            collate_fn=train_task_list[args.task_names[0]].collate_fn,
            sampler=joint_sampler, batch_size=micro_batch, num_workers=4,
            pin_memory=True)
        print_rank_0(f"[mtl] joint dataset: {len(joint_dataset)} records, "
                     f"{args.mtl_epochs} epochs", args.global_rank)
    return train_task_list, eval_task_list, test_task_list, joint_loader


def resolve_batch_contract(args, world_size):
    batch = args.per_device_train_batch_size
    if len(batch) == 1:
        batch = batch * len(args.task_names)
    accum = args.gradient_accumulation_steps
    if len(accum) == 1:
        accum = accum * len(args.task_names)
    if len(batch) != len(args.task_names) or len(accum) != len(args.task_names):
        raise ValueError("batch/accumulation need 1 or one value per task")
    args.batch_by_task = dict(zip(args.task_names, batch))
    args.grad_accum_by_task = dict(zip(args.task_names, accum))
    effective = {name: args.batch_by_task[name] * world_size
                 * args.grad_accum_by_task[name] for name in args.task_names}
    if len(set(effective.values())) != 1:
        raise ValueError(
            f"per-task batch/accumulation must hold one global batch: {effective}")
    args.effective_global_batch = next(iter(effective.values()))
    args.gradient_accumulation_steps = args.grad_accum_by_task[
        args.task_names[0]]
    print_rank_0(f"[contract] effective global batch "
                 f"{args.effective_global_batch}", args.global_rank)


def main():
    args = parse_args()
    args.task_names = (list(ALL_TASKS)
                       if args.dataset_name in (["all"], "all")
                       else list(args.dataset_name))
    if len(args.num_train_epochs) == 1:
        args.num_train_epochs = args.num_train_epochs * len(args.task_names)
    if len(args.num_train_epochs) != len(args.task_names):
        raise ValueError("--num_train_epochs needs 1 or one value per task")

    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl")
    args.global_rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    resolve_batch_contract(args, world_size)
    args.ckpt_tasks = (set(args.task_names) if args.gradient_checkpointing
                       else {t for t in args.gradient_checkpointing_tasks.split(",") if t})
    unknown = args.ckpt_tasks - set(args.task_names)
    if unknown:
        raise ValueError(f"--gradient_checkpointing_tasks unknown: {unknown}")

    args.lora_targets = resolve_lora_targets(args)
    print_rank_0(f"[layout] lora_targets={args.lora_targets} "
                 f"(method default for {args.method})", args.global_rank)

    set_random_seed(args.seed)
    if dist.is_initialized():
        dist.barrier()

    tokenizer = build_tokenizer(args)
    manifest = load_token_cache_manifest(args, tokenizer)
    args.use_pretokenized_train_cache = manifest is not None

    model = create_hf_model(
        AutoModelForCausalLM, args.model_name_or_path, tokenizer,
        disable_dropout=args.disable_dropout, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True).to(device=device)
    attach_layout(model, args)
    resume_state(model, args)
    print_rank_0(f"[parameters] {adapter_parameter_count(model)}",
                 args.global_rank)

    train_task_list, eval_task_list, test_task_list, joint_loader = (
        build_loaders(args, tokenizer, model, manifest))

    if args.method in LORA_METHODS:
        trainer_class = TAB1_TRAINERS[args.method]
    elif args.method in DYMOE_METHODS:
        trainer_class = DyMoETab1
    else:
        trainer_class = TAB1_MOE_TRAINERS[args.method]
    trainer = trainer_class(model, tokenizer, None, train_task_list,
                            eval_task_list, test_task_list, args)
    if joint_loader is not None:
        trainer.joint_loader = joint_loader

    if args.global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "tab1_run.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({k: (sorted(v) if isinstance(v, set) else v)
                       for k, v in vars(args).items()
                       if isinstance(v, (str, int, float, bool, list, set,
                                         dict, type(None)))},
                      handle, indent=2, default=str)

    print_rank_0(f"***** Table-1 baseline: {args.method} *****",
                 args.global_rank)
    trainer.train_continual()


if __name__ == "__main__":
    main()
