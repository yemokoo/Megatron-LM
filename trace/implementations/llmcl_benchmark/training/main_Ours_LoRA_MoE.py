#!/usr/bin/env python
# Adapted from TRACE (BeyonderXX/TRACE, Apache-2.0) training/main.py.
# Stripped to the single method this repo implements: growing FFN LoRA-MoE
# experts per task (see model/Ours_LoRA_MoE.py). All other CL_method branches,
# the llama/bloom flash-attn monkey-patches, and the DeepSpeed hand-rolled
# LoRA (utils/module/lora.py) from the original are intentionally dropped --
# Qwen models use transformers' own attention implementation, and our LoRA-MoE
# module replaces the FFN directly rather than going through peft/deepspeed-lora.
#
# Distributed via plain torchrun + DistributedDataParallel, NOT DeepSpeed: the
# trainer re-initializes its engine every phase (growth creates new
# nn.Parameters), and repeated deepspeed.initialize() on the same model leaks
# GPU memory without bound. See model/Ours_LoRA_MoE.py for the full story.
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, SchedulerType
from datasets import load_from_disk

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import (DataCollator, SLoRATraceDataCollator,
                                      PreTokenizedSLoRATraceDataCollator)
from utils.utils import print_rank_0, set_random_seed, load_hf_tokenizer
from utils.model.model_utils import create_hf_model
from model.Ours_LoRA_MoE import (Ours_LoRA_MoE, Ours_LoRA_MoE_V2,
                                 attach_lora_moe,
                                 add_experts_to_all_layers)

AllDatasetName = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
                  "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]


def list_of_strings(arg):
    return arg.split(',')


def list_of_ints(arg):
    return [int(x) for x in arg.split(',')]


def tokenizer_source_fingerprint(model_path):
    """Hash local tokenizer assets that determine cached token IDs."""
    digest = hashlib.sha256()
    found = False
    for name in ("tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "added_tokens.json"):
        path = os.path.join(model_path, name)
        if os.path.isfile(path):
            found = True
            digest.update(name.encode("utf-8") + b"\0")
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    if not found:
        raise FileNotFoundError(
            f"no tokenizer assets found under {model_path}")
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description="Growing FFN LoRA-MoE continual learning")
    parser.add_argument('--data_path', type=str, required=True,
                        help='Root dir with one subfolder per task (train/eval/test.json).')
    parser.add_argument('--dataset_name', type=list_of_strings, default='all',
                        help='Comma-separated task names, in training order. "all" = AllDatasetName order.')
    parser.add_argument('--data_output_path', type=str, default='/tmp/data_files/')
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=list_of_ints, default=[4],
                        help='Per-device train batch size. One int (uniform), or a comma '
                             'list matching --dataset_name order for per-task batches '
                             '(short tasks can afford a bigger batch than long ones like '
                             'MeetingBank). Phase-2 replay uses the MIN over its tasks.')
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--max_prompt_len", type=int, default=1024)
    parser.add_argument("--max_ans_len", type=int, default=512)
    parser.add_argument("--max_train_len", type=int, default=0,
                        help='Combined prompt+answer training cutoff; 0 uses max_prompt_len+max_ans_len.')
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument(
        "--train_format", choices=["raw_answer", "slora_chat_full"],
        default="raw_answer",
        help="slora_chat_full matches released SLoRA: backbone chat template, "
             "system/user/assistant text, full-sequence LM labels, right padding.")
    parser.add_argument("--num_train_epochs", type=list_of_strings, required=True,
                        help='Comma-separated epoch count per task, matching --dataset_name order.')
    parser.add_argument(
        "--tokenized_train_cache_dir", default="",
        help="Optional save_to_disk cache of unpadded slora_chat_full train IDs.")
    parser.add_argument(
        "--replay_manifest_path", default="",
        help="Optional shared manifest containing pre-sampled random replay indices.")
    parser.add_argument(
        "--gradient_accumulation_steps", type=list_of_ints, default=[1],
        help="Gradient accumulation. One int (uniform), or a comma list "
             "matching task order. Each task must preserve the same effective "
             "global batch with its per-device micro-batch.")
    parser.add_argument("--loss_log_interval", type=int, default=10,
                        help='Copy loss to CPU for progress logging every N microsteps.')
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="constant_with_warmup")
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0,
                        help='Per-phase optimizer-step warmup ratio; overrides num_warmup_steps when positive.')
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--resume_checkpoint", default="",
        help="Completed numeric round checkpoint to resume after. Restores all "
             "grown experts/router and starts at the following TRACE task.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help='Overridden by the LOCAL_RANK env var when launched via torchrun.')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Apply gradient checkpointing to ALL tasks. For per-task '
                             'control use --gradient_checkpointing_tasks instead.')
    parser.add_argument('--gradient_checkpointing_tasks', type=str, default='',
                        help='Comma-separated task names whose PHASE-1 trains with '
                             'gradient checkpointing ON (long-sequence tasks, e.g. '
                             'MeetingBank,Py150,20Minuten). All other tasks train with '
                             'it OFF (faster). Checkpointing is a pure memory/compute '
                             'tradeoff -- identical weights either way. NOTE: phase-2 '
                             'router-retune always runs with checkpointing ON, since '
                             'its replay set can contain long seen-task sequences.')
    parser.add_argument('--disable_dropout', action='store_true')
    parser.add_argument('--print_loss', action='store_true')

    # --- LoRA-MoE specific ---
    parser.add_argument('--training_version', choices=['v1', 'v2', 'v2_5'], default='v1',
                        help='v1: sequential new-task then router retune; v2: '
                             'KD-init plus joint new-task/router-replay updates; '
                             'v2_5: v2 with router aux/z losses disabled.')
    parser.add_argument('--experts_per_task', type=int, default=4,
                        help='New FFN LoRA experts added at the start of each task.')
    parser.add_argument('--lora_moe_rank', type=int, default=8)
    parser.add_argument('--lora_moe_alpha', type=int, default=32)
    parser.add_argument('--lora_moe_dropout', type=float, default=0.0)
    parser.add_argument('--top_k', type=int, default=2,
                        help='Experts activated per token (fixed for a run; vary across runs for ablation).')
    parser.add_argument('--routing_weight_mode', type=str, default='full_softmax',
                        choices=['full_softmax', 'topk_softmax'],
                        help='full_softmax keeps an LM-loss router gradient even at top-k=1; '
                             'topk_softmax is the legacy selected-set normalization.')
    parser.add_argument('--moe_aux_loss_coeff', type=float, default=0.01,
                        help='Switch-style load-balancing aux loss coefficient (matches LLM-continual-learning default).')
    parser.add_argument('--moe_z_loss_coeff', type=float, default=0.001,
                        help='Router logit z-loss coefficient (matches LLM-continual-learning default).')
    parser.add_argument('--router_retune_epochs', type=int, default=1,
                        help='v1 router-retune switch: 0 disables it; a positive value consumes the exact replay stream once.')
    parser.add_argument('--router_replay_exposure_samples', type=int, default=1000,
                        help='Exact global router-replay sample exposures per continual-learning round.')
    parser.add_argument('--past_task_ratio', type=float, default=1.0,
                        help='Deprecated legacy v1 full-data replay option; retained only for old commands.')

    # --- shared v1/v2 fixed replay memory ---
    parser.add_argument('--replay_subset_ratio', type=float, default=0.01,
                        help='Unique fixed subset stored per 5,000-sample task (0.01=50, 0.1=500).')
    parser.add_argument('--replay_distribution', choices=['equal_task', 'proportional'],
                        default='equal_task')
    parser.add_argument('--replay_subset_seed', type=int, default=-1,
                        help='Fixed per-task subset seed; -1 reuses --seed.')

    # --- v2 fixed-memory KD + exact-budget joint replay ---
    parser.add_argument('--v2_memory_batch_size', type=int, default=0,
                        help='KD/replay microbatch size per rank; 0 uses one sample per rank.')
    parser.add_argument('--v2_max_replay_batches_per_step', type=int, default=0,
                        help='Safety cap per microstep; 0 is unlimited.')
    parser.add_argument('--v2_joint_replay_loss_coeff', type=float, default=1.0)
    parser.add_argument('--v2_kd_loss_coeff', type=float, default=1.0)
    parser.add_argument('--v2_kd_temperature', type=float, default=1.0)
    parser.add_argument('--v2_kd_learning_rate', type=float, default=0.0,
                        help='0 reuses --learning_rate.')
    parser.add_argument('--v2_kd_chunk_tokens', type=int, default=256)
    parser.add_argument('--v2_kd_token_scope', choices=['nonpad', 'labels'], default='nonpad')
    parser.add_argument('--disable_training_flop_counter', action='store_true',
                        help='Disable operator FLOP counting; samples/tokens/steps/time are still recorded.')

    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.past_task_ratio <= 1.0:
        raise ValueError("--past_task_ratio must be in (0, 1]")
    if (not args.gradient_accumulation_steps
            or any(value < 1 for value in args.gradient_accumulation_steps)
            or args.loss_log_interval < 1):
        raise ValueError("gradient accumulation and loss log interval must be positive")
    if args.max_train_len < 0:
        raise ValueError("--max_train_len cannot be negative")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup_ratio must be in [0, 1)")
    if not 0.0 <= args.lora_moe_dropout < 1.0:
        raise ValueError("--lora_moe_dropout must be in [0, 1)")
    if not 0.0 <= args.adam_beta1 < 1.0 or not 0.0 <= args.adam_beta2 < 1.0:
        raise ValueError("Adam betas must be in [0, 1)")
    if not 0.0 < args.replay_subset_ratio <= 1.0:
        raise ValueError("--replay_subset_ratio must be in (0, 1]")
    if args.router_replay_exposure_samples < 1:
        raise ValueError("--router_replay_exposure_samples must be positive")
    if args.training_version in ('v2', 'v2_5'):
        if args.v2_memory_batch_size < 0 or args.v2_max_replay_batches_per_step < 0:
            raise ValueError("v2 replay batch sizes/caps cannot be negative")
        if args.v2_kd_temperature <= 0:
            raise ValueError("v2 KD temperature must be positive")
        if args.v2_kd_chunk_tokens < 1:
            raise ValueError("--v2_kd_chunk_tokens must be positive")
        if args.v2_kd_loss_coeff < 0 or args.v2_joint_replay_loss_coeff < 0:
            raise ValueError("v2 KD/replay loss coefficients cannot be negative")
    if (args.training_version == 'v2_5'
            and (args.moe_aux_loss_coeff != 0 or args.moe_z_loss_coeff != 0)):
        raise ValueError("v2_5 requires --moe_aux_loss_coeff 0 and --moe_z_loss_coeff 0")
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl")
    args.global_rank = torch.distributed.get_rank() if dist.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if dist.is_initialized() else 1
    if args.router_replay_exposure_samples % world_size != 0:
        raise ValueError(
            "--router_replay_exposure_samples must be divisible by world size "
            "for an exact non-duplicated global DDP exposure budget: "
            f"{args.router_replay_exposure_samples} % {world_size} != 0")

    set_random_seed(args.seed)
    if dist.is_initialized():
        torch.distributed.barrier()

    if args.train_format == "slora_chat_full":
        # Match the released SLoRA trainer rather than TRACE legacy defaults.
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True, use_fast=True,
            local_files_only=True)
        if tokenizer.pad_token is None:
            if "llama" not in args.model_name_or_path.lower():
                raise ValueError(
                    "SLoRA training requires a configured non-Llama pad token")
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
            tokenizer.pad_token_id = 128004
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
    else:
        tokenizer = load_hf_tokenizer(
            args.model_name_or_path, fast_tokenizer=True)
        assert tokenizer.padding_side == "left"
        assert tokenizer.truncation_side == "left"


    token_cache_manifest = None
    if args.tokenized_train_cache_dir:
        if args.train_format != "slora_chat_full":
            raise ValueError(
                "--tokenized_train_cache_dir supports slora_chat_full only")
        manifest_path = os.path.join(
            args.tokenized_train_cache_dir, "manifest.json")
        with open(manifest_path, encoding="utf-8") as handle:
            token_cache_manifest = json.load(handle)
        expected_length = args.max_train_len or (
            args.max_prompt_len + args.max_ans_len)
        checks = {
            "format": "slora_chat_full",
            "max_length": expected_length,
            "tokenizer_fingerprint": tokenizer_source_fingerprint(
                args.model_name_or_path),
        }
        mismatches = {
            key: (token_cache_manifest.get(key), value)
            for key, value in checks.items()
            if token_cache_manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"incompatible token cache: {mismatches}")
        print_rank_0(
            f"Using pre-tokenized training cache: "
            f"{args.tokenized_train_cache_dir}", args.global_rank)
    args.use_pretokenized_train_cache = token_cache_manifest is not None

    # Load directly in bf16 to avoid first materializing a full fp32 8B backbone.
    model = create_hf_model(AutoModelForCausalLM, args.model_name_or_path, tokenizer,
                            disable_dropout=args.disable_dropout,
                            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model = model.to(device=device)

    # Wrap every FFN with an (initially expert-less) LoRA-MoE module. Experts
    # are added per task inside Ours_LoRA_MoE.train_one_task, not here.
    attach_lora_moe(model, r=args.lora_moe_rank, alpha=args.lora_moe_alpha,
                    top_k=args.top_k, aux_loss_coeff=args.moe_aux_loss_coeff,
                    z_loss_coeff=args.moe_z_loss_coeff,
                    routing_weight_mode=args.routing_weight_mode,
                    dropout=args.lora_moe_dropout)
    if args.training_version == 'v2_5':
        applied_coeffs = {
            (layer.mlp.aux_loss_coeff, layer.mlp.z_loss_coeff)
            for layer in model.model.layers
        }
        if applied_coeffs != {(0.0, 0.0)}:
            raise RuntimeError(
                f"v2_5 router loss coefficients were not disabled: {applied_coeffs}")
        print_rank_0(
            "v2_5 router regularization verified: aux_loss_coeff=0, z_loss_coeff=0",
            args.global_rank)

    if args.resume_checkpoint:
        checkpoint_dir = os.path.normpath(args.resume_checkpoint)
        round_name = os.path.basename(checkpoint_dir)
        if not round_name.isdigit():
            raise ValueError("--resume_checkpoint must end in a numeric round")
        meta_path = os.path.join(checkpoint_dir, "lora_moe_meta.json")
        weights_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
        expected = {
            "r": args.lora_moe_rank,
            "alpha": args.lora_moe_alpha,
            "dropout": args.lora_moe_dropout,
            "top_k": args.top_k,
            "routing_weight_mode": args.routing_weight_mode,
            "training_version": args.training_version,
        }
        actual = dict(meta)
        actual.setdefault("training_version", "v1")
        mismatches = {key: (actual.get(key), value) for key, value in expected.items()
                      if actual.get(key) != value}
        expected_training_profile = {
            "format": args.train_format,
            "max_length": args.max_train_len or (
                args.max_prompt_len + args.max_ans_len),
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "adam_epsilon": args.adam_epsilon,
        }
        actual_training_profile = actual.get("training_profile", {})
        mismatches.update({
            f"training_profile.{key}": (actual_training_profile.get(key), value)
            for key, value in expected_training_profile.items()
            if actual_training_profile.get(key) != value
        })
        expected_replay = {
            "subset_ratio_per_task": args.replay_subset_ratio,
            "exposure_samples_per_round":
                args.router_replay_exposure_samples,
            "v1_router_retune_enabled": args.router_retune_epochs > 0,
            "distribution": args.replay_distribution,
            "subset_seed": args.replay_subset_seed,
        }
        actual_replay = actual.get("replay_memory", {})
        mismatches.update({
            f"replay_memory.{key}": (actual_replay.get(key), value)
            for key, value in expected_replay.items()
            if actual_replay.get(key) != value
        })
        if args.training_version in ("v2", "v2_5"):
            expected_v2 = {
                "memory_batch_size": args.v2_memory_batch_size,
                "kd_loss_coeff": args.v2_kd_loss_coeff,
                "kd_temperature": args.v2_kd_temperature,
                "kd_learning_rate": args.v2_kd_learning_rate,
                "kd_chunk_tokens": args.v2_kd_chunk_tokens,
                "kd_token_scope": args.v2_kd_token_scope,
                "joint_replay_loss_coeff": args.v2_joint_replay_loss_coeff,
                "max_replay_batches_per_step": args.v2_max_replay_batches_per_step,
            }
            actual_v2 = actual.get("v2", {})
            mismatches.update({
                f"v2.{key}": (actual_v2.get(key), value)
                for key, value in expected_v2.items()
                if actual_v2.get(key) != value
            })
        if mismatches:
            raise ValueError(f"resume hyperparameter mismatch: {mismatches}")
        completed_round = int(round_name)
        expected_experts = (completed_round + 1) * args.experts_per_task
        if int(meta["num_experts"]) != expected_experts:
            raise ValueError(
                f"round {completed_round} should have {expected_experts} experts, "
                f"checkpoint has {meta['num_experts']}")
        add_experts_to_all_layers(model, expected_experts)
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state, strict=False)
        grown_missing = [key for key in missing
                         if ".mlp.experts." in key or ".mlp.router." in key]
        if unexpected or grown_missing:
            raise RuntimeError(
                f"invalid resume state: unexpected={unexpected[:5]} "
                f"grown_missing={grown_missing[:5]}")
        args.start_task = completed_round + 1
        print_rank_0(
            f"Resumed Track1 after round {completed_round}: "
            f"experts={expected_experts}, next_task={args.start_task}",
            args.global_rank)

    datasets = AllDatasetName if args.dataset_name[0] == "all" else args.dataset_name

    # Resolve per-task train batch size: single value -> uniform; list -> by task order.
    bs = args.per_device_train_batch_size
    if len(bs) == 1:
        args.batch_by_task = {d: bs[0] for d in datasets}
    else:
        assert len(bs) == len(datasets), \
            f"--per_device_train_batch_size has {len(bs)} values but there are {len(datasets)} tasks"
        args.batch_by_task = {d: b for d, b in zip(datasets, bs)}
    grad_accum = args.gradient_accumulation_steps
    if len(grad_accum) == 1:
        args.grad_accum_by_task = {d: grad_accum[0] for d in datasets}
    else:
        if len(grad_accum) != len(datasets):
            raise ValueError(
                "--gradient_accumulation_steps must contain one value or "
                f"{len(datasets)} task values, got {len(grad_accum)}")
        args.grad_accum_by_task = {
            d: value for d, value in zip(datasets, grad_accum)}
    effective_batches = {
        d: args.batch_by_task[d] * world_size * args.grad_accum_by_task[d]
        for d in datasets
    }
    if len(set(effective_batches.values())) != 1:
        raise ValueError(
            "task-specific batch/accumulation values must preserve one "
            f"effective global batch, got {effective_batches}")
    args.effective_global_batch = next(iter(effective_batches.values()))
    args.gradient_accumulation_steps = args.grad_accum_by_task[datasets[0]]
    print_rank_0(f"per-task train batch: {args.batch_by_task}", args.global_rank)
    print_rank_0(
        f"per-task gradient accumulation: {args.grad_accum_by_task}; "
        f"effective global batch={args.effective_global_batch}",
        args.global_rank)

    # Resolve which tasks train with gradient checkpointing in PHASE 1. Global flag
    # wins (all tasks); otherwise the explicit per-task list. Phase-2 replay forces
    # it ON regardless (handled in the trainer).
    if args.gradient_checkpointing:
        args.ckpt_tasks = set(datasets)
    else:
        args.ckpt_tasks = {t for t in args.gradient_checkpointing_tasks.split(',') if t}
    unknown = args.ckpt_tasks - set(datasets)
    assert not unknown, f"--gradient_checkpointing_tasks names unknown tasks: {unknown}"
    print_rank_0(f"phase-1 grad-checkpointing tasks: {sorted(args.ckpt_tasks)} "
                 f"(phase-2 replay always ON)", args.global_rank)

    train_task_list, eval_task_list, test_task_list = {}, {}, {}
    for dataset in datasets:
        dataset_path = os.path.join(args.data_path, dataset)
        train_dataset, eval_dataset, test_dataset = create_prompt_dataset(
            args.local_rank, dataset_path, args.data_output_path, args.seed)


        if token_cache_manifest is not None:
            entry = token_cache_manifest.get("tasks", {}).get(dataset)
            if entry is None:
                raise KeyError(f"token cache has no task entry for {dataset}")
            cached_path = os.path.join(
                args.tokenized_train_cache_dir, entry["path"])
            cached_train_dataset = load_from_disk(cached_path)
            if len(cached_train_dataset) != len(train_dataset):
                raise ValueError(
                    f"token cache size mismatch for {dataset}: "
                    f"{len(cached_train_dataset)} != {len(train_dataset)}")
            train_dataset = cached_train_dataset

        if args.local_rank == -1:
            train_sampler, eval_sampler, test_sampler = (
                RandomSampler(train_dataset), SequentialSampler(eval_dataset), SequentialSampler(test_dataset))
        else:
            train_sampler, eval_sampler, test_sampler = (
                DistributedSampler(train_dataset), DistributedSampler(eval_dataset), DistributedSampler(test_dataset))

        if args.train_format == "slora_chat_full":
            eval_data_collator = SLoRATraceDataCollator(
                tokenizer, max_length=(args.max_train_len or (
                    args.max_prompt_len + args.max_ans_len)))
            if args.use_pretokenized_train_cache:
                data_collator = PreTokenizedSLoRATraceDataCollator(tokenizer)
            else:
                data_collator = eval_data_collator
        else:
            data_collator = DataCollator(
                tokenizer, padding="longest",
                max_prompt_len=(args.max_train_len or args.max_prompt_len),
                max_ans_len=(0 if args.max_train_len else args.max_ans_len),
                pad_to_multiple_of=8, inference=False)
            eval_data_collator = data_collator
        inf_data_collator = DataCollator(tokenizer, model=model, padding="longest", max_prompt_len=args.max_prompt_len,
                                         max_ans_len=args.max_ans_len, pad_to_multiple_of=8, inference=True)

        train_task_list[dataset] = DataLoader(train_dataset, collate_fn=data_collator, sampler=train_sampler,
                                              batch_size=args.batch_by_task[dataset],
                                              num_workers=4, pin_memory=True)
        eval_task_list[dataset] = DataLoader(eval_dataset, collate_fn=eval_data_collator, sampler=eval_sampler,
                                             batch_size=args.per_device_eval_batch_size)
        test_task_list[dataset] = DataLoader(test_dataset, collate_fn=inf_data_collator, sampler=test_sampler,
                                             batch_size=args.per_device_eval_batch_size)

    # Gradient checkpointing is applied PER TASK inside the trainer (train_one_task
    # -> _set_grad_ckpt), not globally here, so short tasks can run OFF (faster) while
    # long ones run ON. With the whole backbone frozen, a checkpointed segment's
    # embedding output has requires_grad=False and no grad_fn; _set_grad_ckpt pairs
    # gradient_checkpointing_enable with enable_input_require_grads to restore the
    # graph without unfreezing any weights.

    # NOTE: no engine is built here. Each task's add_experts_to_all_layers()
    # creates brand-new nn.Parameters (new expert A/B, a new (bigger) router
    # Linear) -- a DDP wrapper/optimizer built once, up front, would never see
    # or update them. Ours_LoRA_MoE instead builds a fresh optimizer and DDP
    # wrapper itself (_reinit_engine) at the start of every phase.
    print_rank_0(
        f"***** Running LoRA-MoE continual training ({args.training_version}) *****",
        args.global_rank)
    trainer_class = (
        Ours_LoRA_MoE_V2 if args.training_version in ('v2', 'v2_5')
        else Ours_LoRA_MoE)
    trainer = trainer_class(
        model, tokenizer, None, train_task_list, eval_task_list,
        test_task_list, args)
    trainer.train_continual()


if __name__ == "__main__":
    main()
