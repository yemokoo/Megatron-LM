#!/usr/bin/env python
# Adapted from TRACE (BeyonderXX/TRACE, Apache-2.0) training/main.py.
# Stripped to the local growing LoRA-MoE methods: FFN-only V1/V2 and the V3
# shared-router QKVO+FFN extension. All other CL_method branches,
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
import model.Ours_LoRA_MoE as ours_lora_moe_module
from model.Ours_LoRA_MoE import (Ours_LoRA_MoE, Ours_LoRA_MoE_V2,
                                 Ours_LoRA_MoE_V2_New,
                                 Ours_LoRA_MoE_V1_Expert_First,
                                 attach_lora_moe,
                                 add_experts_to_all_layers)
from model.Ours_LoRA_MoE_V3 import (
    V3_ARCHITECTURE,
    Ours_LoRA_MoE_V3,
    Ours_LoRA_MoE_V3_New,
    add_v3_experts,
    attach_shared_qkvo_lora_moe,
)
from utils.chat_templates import (
    ensure_llama31_chat_template,
    update_fingerprint_for_chat_template,
)

AllDatasetName = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
                  "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]


def list_of_strings(arg):
    return arg.split(',')


def list_of_ints(arg):
    return [int(x) for x in arg.split(',')]


def list_of_floats(arg):
    if not arg:
        return []
    return [float(x) for x in arg.split(',')]


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
    tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
    if os.path.isfile(tokenizer_config_path):
        with open(tokenizer_config_path, encoding="utf-8") as handle:
            update_fingerprint_for_chat_template(digest, json.load(handle))
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
    parser.add_argument('--v3_epoch_probe_samples', type=int, default=64,
                        help='Fixed distributed validation samples per V3 epoch; '
                             '64 is one global batch. 0 disables the probe.')
    parser.add_argument(
        '--v2_acquisition_diagnostic_interval', type=int, default=0,
        help='V2/V2-new only: write rank-0-local KD/new-task routing and '
             'gradient diagnostics every N optimizer updates. 0 disables it.')
    parser.add_argument(
        '--v2_new_expert_quota_schedule', type=list_of_floats, default=[],
        help='Opt-in V2-new top-1 treatment: comma-separated per-primary-epoch '
             'minimum valid-token shares for the newly-added expert. Positive '
             'epochs use separate natural router-only and quota expert-only '
             'new-data branches; zero epochs retain the original V2 path.')
    parser.add_argument(
        '--v2_new_expert_aux_mix', type=float, default=0.0,
        help='Opt-in V2-new top-1 auxiliary acquisition branch. On the same '
             'new-task batch, interpolate this fraction of the newly-added '
             'expert into tokens that natural routing assigned elsewhere. '
             'The ordinary V2-new branch is unchanged and the auxiliary '
             'branch freezes router parameters. 0 disables it exactly.')
    parser.add_argument(
        '--v2_new_expert_aux_loss_coeff', type=float, default=1.0,
        help='Multiplier for the expert-only auxiliary LM loss. The adapter '
             'gradient is already scaled by --v2_new_expert_aux_mix.')
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="constant_with_warmup")
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0,
                        help='Per-phase optimizer-step warmup ratio; overrides num_warmup_steps when positive.')
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--resume_checkpoint", default="",
        help="Completed numeric round checkpoint to resume after. Restores all "
             "grown experts/router and starts at the following TRACE task.")
    parser.add_argument(
        "--stop_after_task", default="",
        help="Optional task name at which continual training exits after saving "
             "that task checkpoint. Empty runs every remaining task.")
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
    parser.add_argument('--training_version',
                        choices=['v1', 'v1_expert_first', 'v2', 'v2_new',
                                 'v2_new_top4', 'v2_5', 'v3', 'v3_new',
                                 'v3_new_top4', 'v3_new_replay40',
                                 'v3_new_hidden_mse_full',
                                 'v3_new_replay1to1',
                                 'v3_new_hidden_mse_1to1',
                                 'v3_new_replay1to1_recency',
                                 'v3_new_replay1to1_p5k',
                                 'v3_new_hidden_mse_1to1_p5k',
                                 'v3_new_kd35k',
                                 'v3_new_recency_kd175k',
                                 'v3_new_p5k_kd175k',
                                 'v3_new_r20_kd100',
                                 'v3_new_kd200',
                                 'v3_new_recency_p2',
                                 'v3_new_hmse_kd200'],
                        default='v1',
                        help='v1: sequential new-task then router retune; v2: '
                             'KD-init plus joint new-task/router-replay updates; '
                             'v2_new: v2 with persistent 10%% task memories, a '
                             'nested-prefix active-memory cap, and one active '
                             'memory pass per primary epoch; '
                             'v2_new_top4: the same V2-new memory/training '
                             'contract with four rank-16 experts added per '
                             'task and normalized top-4 dispatch; '
                             'v1_expert_first: V2-new KD init, then one '
                             'standalone full-token expert-only phase, then '
                             'a separate V2-new-memory router-only retune; '
                             'v2_5: v2 with router aux/z losses disabled; v3: '
                             'v2 mechanics with one pre-attention router shared '
                             'by equal-rank QKVO and FFN LoRA experts; v3_new: '
                             'the same QKVO+FFN architecture with the strict '
                             'V2-new persistent/active-memory schedule; '
                             'v3_new_top4: the same V3-new architecture and '
                             'schedule with four rank-16 experts per task and '
                             'normalized top-4 dispatch.')
    # --- ablation switches (V3 only; defaults reproduce the published arm) ---
    parser.add_argument(
        '--ablation_phase_mode', choices=['1phase', '2phase'],
        default='1phase',
        help='1phase: new-task LM and router replay share every optimizer '
             'update (published Ours). 2phase: train the new expert plus its '
             'single new router row, checkpoint, then retune the whole router '
             'on the same replay stream. Both arms see the same replay '
             'exposures; only the timing differs.')
    parser.add_argument(
        '--ablation_kd_init', choices=['on', 'off'], default='on',
        help='on: run the expansion KD-init pass before each task (published '
             'Ours). off: skip it, so the new expert starts from its random '
             'init. Recorded in the checkpoint meta and provable from the '
             'absence of the kd_init role in training_workload.json.')
    parser.add_argument(
        '--ablation_replay_source', choices=['real', 'selfgen'],
        default='real',
        help='Declares which replay memory this run is supposed to use. '
             'real refuses to start when SELFGEN_ROOT is set, selfgen refuses '
             'to start unless scripts/selfgen/train_selfgen.py has patched '
             'the replay source -- the failure mode this guards is a selfgen '
             'arm silently training on real data.')
    parser.add_argument(
        '--ablation_phase2_memory', choices=['past_only', 'include_current'],
        default='past_only',
        help='2phase only. past_only keeps the phase-2 replay set identical '
             'to what the 1-phase arm replays (strictly past tasks).')
    parser.add_argument('--experts_per_task', type=int, default=4,
                        help='New FFN LoRA experts added per task; V3 adds the '
                             'same number of QKVO experts at the same time.')
    parser.add_argument('--lora_moe_rank', type=int, default=8)
    parser.add_argument('--lora_moe_alpha', type=int, default=32)
    parser.add_argument('--lora_moe_dropout', type=float, default=0.0)
    parser.add_argument('--top_k', type=int, default=2,
                        help='Experts activated per token (fixed for a run; vary across runs for ablation).')
    parser.add_argument('--routing_weight_mode', type=str, default=None,
                        choices=['full_softmax', 'topk_softmax',
                                 'straight_through_topk'],
                        help='full_softmax keeps an LM-loss router gradient even at top-k=1; '
                             'topk_softmax is the legacy selected-set normalization. '
                             'Defaults to straight_through_topk for v2_new and '
                             'full_softmax for every existing version.')
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
    parser.add_argument('--replay_subset_ratio', type=float, default=None,
                        help='Unique fixed subset stored per 5,000-sample task. '
                             'Defaults to 0.1 (500) for v2_new and 0.01 (50) '
                             'for every existing training version.')
    parser.add_argument('--replay_recency_power', type=float, default=1.0,
                        help='recency_weighted only: task at recency rank k gets weight '
                             'k**power. 1.0 is a linear ramp, larger concentrates on '
                             'the newest task, 0.0 degenerates to equal_task.')
    parser.add_argument('--replay_distribution',
                        choices=['equal_task', 'proportional', 'recency_weighted'],
                        default='equal_task')
    parser.add_argument('--replay_subset_seed', type=int, default=-1,
                        help='Fixed per-task subset seed; -1 reuses --seed.')
    parser.add_argument(
        '--replay_selection_mode',
        choices=['random', 'router_gradient'], default='random',
        help='How the persistent per-task replay subset is selected. '
             'router_gradient scores examples during the ordinary V3 primary '
             'backward and keeps the largest mean per-sample router-gradient '
             'norm; it does not add an extra backward pass.')

    # --- v2 fixed-memory KD + exact-budget joint replay ---
    parser.add_argument('--v2_memory_batch_size', type=int, default=0,
                        help='Legacy replay stream/KD fallback batch size. '
                             'Exact replay keeps this at 0 or 1; use '
                             '--v2_replay_forward_batch_size to pack records '
                             'without changing sample weighting.')
    parser.add_argument(
        '--v2_replay_forward_batch_size', type=int, default=8,
        help='Maximum replay records per rank packed into one backbone '
             'forward. Each record retains its original batch-size-one '
             'token-mean CE, so exposure and active-sample-mean gradients '
             'are unchanged.')
    parser.add_argument('--v2_kd_memory_batch_size', type=int, default=0,
                        help='KD-only microbatch size per rank; 0 falls back to '
                             '--v2_memory_batch_size (and then one sample per rank).')
    parser.add_argument(
        '--allow_v2_memory_batch_resume_override', action='store_true',
        help='Allow changing only operational V2 KD/replay memory '
             'microbatch fields on resume; exposure, loss, and optimizer '
             'update contracts remain strict.')
    parser.add_argument('--v2_max_replay_batches_per_step', type=int, default=0,
                        help='Safety cap on global replay samples assigned to '
                             'one optimizer update; 0 is unlimited.')
    parser.add_argument('--v2_joint_replay_loss_coeff', type=float, default=1.0)
    parser.add_argument(
        '--v2_joint_replay_objective', choices=['lm', 'hidden_mse'],
        default='lm',
        help='Past-data objective in the joint router-only branch. hidden_mse '
             'matches every decoder-layer output to an expanded post-KD-init '
             'frozen teacher while leaving new-task LM unchanged.')
    parser.add_argument(
        '--v2_hidden_mse_loss_coeff', type=float, default=1.0,
        help='Coefficient for sample-mean, layer-mean hidden MSE replay.')
    parser.add_argument('--v2_joint_new_to_replay_ratio', type=int, default=None,
                        help='Set replay from a new-task dataset pass at N:1. '
                             'Legacy v2/v2_5/v3 distribute one fixed pass over '
                             'the whole phase; v2_new repeats its active stream '
                             'once per primary epoch. Defaults to 5 for v2_new '
                             'and 0 for existing versions; 0 uses '
                             '--router_replay_exposure_samples directly.')
    parser.add_argument('--v2_kd_loss_coeff', type=float, default=1.0)
    parser.add_argument(
        '--v2_kd_pass_multiplier', type=int, default=1,
        help='Repeat the same fixed KD memory stream this many times per '
             'primary epoch. This changes KD optimization steps only; it '
             'does not add unique memory records or change joint replay.')
    parser.add_argument('--v2_kd_temperature', type=float, default=1.0)
    parser.add_argument('--v2_kd_learning_rate', type=float, default=0.0,
                        help='0 reuses --learning_rate.')
    parser.add_argument('--v2_kd_chunk_tokens', type=int, default=256)
    parser.add_argument('--v2_kd_token_scope', choices=['nonpad', 'labels'], default='nonpad')
    parser.add_argument(
        '--v2_new_active_memory_cap', type=int, default=1000,
        help='v2_new only: exact aggregate past-memory stream size per primary '
             'epoch. The stream is drawn from persistent per-task memories by '
             'equal-task nested prefixes and is reused by KD and joint replay.')
    parser.add_argument(
        '--v2_kd_exposure_samples', type=int, default=0,
        help='v2_new relaxed versions only: memory records consumed per KD-init '
             'pass. 0 keeps KD tied to --v2_new_active_memory_cap, which is the '
             'historical behaviour and makes KD grow whenever replay does.')
    parser.add_argument(
        '--v2_kd_epochs', type=int, default=0,
        help='v2_new relaxed versions only: KD-init passes over the memory '
             'stream. 0 derives it from the primary epoch count (3/5/7).')
    parser.add_argument(
        '--v2_new_active_unique_cap', type=int, default=0,
        help='v2_new relaxed versions only: cap on the number of DISTINCT old '
             'records joint replay may draw from, decoupled from the exposure '
             'budget set by --v2_new_active_memory_cap. 0 keeps them tied, '
             'which is the historical behaviour. KD is unaffected.')
    parser.add_argument(
        '--v2_new_persistent_samples_per_task', type=int, default=500,
        help='v2_new only: exact persistent records selected once per 5,000 '
             'sample TRACE task. V2-new intentionally fixes this to 500.')
    parser.add_argument('--disable_training_flop_counter', action='store_true',
                        help='Disable operator FLOP counting; samples/tokens/steps/time are still recorded.')

    return parser.parse_args()


V2_NEW_TRAINING_VERSIONS = frozenset({
    "v2_new", "v2_new_top4", "v3_new", "v3_new_top4",
    "v3_new_replay40", "v3_new_hidden_mse_full", "v3_new_replay1to1",
    "v3_new_hidden_mse_1to1", "v3_new_replay1to1_recency",
    "v3_new_replay1to1_p5k", "v3_new_hidden_mse_1to1_p5k",
    "v3_new_kd35k", "v3_new_recency_kd175k", "v3_new_p5k_kd175k", "v3_new_r20_kd100",
    "v3_new_kd200", "v3_new_recency_p2", "v3_new_hmse_kd200",
})
# v3_new with a larger router-replay budget.  Architecture, persistent memory,
# KD and objectives are identical to v3_new; only the active memory stream
# consumed per primary epoch grows, which is what moves the new:replay
# exposure ratio away from 5:1.  It is deliberately excluded from
# V2_NEW_FIXED_CONTRACT_VERSIONS so the 1000-sample / 5:1 assertions that pin
# the published v3_new runs remain in force for those runs.
V2_NEW_FIXED_CONTRACT_VERSIONS = frozenset({
    "v2_new", "v2_new_top4", "v3_new", "v3_new_top4", "v1_expert_first",
})


def uses_fixed_v2_new_contract(training_version):
    """Whether the hard 1000-sample / 5:1 TRACE contract is enforced."""
    return training_version in V2_NEW_FIXED_CONTRACT_VERSIONS
V2_NEW_MEMORY_TRAINING_VERSIONS = frozenset({
    "v2_new", "v2_new_top4", "v3_new", "v3_new_top4",
    "v1_expert_first", "v3_new_replay40", "v3_new_hidden_mse_full",
    "v3_new_replay1to1", "v3_new_hidden_mse_1to1",
    "v3_new_replay1to1_recency",
})

V3_TRAINING_VERSIONS = frozenset({
    "v3", "v3_new", "v3_new_top4", "v3_new_replay40",
    "v3_new_hidden_mse_full", "v3_new_replay1to1",
    "v3_new_hidden_mse_1to1", "v3_new_replay1to1_recency",
    "v3_new_replay1to1_p5k", "v3_new_hidden_mse_1to1_p5k",
    "v3_new_kd35k", "v3_new_recency_kd175k", "v3_new_p5k_kd175k", "v3_new_r20_kd100",
    "v3_new_kd200", "v3_new_recency_p2", "v3_new_hmse_kd200",
})


def is_v2_new_training_version(training_version):
    """Return whether a profile uses the strict V2-new memory contract."""
    return training_version in V2_NEW_TRAINING_VERSIONS


def uses_v2_new_memory(training_version):
    """Return whether a profile uses V2-new persistent memory and KD data."""
    return training_version in V2_NEW_MEMORY_TRAINING_VERSIONS


def resolve_training_version_defaults(args):
    """Resolve version-dependent CLI defaults without changing legacy runs."""
    is_v2_new = is_v2_new_training_version(args.training_version)
    uses_v2_new = uses_v2_new_memory(args.training_version)
    if args.replay_subset_ratio is None:
        args.replay_subset_ratio = 0.1 if uses_v2_new else 0.01
    if args.v2_joint_new_to_replay_ratio is None:
        args.v2_joint_new_to_replay_ratio = 5 if is_v2_new else 0
    if args.routing_weight_mode is None:
        args.routing_weight_mode = (
            'straight_through_topk'
            if uses_v2_new else 'full_softmax')
    # Resolve the sentinel for reproducible selection and auditable metadata.
    # On resume, already-saved per-task indices+SHA256 are authoritative; a
    # scalar command-seed change alone does not replace or reject those files.
    if uses_v2_new and args.replay_subset_seed < 0:
        args.replay_subset_seed = args.seed
    validate_ablation_switches(args)
    return args


def ablation_contract(args):
    """The ablation switches a resumed chain must not silently change."""
    return {
        "phase_mode": getattr(args, "ablation_phase_mode", "1phase"),
        "kd_init": getattr(args, "ablation_kd_init", "on"),
        "replay_source": getattr(args, "ablation_replay_source", "real"),
        "phase2_memory": getattr(
            args, "ablation_phase2_memory", "past_only"),
    }


def validate_ablation_switches(args):
    """Refuse configurations where an ablation switch would not do what it says.

    Every check here exists because the corresponding mistake is silent: a
    selfgen arm that trains on real data, a 2-phase flag on an architecture
    that has no phase-2 path, or hidden-MSE replay whose teacher the KD-off
    arm never builds.
    """
    phase_mode = getattr(args, "ablation_phase_mode", "1phase")
    replay_source = getattr(args, "ablation_replay_source", "real")
    kd_init = getattr(args, "ablation_kd_init", "on")
    selfgen_root = os.environ.get("SELFGEN_ROOT", "").strip()
    if phase_mode == "2phase":
        if args.training_version not in V3_TRAINING_VERSIONS:
            raise ValueError(
                "--ablation_phase_mode 2phase is implemented for V3 training "
                f"versions only, got {args.training_version!r}")
        if args.v2_joint_replay_objective != "lm":
            raise ValueError(
                "--ablation_phase_mode 2phase requires "
                "--v2_joint_replay_objective lm, got "
                f"{args.v2_joint_replay_objective!r}")
    if replay_source == "selfgen" and not selfgen_root:
        raise ValueError(
            "--ablation_replay_source selfgen requires SELFGEN_ROOT and the "
            "scripts/selfgen/train_selfgen.py entry point; without them the "
            "run would train on the real replay memory instead")
    if replay_source == "real" and selfgen_root:
        raise ValueError(
            "--ablation_replay_source real refuses to run with SELFGEN_ROOT "
            f"set ({selfgen_root!r}); pass --ablation_replay_source selfgen "
            "or unset the variable")
    if kd_init == "off" and args.v2_joint_replay_objective == "hidden_mse":
        raise ValueError(
            "hidden-MSE replay snapshots its teacher right after KD-init, so "
            "--ablation_kd_init off cannot be combined with it")
    if kd_init == "off" and args.v2_kd_loss_coeff != 0:
        # One source of truth: the V2-new sampler-pass validator asserts that
        # KD consumed its passes whenever the coefficient is positive, so an
        # arm that skips KD-init must also carry coefficient 0. Forcing it
        # here keeps the flag, the loss contract, and the saved metadata from
        # telling three different stories.
        print(
            "[ablation] kd_init=off -> forcing --v2_kd_loss_coeff "
            f"{args.v2_kd_loss_coeff} -> 0", flush=True)
        args.v2_kd_loss_coeff = 0.0
    return args


def v2_new_metadata_contract(args):
    """Immutable V2-new mechanics required on resume.

    The checkpoint also records the original scalar seed for audit, but the
    saved per-task index file and its SHA256—not the current command seed—own
    replay identity after selection.
    """
    contract = {
        "persistent_subset_policy":
            "validated_output_dir_json_else_deterministic_random",
        "persistent_samples_per_task":
            args.v2_new_persistent_samples_per_task,
        "persistent_memory_integrity":
            "source_count_unique_range_sha256",
        "persistent_memory_resume_source":
            "output_dir/fixed_replay_memory",
        "selection_mode": args.replay_selection_mode,
        "active_memory_cap_unique": args.v2_new_active_memory_cap,
        "active_memory_distribution": "equal_task",
        "active_memory_selection": "stable_nested_task_prefix",
        "active_stream_samples_per_pass": args.v2_new_active_memory_cap,
        "active_stream_identity_order": "shared_between_kd_and_replay",
        "active_stream_seed_phase": "v2_new_shared_active_memory",
        "kd_stream_passes": "match_primary_epochs",
        "joint_replay_stream_passes": "match_primary_epochs",
    }
    kd_pass_multiplier = int(getattr(args, "v2_kd_pass_multiplier", 1))
    if kd_pass_multiplier != 1:
        contract["kd_stream_passes"] = "primary_epochs_times_multiplier"
        contract["kd_stream_pass_multiplier"] = kd_pass_multiplier
    return contract


def v2_new_replay_memory_contract(args):
    """Immutable replay-memory mechanics required on V2-new resume."""
    return {
        "persistent_samples_per_task":
            args.v2_new_persistent_samples_per_task,
        "persistent_selection_mode": args.replay_selection_mode,
        "active_stream_samples_per_primary_epoch":
            args.v2_new_active_memory_cap,
        "distribution": "equal_task",
        "v1_router_retune_enabled": False,
    }


def v1_expert_first_replay_memory_contract(args):
    """Persistent V2-new identities with a separate V1 router phase."""
    return {
        "persistent_samples_per_task":
            args.v2_new_persistent_samples_per_task,
        "persistent_selection_mode": args.replay_selection_mode,
        "active_stream_samples_per_primary_epoch":
            args.v2_new_active_memory_cap,
        "distribution": "equal_task",
        "v1_router_retune_enabled": args.router_retune_epochs > 0,
    }


def v2_new_v2_metadata_contract(args):
    """V2 phase metadata that is mandatory for V2-new resumes."""
    memory_batch_size = args.v2_memory_batch_size
    kd_memory_batch_size = args.v2_kd_memory_batch_size
    active_stream_samples = args.v2_new_active_memory_cap
    contract = {
        "memory_batch_size": memory_batch_size,
        "kd_memory_batch_size": kd_memory_batch_size,
        "effective_kd_memory_batch_size": (
            kd_memory_batch_size or memory_batch_size or 1),
        "effective_replay_memory_batch_size": memory_batch_size or 1,
        "replay_forward_batch_size": args.v2_replay_forward_batch_size,
        "kd_loss_coeff": args.v2_kd_loss_coeff,
        "kd_temperature": args.v2_kd_temperature,
        "kd_learning_rate": args.v2_kd_learning_rate,
        "kd_chunk_tokens": args.v2_kd_chunk_tokens,
        "kd_token_scope": args.v2_kd_token_scope,
        "joint_replay_loss_coeff": args.v2_joint_replay_loss_coeff,
        "joint_new_to_replay_sample_ratio":
            args.v2_joint_new_to_replay_ratio,
        "joint_replay_schedule":
            "every_optimizer_update_active_stream_per_primary_epoch",
        "joint_replay_reduction": "active_sample_mean",
        "joint_replay_forward_reduction":
            "packed_per_sample_token_mean_then_sum",
        "max_replay_batches_per_step":
            args.v2_max_replay_batches_per_step,
        "kd_active_stream_samples_per_pass": active_stream_samples,
        "kd_active_stream_passes": "match_primary_epochs",
        "kd_total_exposure_strategy":
            "samples_per_pass_times_primary_epochs",
        "joint_replay_active_stream_samples_per_primary_epoch":
            active_stream_samples,
        "joint_replay_total_exposure_strategy":
            "samples_per_primary_epoch_times_primary_epochs",
    }
    if getattr(args, "v2_joint_replay_objective", "lm") == "hidden_mse":
        contract.update({
            "joint_replay_objective": "hidden_mse",
            "hidden_mse_loss_coeff": float(getattr(
                args, "v2_hidden_mse_loss_coeff", 1.0)),
            "hidden_mse_teacher": "expanded_post_kd_init",
            "hidden_mse_targets": "all_decoder_layer_outputs",
            "hidden_mse_reduction":
                "active_sample_mean_equal_layer_mean",
        })
    kd_pass_multiplier = int(getattr(args, "v2_kd_pass_multiplier", 1))
    if kd_pass_multiplier != 1:
        contract["kd_active_stream_passes"] = (
            "primary_epochs_times_multiplier")
        contract["kd_pass_multiplier"] = kd_pass_multiplier
        contract["kd_total_exposure_strategy"] = (
            "samples_per_pass_times_primary_epochs_times_multiplier")
    aux_mix = float(getattr(args, "v2_new_expert_aux_mix", 0.0))
    if aux_mix > 0:
        contract.update({
            "new_expert_auxiliary_route":
                "detached_router_hard_top1_interpolation",
            "new_expert_aux_mix": aux_mix,
            "new_expert_aux_loss_coeff": float(getattr(
                args, "v2_new_expert_aux_loss_coeff", 1.0)),
            "new_expert_aux_optimizer_schedule":
                "same_primary_update_single_optimizer_step",
        })
    return contract


def v1_expert_first_v2_metadata_contract(args):
    """KD and router-integration contract for expert-first V1."""
    memory_batch_size = args.v2_memory_batch_size
    kd_memory_batch_size = args.v2_kd_memory_batch_size
    return {
        "memory_batch_size": memory_batch_size,
        "kd_memory_batch_size": kd_memory_batch_size,
        "effective_kd_memory_batch_size": (
            kd_memory_batch_size or memory_batch_size or 1),
        "kd_loss_coeff": args.v2_kd_loss_coeff,
        "kd_temperature": args.v2_kd_temperature,
        "kd_learning_rate": args.v2_kd_learning_rate,
        "kd_chunk_tokens": args.v2_kd_chunk_tokens,
        "kd_token_scope": args.v2_kd_token_scope,
        "kd_active_stream_samples_per_pass": args.v2_new_active_memory_cap,
        "kd_active_stream_passes": "primary_epochs_times_multiplier",
        "kd_pass_multiplier": args.v2_kd_pass_multiplier,
        "router_ft_schedule": "post_expert_training_router_only",
        "router_ft_seen_memory_exposures": args.v2_new_active_memory_cap,
    }


def v1_expert_first_memory_metadata_contract(args):
    """Auditable V2-new memory mechanics used by expert-first V1."""
    return {
        "persistent_subset_policy":
            "validated_output_dir_json_else_deterministic_random",
        "persistent_samples_per_task":
            args.v2_new_persistent_samples_per_task,
        "persistent_memory_integrity":
            "source_count_unique_range_sha256",
        "persistent_memory_resume_source":
            "output_dir/fixed_replay_memory",
        "selection_mode": args.replay_selection_mode,
        "active_memory_cap_unique": args.v2_new_active_memory_cap,
        "active_memory_distribution": "equal_task",
        "active_memory_selection": "stable_nested_task_prefix",
        "active_stream_samples_per_pass": args.v2_new_active_memory_cap,
        "active_stream_seed_phase": "v2_new_shared_active_memory",
        "kd_stream_passes": "primary_epochs_times_multiplier",
        "kd_stream_pass_multiplier": args.v2_kd_pass_multiplier,
        "router_ft_stream_passes": "one_seen_task_stream",
    }


def metadata_contract_mismatches(expected, actual, prefix):
    """Return checkpoint contract differences with user-facing key paths."""
    return {
        f"{prefix}.{key}": (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }


def v2_resume_metadata_mismatches(args, actual_v2, completed_round):
    """Compare V2 phase metadata without tightening legacy checkpoints."""
    if args.training_version == "v1_expert_first":
        expected = v1_expert_first_v2_metadata_contract(args)
        if getattr(args, "allow_v2_memory_batch_resume_override", False):
            for key in (
                    "memory_batch_size", "kd_memory_batch_size",
                    "effective_kd_memory_batch_size"):
                expected.pop(key, None)
        return metadata_contract_mismatches(expected, actual_v2, "v2")
    expected = {
        "memory_batch_size": args.v2_memory_batch_size,
        "kd_loss_coeff": args.v2_kd_loss_coeff,
        "kd_temperature": args.v2_kd_temperature,
        "kd_learning_rate": args.v2_kd_learning_rate,
        "kd_chunk_tokens": args.v2_kd_chunk_tokens,
        "kd_token_scope": args.v2_kd_token_scope,
        "joint_replay_loss_coeff": args.v2_joint_replay_loss_coeff,
        "max_replay_batches_per_step":
            args.v2_max_replay_batches_per_step,
    }
    if is_v2_new_training_version(args.training_version):
        # There is no legacy V2-new format, so every exposure-defining field
        # is mandatory even for round 0.
        expected = v2_new_v2_metadata_contract(args)
        if getattr(args, "allow_v2_memory_batch_resume_override", False):
            for key in (
                    "memory_batch_size", "kd_memory_batch_size",
                    "effective_kd_memory_batch_size",
                    "effective_replay_memory_batch_size",
                    "replay_forward_batch_size"):
                expected.pop(key, None)
        # Older checkpoints predate packed replay forwards.  This field is
        # operational only: absence means the historical one-record forward
        # and does not alter exposure identities or the loss contract.
        if "replay_forward_batch_size" not in actual_v2:
            expected.pop("replay_forward_batch_size", None)
            expected.pop("joint_replay_forward_reduction", None)
        return metadata_contract_mismatches(expected, actual_v2, "v2")

    mismatches = metadata_contract_mismatches(expected, actual_v2, "v2")
    optional = {
        "kd_memory_batch_size": args.v2_kd_memory_batch_size,
        "joint_new_to_replay_sample_ratio":
            args.v2_joint_new_to_replay_ratio,
        "joint_replay_reduction": "active_sample_mean",
        "joint_replay_forward_reduction":
            "packed_per_sample_token_mean_then_sum",
    }
    for key, expected_value in optional.items():
        if key in actual_v2 and actual_v2[key] != expected_value:
            mismatches[f"v2.{key}"] = (
                actual_v2[key], expected_value)
    if "joint_replay_schedule" in actual_v2:
        expected_schedule = (
            "every_optimizer_update_fixed_total_no_epoch_multiplier")
        legacy_first_task_checkpoint = (
            completed_round == 0
            and actual_v2["joint_replay_schedule"] ==
            "every_optimizer_update_exact_global_exposure")
        if (actual_v2["joint_replay_schedule"] != expected_schedule
                and not legacy_first_task_checkpoint):
            mismatches["v2.joint_replay_schedule"] = (
                actual_v2["joint_replay_schedule"], expected_schedule)
    return mismatches


def validate_v2_new_resume_persisted_identities(
        meta, completed_round, task_names):
    """Validate completed-task replay identities recorded by V2-new.

    The same compact task -> {resolved_seed, indices_sha256} mapping is stored
    in both replay-memory and V2-new metadata.  It is passed to the trainer so
    loading each authoritative JSON can cross-check the checkpoint identity.
    """
    task_names = list(task_names)
    if not 0 <= completed_round < len(task_names):
        raise ValueError(
            "V2-new resume round is outside the configured task order: "
            f"round={completed_round}, tasks={len(task_names)}")
    expected_tasks = task_names[:completed_round + 1]
    replay_identities = (meta.get("replay_memory", {})
                         .get("persisted_identities"))
    v2_new_identities = (meta.get("v2_new", {})
                         .get("persisted_identities"))

    def validate_mapping(value, location):
        if not isinstance(value, dict):
            raise ValueError(
                f"{location}.persisted_identities must be a mapping")
        if set(value) != set(expected_tasks) or len(value) != len(expected_tasks):
            raise ValueError(
                f"{location}.persisted_identities task keys mismatch: "
                f"actual={sorted(value)}, expected={sorted(expected_tasks)}")
        validated = {}
        for task in expected_tasks:
            record = value[task]
            if (not isinstance(record, dict)
                    or set(record) != {"resolved_seed", "indices_sha256"}):
                raise ValueError(
                    f"{location}.persisted_identities.{task} must contain "
                    "exactly resolved_seed and indices_sha256")
            resolved_seed = record["resolved_seed"]
            digest = record["indices_sha256"]
            if (not isinstance(resolved_seed, int)
                    or isinstance(resolved_seed, bool)):
                raise ValueError(
                    f"{location}.persisted_identities.{task}.resolved_seed "
                    "must be an integer")
            if (not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef"
                           for character in digest)):
                raise ValueError(
                    f"{location}.persisted_identities.{task}.indices_sha256 "
                    "must be a lowercase 64-character SHA256")
            validated[task] = {
                "resolved_seed": int(resolved_seed),
                "indices_sha256": digest,
            }
        return validated

    replay_validated = validate_mapping(replay_identities, "replay_memory")
    v2_new_validated = validate_mapping(v2_new_identities, "v2_new")
    if replay_validated != v2_new_validated:
        raise ValueError(
            "V2-new persisted identity metadata differs between "
            "replay_memory and v2_new")
    return replay_validated


def validate_v2_new_args(args):
    """Fail early on settings that would violate the named V2-new contract."""
    aux_mix = float(getattr(args, "v2_new_expert_aux_mix", 0.0))
    aux_loss_coeff = float(getattr(
        args, "v2_new_expert_aux_loss_coeff", 1.0))
    if not 0.0 <= aux_mix <= 1.0:
        raise ValueError("--v2_new_expert_aux_mix must be in [0, 1]")
    if aux_loss_coeff < 0.0:
        raise ValueError(
            "--v2_new_expert_aux_loss_coeff must be non-negative")
    if aux_mix > 0 and args.training_version != "v2_new":
        raise ValueError(
            "--v2_new_expert_aux_mix currently requires v2_new")
    if not uses_v2_new_memory(args.training_version):
        return
    fixed_contract = uses_fixed_v2_new_contract(args.training_version)
    if fixed_contract:
        if args.v2_new_persistent_samples_per_task != 500:
            raise ValueError(
                "v2_new requires --v2_new_persistent_samples_per_task 500")
        if args.replay_subset_ratio != 0.1:
            raise ValueError(
                "v2_new requires --replay_subset_ratio 0.1 (500/5,000)")
        if args.v2_new_active_memory_cap != 1000:
            raise ValueError(
                "v2_new requires --v2_new_active_memory_cap 1000")
    else:
        # Relaxed profile: the replay stream is sized directly.  The pool must
        # still be able to fill the active cap, so persistent-per-task times
        # the number of tasks has to reach it, and the subset ratio must match
        # the persistent count against TRACE's 5,000-sample tasks.
        if args.v2_new_persistent_samples_per_task < 1:
            raise ValueError(
                "--v2_new_persistent_samples_per_task must be positive")
        if args.v2_new_active_memory_cap < 1:
            raise ValueError("--v2_new_active_memory_cap must be positive")
        if not 0.0 < args.replay_subset_ratio <= 1.0:
            raise ValueError("--replay_subset_ratio must be in (0, 1]")
    if args.replay_selection_mode != 'random':
        raise ValueError(
            "v2_new requires --replay_selection_mode random")
    if args.replay_distribution not in ('equal_task', 'recency_weighted'):
        raise ValueError(
            "v2_new replay must be equal_task or recency_weighted")
    if (uses_fixed_v2_new_contract(args.training_version)
            and args.replay_distribution != 'equal_task'):
        raise ValueError(
            "the published v2_new contract fixes --replay_distribution "
            "equal_task; recency_weighted needs a relaxed training version")
    if (uses_fixed_v2_new_contract(args.training_version)
            and args.v2_joint_new_to_replay_ratio != 5):
        raise ValueError(
            "v2_new requires --v2_joint_new_to_replay_ratio 5")
    if args.lora_moe_rank < 1:
        raise ValueError("v2_new requires --lora_moe_rank >= 1")
    if args.experts_per_task < 1:
        raise ValueError("v2_new requires --experts_per_task >= 1")
    if args.top_k < 1:
        raise ValueError("v2_new requires --top_k >= 1")
    if args.top_k > args.experts_per_task:
        raise ValueError(
            "v2_new requires --top_k <= --experts_per_task so round 0 "
            "has enough active experts")
    if (args.training_version == "v1_expert_first"
            and (args.experts_per_task != 1 or args.top_k != 1)):
        raise ValueError(
            "v1_expert_first requires --experts_per_task 1 --top_k 1")
    quota_schedule = list(getattr(
        args, "v2_new_expert_quota_schedule", []) or [])
    if quota_schedule:
        if args.training_version != "v2_new":
            raise ValueError(
                "--v2_new_expert_quota_schedule currently requires v2_new")
        if args.experts_per_task != 1 or args.top_k != 1:
            raise ValueError(
                "--v2_new_expert_quota_schedule requires "
                "--experts_per_task 1 --top_k 1")
        if any(value < 0.0 or value > 1.0 for value in quota_schedule):
            raise ValueError(
                "every expert quota schedule value must be in [0, 1]")
    if aux_mix > 0:
        if args.experts_per_task != 1 or args.top_k != 1:
            raise ValueError(
                "--v2_new_expert_aux_mix requires "
                "--experts_per_task 1 --top_k 1")
        if args.routing_weight_mode != "straight_through_topk":
            raise ValueError(
                "--v2_new_expert_aux_mix requires "
                "--routing_weight_mode straight_through_topk")
        if aux_loss_coeff == 0.0:
            raise ValueError(
                "positive --v2_new_expert_aux_mix requires a positive "
                "--v2_new_expert_aux_loss_coeff")
        if quota_schedule:
            raise ValueError(
                "expert auxiliary interpolation and quota routing are "
                "mutually exclusive")
    if args.training_version in {"v2_new_top4", "v3_new_top4"}:
        exact = {
            "experts_per_task": 4,
            "lora_moe_rank": 16,
            "lora_moe_alpha": 128,
            "top_k": 4,
            "routing_weight_mode": "straight_through_topk",
            "v2_kd_pass_multiplier": 2,
        }
        mismatches = {
            name: (getattr(args, name), expected)
            for name, expected in exact.items()
            if getattr(args, name) != expected
        }
        if mismatches:
            raise ValueError(
                f"{args.training_version} requires its exact 4 x rank-16 "
                "top-4 "
                f"profile (actual, expected): {mismatches}")


def main():
    args = resolve_training_version_defaults(parse_args())
    # _allocate_memory_counts is a staticmethod shared by every variant, so the
    # recency exponent is published on the module instead of threaded through it.
    ours_lora_moe_module.RECENCY_POWER[0] = float(args.replay_recency_power)
    args.v2_new_resume_persisted_identities = None
    dataset_argument = args.dataset_name
    datasets = (
        list(AllDatasetName)
        if dataset_argument == "all" or dataset_argument[0] == "all"
        else list(dataset_argument))
    if not 0.0 < args.past_task_ratio <= 1.0:
        raise ValueError("--past_task_ratio must be in (0, 1]")
    if (not args.gradient_accumulation_steps
            or any(value < 1 for value in args.gradient_accumulation_steps)
            or args.loss_log_interval < 1):
        raise ValueError("gradient accumulation and loss log interval must be positive")
    if args.v3_epoch_probe_samples < 0:
        raise ValueError("--v3_epoch_probe_samples cannot be negative")
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
    if args.training_version in (
            'v1_expert_first', 'v2', 'v2_new', 'v2_new_top4',
            'v2_5', 'v3', 'v3_new', 'v3_new_top4'):
        if (args.v2_memory_batch_size < 0
                or args.v2_replay_forward_batch_size < 1
                or args.v2_kd_memory_batch_size < 0
                or args.v2_max_replay_batches_per_step < 0):
            raise ValueError(
                "v2 replay forward batch size must be positive and other "
                "V2 batch sizes/caps cannot be negative")
        if args.v2_memory_batch_size not in (0, 1):
            raise ValueError(
                "--v2_memory_batch_size must be 0 or 1: exact-budget "
                "router replay is assigned per sample at every optimizer step")
        if args.v2_kd_temperature <= 0:
            raise ValueError("v2 KD temperature must be positive")
        if args.v2_kd_pass_multiplier < 1:
            raise ValueError("v2 KD pass multiplier must be positive")
        if (not uses_v2_new_memory(args.training_version)
                and args.v2_kd_pass_multiplier != 1):
            raise ValueError(
                "--v2_kd_pass_multiplier is supported only by V2-new "
                "profiles; legacy V2/V2.5/V3 must keep it at 1")
        if args.v2_kd_chunk_tokens < 1:
            raise ValueError("--v2_kd_chunk_tokens must be positive")
        if args.v2_kd_loss_coeff < 0 or args.v2_joint_replay_loss_coeff < 0:
            raise ValueError("v2 KD/replay loss coefficients cannot be negative")
        if args.v2_hidden_mse_loss_coeff <= 0:
            raise ValueError("v2 hidden-MSE loss coefficient must be positive")
        if (args.v2_joint_replay_objective == 'hidden_mse'
                and args.training_version not in {'v2_new', 'v3_new'}):
            raise ValueError(
                "hidden-MSE replay is currently defined for the exact "
                "V2-new/V3-new one-expert profiles only")
        if (args.v2_joint_replay_objective == 'hidden_mse'
                and args.v2_kd_loss_coeff <= 0):
            raise ValueError(
                "hidden-MSE replay requires enabled expansion KD-init so its "
                "post-KD teacher is well-defined")
        if args.v2_joint_new_to_replay_ratio < 0:
            raise ValueError("--v2_joint_new_to_replay_ratio cannot be negative")
    validate_v2_new_args(args)
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
    if (uses_v2_new_memory(args.training_version)
            and args.v2_new_active_memory_cap % world_size != 0):
        raise ValueError(
            "--v2_new_active_memory_cap must be divisible by world size for "
            "an exact non-duplicated DDP active-memory stream: "
            f"{args.v2_new_active_memory_cap} % {world_size} != 0")

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
        args.chat_template_source = ensure_llama31_chat_template(
            tokenizer, args.model_name_or_path)
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
    else:
        args.chat_template_source = None
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

    # V1/V2 keep the released FFN-only layout. V3 lifts the router to the
    # decoder layer and shares its one pre-attention decision across equal-rank
    # QKVO and FFN LoRA experts.
    if args.training_version in V3_TRAINING_VERSIONS:
        attach_shared_qkvo_lora_moe(
            model, r=args.lora_moe_rank, alpha=args.lora_moe_alpha,
            top_k=args.top_k, aux_loss_coeff=args.moe_aux_loss_coeff,
            z_loss_coeff=args.moe_z_loss_coeff,
            routing_weight_mode=args.routing_weight_mode,
            dropout=args.lora_moe_dropout)
    else:
        attach_lora_moe(
            model, r=args.lora_moe_rank, alpha=args.lora_moe_alpha,
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
        completed_round = int(round_name)
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
        if args.training_version in V3_TRAINING_VERSIONS:
            expected.update({
                "architecture": V3_ARCHITECTURE,
                "attention_rank": args.lora_moe_rank,
                "attention_targets": ["q", "k", "v", "o"],
                "router_position":
                    "post_input_layernorm_pre_self_attention",
            })
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
        if args.training_version in V3_TRAINING_VERSIONS:
            expected_training_profile["chat_template_source"] = (
                args.chat_template_source)
        actual_training_profile = actual.get("training_profile", {})
        mismatches.update({
            f"training_profile.{key}": (actual_training_profile.get(key), value)
            for key, value in expected_training_profile.items()
            if actual_training_profile.get(key) != value
        })
        if args.training_version == "v1_expert_first":
            expected_replay = v1_expert_first_replay_memory_contract(args)
        elif is_v2_new_training_version(args.training_version):
            expected_replay = v2_new_replay_memory_contract(args)
        else:
            expected_replay = {
                "subset_ratio_per_task": args.replay_subset_ratio,
                "exposure_samples_per_round":
                    args.router_replay_exposure_samples,
                "v1_router_retune_enabled": (
                    False if args.training_version in V3_TRAINING_VERSIONS
                    else args.router_retune_epochs > 0),
                "distribution": args.replay_distribution,
                "subset_seed": args.replay_subset_seed,
            }
        actual_replay = actual.get("replay_memory", {})
        mismatches.update(metadata_contract_mismatches(
            expected_replay, actual_replay, "replay_memory"))
        actual_ablation = actual.get("ablation")
        if args.training_version in V3_TRAINING_VERSIONS and actual_ablation:
            # A chained arm (selfgen rounds resume one task at a time) must
            # keep every ablation switch fixed for the whole run. Checkpoints
            # written before the switches existed carry no block and are
            # resumed under the defaults they were trained with.
            mismatches.update(metadata_contract_mismatches(
                ablation_contract(args), actual_ablation, "ablation"))
        if args.training_version in (
                "v1_expert_first", "v2", "v2_new", "v2_new_top4",
                "v2_5", "v3", "v3_new", "v3_new_top4"):
            actual_v2 = actual.get("v2", {})
            mismatches.update(v2_resume_metadata_mismatches(
                args, actual_v2, completed_round))
            if uses_v2_new_memory(args.training_version):
                expected_v2_new = (
                    v1_expert_first_memory_metadata_contract(args)
                    if args.training_version == "v1_expert_first"
                    else v2_new_metadata_contract(args))
                actual_v2_new = actual.get("v2_new", {})
                mismatches.update(metadata_contract_mismatches(
                    expected_v2_new, actual_v2_new, "v2_new"))
                args.v2_new_resume_persisted_identities = (
                    validate_v2_new_resume_persisted_identities(
                        actual, completed_round, datasets))
        # The contract keeps a resume from silently changing replay mechanics.  A
        # deliberate change -- re-sharding a run onto a different card count, where DDP
        # needs the active-memory cap divisible by the world size -- can waive named keys
        # through RESUME_CONTRACT_ALLOW_DRIFT; unnamed keys still abort.  An entry matches
        # either the full dotted key or its final component.
        allow_drift = {k.strip() for k in os.environ.get(
            "RESUME_CONTRACT_ALLOW_DRIFT", "").split(",") if k.strip()}
        if allow_drift and mismatches:
            waived = {k: v for k, v in mismatches.items()
                      if k in allow_drift or k.rsplit(".", 1)[-1] in allow_drift}
            if waived:
                print(f"[resume] waiving contract drift: {waived}", flush=True)
                mismatches = {k: v for k, v in mismatches.items()
                              if k not in waived}
        if mismatches:
            raise ValueError(f"resume hyperparameter mismatch: {mismatches}")
        expected_experts = (completed_round + 1) * args.experts_per_task
        if int(meta["num_experts"]) != expected_experts:
            raise ValueError(
                f"round {completed_round} should have {expected_experts} experts, "
                f"checkpoint has {meta['num_experts']}")
        if args.training_version in V3_TRAINING_VERSIONS:
            add_v3_experts(model, expected_experts)
        else:
            add_experts_to_all_layers(model, expected_experts)
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if args.training_version in V3_TRAINING_VERSIONS:
            grown_fragments = (
                ".shared_expert_router.router.",
                ".self_attn.q_proj.experts.",
                ".self_attn.k_proj.experts.",
                ".self_attn.v_proj.experts.",
                ".self_attn.o_proj.experts.",
                ".mlp.experts.",
            )
        else:
            grown_fragments = (".mlp.experts.", ".mlp.router.")
        grown_missing = [key for key in missing
                         if any(fragment in key
                                for fragment in grown_fragments)]
        if unexpected or grown_missing:
            raise RuntimeError(
                f"invalid resume state: unexpected={unexpected[:5]} "
                f"grown_missing={grown_missing[:5]}")
        args.start_task = completed_round + 1
        print_rank_0(
            f"Resumed Track1 after round {completed_round}: "
            f"experts={expected_experts}, next_task={args.start_task}",
            args.global_rank)

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
            slora_max_length = (args.max_train_len or (
                args.max_prompt_len + args.max_ans_len))
            eval_data_collator = SLoRATraceDataCollator(
                tokenizer, max_length=slora_max_length,
                label_scope="answer")
            if args.use_pretokenized_train_cache:
                data_collator = PreTokenizedSLoRATraceDataCollator(tokenizer)
            else:
                data_collator = SLoRATraceDataCollator(
                    tokenizer, max_length=slora_max_length,
                    label_scope="full")
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

    # NOTE: no engine is built here. Each task's expert-growth helper creates
    # brand-new nn.Parameters (new expert A/B, a new (bigger) router
    # Linear) -- a DDP wrapper/optimizer built once, up front, would never see
    # or update them. Ours_LoRA_MoE instead builds a fresh optimizer and DDP
    # wrapper itself (_reinit_engine) at the start of every phase.
    print_rank_0(
        f"***** Running LoRA-MoE continual training ({args.training_version}) *****",
        args.global_rank)
    if args.training_version in {'v3_new', 'v3_new_top4', 'v3_new_replay40',
                                 'v3_new_hidden_mse_full',
                                 'v3_new_replay1to1',
                                 'v3_new_hidden_mse_1to1',
                                 'v3_new_replay1to1_recency',
                                 'v3_new_replay1to1_p5k',
                                 'v3_new_hidden_mse_1to1_p5k',
                                 'v3_new_kd35k',
                                 'v3_new_recency_kd175k',
                                 'v3_new_p5k_kd175k',
                                 'v3_new_r20_kd100',
                                 'v3_new_kd200',
                                 'v3_new_recency_p2',
                                 'v3_new_hmse_kd200'}:
        trainer_class = Ours_LoRA_MoE_V3_New
    elif args.training_version == 'v3':
        trainer_class = Ours_LoRA_MoE_V3
    elif args.training_version == 'v1_expert_first':
        trainer_class = Ours_LoRA_MoE_V1_Expert_First
    elif is_v2_new_training_version(args.training_version):
        trainer_class = Ours_LoRA_MoE_V2_New
    elif args.training_version in ('v2', 'v2_5'):
        trainer_class = Ours_LoRA_MoE_V2
    else:
        trainer_class = Ours_LoRA_MoE
    trainer = trainer_class(
        model, tokenizer, None, train_task_list, eval_task_list,
        test_task_list, args)
    trainer.train_continual()


if __name__ == "__main__":
    main()
