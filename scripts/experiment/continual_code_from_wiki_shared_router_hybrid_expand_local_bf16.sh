#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/a100/common.sh"

export PYTHONPATH="$PROJECT_ROOT/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"

resolve_stage1_dir() {
    "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

explicit = os.environ.get('STAGE1_WEIGHTS_DIR', '').strip()
if explicit:
    print(explicit)
    raise SystemExit(0)

stage1_subdir = os.environ.get('STAGE1_SUBDIR', 'a100/wiki-shared-router-hybrid-pretrain-local')
local_weights = Path(os.environ['LOCAL_WEIGHTS']) / stage1_subdir
required_iters = int(os.environ.get('SOURCE_REQUIRED_ITERS', '1'))
best = None
best_mtime = -1.0
for candidate in local_weights.iterdir() if local_weights.exists() else []:
    tracker = candidate / 'latest_checkpointed_iteration.txt'
    metadata = candidate / 'logs' / 'run_metadata.json'
    if not candidate.is_dir() or not tracker.exists() or not metadata.exists():
        continue
    try:
        tracker_step = int(tracker.read_text(encoding='utf-8').strip())
        train_iters = int(json.loads(metadata.read_text(encoding='utf-8'))['train_iters'])
    except Exception:
        continue
    if tracker_step < required_iters or train_iters < required_iters:
        continue
    mtime = tracker.stat().st_mtime
    if mtime > best_mtime:
        best = candidate
        best_mtime = mtime
if best is None:
    raise SystemExit(
        'ERROR: could not find a completed wiki shared-router hybrid pretrain run. '
        'Set STAGE1_WEIGHTS_DIR explicitly.'
    )
print(best)
PY
}

read_stage1_train_iters() {
    "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
metadata = json.loads((Path(os.environ['STAGE1_WEIGHTS_DIR']) / 'logs' / 'run_metadata.json').read_text(encoding='utf-8'))
print(int(metadata['train_iters']))
PY
}

export RUN_ID="${RUN_ID:-code-from-wiki-shared-router-hybrid-expand-local-bf16-$(date -u +%Y%m%d-%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29574}"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_DATASET="${LOCAL_DATASET:-$LOCAL_BASE/dataset}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
export DIRECT_LOCAL_SAVE="${DIRECT_LOCAL_SAVE:-1}"

export SSD_MOUNT="${LOCAL_SSD_ROOT}/${RUN_ID}"
export SSD_CODE_TRAIN="${SSD_MOUNT}/dataset/code_train"
export SSD_ROUTER_MEMORY="${SSD_MOUNT}/dataset/router_memory"
export SSD_ROUTER_MEMORY_EVAL="${SSD_MOUNT}/dataset/router_memory_eval"
export SSD_SOURCE_WEIGHTS="${SSD_MOUNT}/source_weights"
export SSD_RESUME_WEIGHTS="${SSD_MOUNT}/resume_weights"
export SSD_TARGET_WEIGHTS="${SSD_MOUNT}/target_weights"

export NUM_LAYERS="${NUM_LAYERS:-9}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-5472}"
export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-704}"
export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-4}"
export NUM_EXPERTS="${NUM_EXPERTS:-7}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-2}"
export MOE_LAYER_FREQ="${MOE_LAYER_FREQ:-[0,1,1,1,1,1,1,1,1]}"
export ATTN_LORA_RANK="${ATTN_LORA_RANK:-16}"
export ATTN_LORA_ALPHA="${ATTN_LORA_ALPHA:-16}"
export ATTN_FULL_RANK_LORA_RANK="${ATTN_FULL_RANK_LORA_RANK:-0}"
export ATTN_FULL_RANK_LORA_ALPHA="${ATTN_FULL_RANK_LORA_ALPHA:-1.0}"
export ATTN_FULL_RANK_LORA_TARGETS="${ATTN_FULL_RANK_LORA_TARGETS:-qkvo}"
export ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="${ATTN_FULL_RANK_LORA_ACTIVE_TARGETS:-}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=1
export TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-local}"
export PRECISION="bf16"

export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-300}"
export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-1}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export TRAIN_LOG_STEP_TIME_ONLY="${TRAIN_LOG_STEP_TIME_ONLY:-1}"
export LR="${LR:-3e-4}"
export MIN_LR="${MIN_LR:-3e-5}"
export LR_DECAY_STYLE="${LR_DECAY_STYLE:-WSD}"
export LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.01}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-$TRAIN_ITERS}"
export LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-$((TRAIN_ITERS / 10))}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-configs/model/flame-shared-router-hybrid-experts.sh}"
export DATASET_SPLIT="${DATASET_SPLIT:-100,0,0}"

export TRAIN_DATASET="${TRAIN_DATASET:-$(dataset_dir_for_task code)}"
export ROUTER_MEMORY_KL_COEFF="${ROUTER_MEMORY_KL_COEFF:-0.0}"
export ROUTER_MEMORY_FORCE_ENABLE_ZERO_COEFF="${ROUTER_MEMORY_FORCE_ENABLE_ZERO_COEFF:-0}"
export ROUTER_MEMORY_FRACTION="${ROUTER_MEMORY_FRACTION:-0.05}"
export ROUTER_MEMORY_INTERVAL="${ROUTER_MEMORY_INTERVAL:-0}"
export ROUTER_MEMORY_DATASET="${ROUTER_MEMORY_DATASET:-$PROJECT_ROOT/data/wiki/router_memory_5pct}"
export ROUTER_MEMORY_EVAL_DATASET="${ROUTER_MEMORY_EVAL_DATASET:-$ROUTER_MEMORY_DATASET}"
export ROUTER_MEMORY_EVAL_INTERVAL="${ROUTER_MEMORY_EVAL_INTERVAL:-0}"
export ROUTER_MEMORY_EVAL_ITERS="${ROUTER_MEMORY_EVAL_ITERS:-1}"
export ROUTER_MEMORY_TEACHER_STUDENT_KL="${ROUTER_MEMORY_TEACHER_STUDENT_KL:-0}"
export ROUTER_MEMORY_TEACHER_STUDENT_KL_EXISTING_EXPERTS_ONLY="${ROUTER_MEMORY_TEACHER_STUDENT_KL_EXISTING_EXPERTS_ONLY:-0}"
export ROUTER_MEMORY_JOINT_UPDATE="${ROUTER_MEMORY_JOINT_UPDATE:-0}"
export ROUTER_KL_STOP_STEP="${ROUTER_KL_STOP_STEP:-}"
export ROUTER_KL_EARLY_STOP_ENABLED="${ROUTER_KL_EARLY_STOP_ENABLED:-0}"
export ROUTER_KL_EARLY_STOP_METRIC="${ROUTER_KL_EARLY_STOP_METRIC:-fixed_probe_kl}"
export ROUTER_KL_PATIENCE="${ROUTER_KL_PATIENCE:-3}"
export ROUTER_KL_MIN_DELTA="${ROUTER_KL_MIN_DELTA:-0.01}"
export ROUTER_KL_WARMUP_STEPS="${ROUTER_KL_WARMUP_STEPS:-300}"
export ROUTER_KL_SMOOTHING_WINDOW="${ROUTER_KL_SMOOTHING_WINDOW:-3}"
export SHARED_ROUTER_TRAIN_MASK_EXISTING_EXPERTS="${SHARED_ROUTER_TRAIN_MASK_EXISTING_EXPERTS:-0}"
export SHARED_ROUTER_TRAIN_MASK_EXISTING_EXPERTS_FROM_NUM_EXPERTS="${SHARED_ROUTER_TRAIN_MASK_EXISTING_EXPERTS_FROM_NUM_EXPERTS:-$SOURCE_NUM_EXPERTS}"
export SHARED_ROUTER_HYBRID_TRAIN_ALL_EXPERTS_AND_ROUTER="${SHARED_ROUTER_HYBRID_TRAIN_ALL_EXPERTS_AND_ROUTER:-0}"
export SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS="${SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS:-0}"
export SHARED_ROUTER_HYBRID_PARTIAL_FREEZE_MASK="${SHARED_ROUTER_HYBRID_PARTIAL_FREEZE_MASK:-}"
export SHARED_ROUTER_HYBRID_TOPK_WITH_ALL_NEW_EXPERTS="${SHARED_ROUTER_HYBRID_TOPK_WITH_ALL_NEW_EXPERTS:-0}"
export SHARED_ROUTER_HYBRID_ALL_NEW_EXPERTS_FROM_NUM_EXPERTS="${SHARED_ROUTER_HYBRID_ALL_NEW_EXPERTS_FROM_NUM_EXPERTS:-$SOURCE_NUM_EXPERTS}"
export STAGE1_WEIGHTS_DIR="${STAGE1_WEIGHTS_DIR:-}"
export STAGE1_SUBDIR="${STAGE1_SUBDIR:-a100/wiki-shared-router-hybrid-pretrain-local}"
export SOURCE_REQUIRED_ITERS="${SOURCE_REQUIRED_ITERS:-1}"
export RESUME_FROM_WEIGHTS="${RESUME_FROM_WEIGHTS:-}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$LOCAL_WEIGHTS/a100/code-from-wiki-shared-router-hybrid-expand-local/$RUN_ID}"
export LOG_DIR="${LOG_DIR:-$TRAIN_WEIGHTS/logs}"
export RUN_METADATA="${RUN_METADATA:-$LOG_DIR/run_metadata.json}"
export RUN_LOG="${RUN_LOG:-$LOG_DIR/run.log}"
export GPU_LOG="${GPU_LOG:-$LOG_DIR/gpu.log}"
export DATASET_NAME="${DATASET_NAME:-code_exact}"
export DATASET_SOURCE="${DATASET_SOURCE:-Python code exact train}"
export PROBE_DATASET="${PROBE_DATASET:-$(probe_dir_for_task code)}"
export PROBE_NAME="${PROBE_NAME:-code_probe}"
export PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-25}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-100}"
export SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET:-$(probe_dir_for_task wiki)}"
export SECONDARY_PROBE_NAME="${SECONDARY_PROBE_NAME:-wiki_probe}"
export SECONDARY_PROBE_EVAL_ITERS="${SECONDARY_PROBE_EVAL_ITERS:-25}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-100}"
export TERTIARY_PROBE_DATASET="${TERTIARY_PROBE_DATASET:-}"
export TERTIARY_PROBE_NAME="${TERTIARY_PROBE_NAME:-}"
export TERTIARY_PROBE_EVAL_ITERS="${TERTIARY_PROBE_EVAL_ITERS:-25}"
export TERTIARY_PROBE_EVAL_INTERVAL="${TERTIARY_PROBE_EVAL_INTERVAL:-0}"
export PROBE_STEP_OFFSET="${PROBE_STEP_OFFSET:-}"
export SECONDARY_PROBE_STEP_OFFSET="${SECONDARY_PROBE_STEP_OFFSET:-}"
export TERTIARY_PROBE_STEP_OFFSET="${TERTIARY_PROBE_STEP_OFFSET:-}"
export RUN_INITIAL_PROBE_EVAL="${RUN_INITIAL_PROBE_EVAL:-1}"
export RUN_INITIAL_VALID_EVAL="${RUN_INITIAL_VALID_EVAL:-1}"
export WANDB_STEP_OFFSET="${WANDB_STEP_OFFSET:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-$RUN_ID}"
export WANDB_SAVE_DIR="${WANDB_SAVE_DIR:-$TRAIN_WEIGHTS/wandb}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$RUN_ID}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_DISABLE_CONFIG="${WANDB_DISABLE_CONFIG:-1}"
export WANDB_LOG_CHECKPOINTS="${WANDB_LOG_CHECKPOINTS:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

router_memory_requested() {
    [ "$ROUTER_MEMORY_FORCE_ENABLE_ZERO_COEFF" = "1" ] || {
        [ "$ROUTER_MEMORY_KL_COEFF" != "0" ] && [ "$ROUTER_MEMORY_KL_COEFF" != "0.0" ]
    }
}

export STAGE1_WEIGHTS_DIR="$(resolve_stage1_dir)"
export PROBE_STEP_OFFSET="${PROBE_STEP_OFFSET:-$(read_stage1_train_iters)}"
export SECONDARY_PROBE_STEP_OFFSET="${SECONDARY_PROBE_STEP_OFFSET:-$PROBE_STEP_OFFSET}"
export TERTIARY_PROBE_STEP_OFFSET="${TERTIARY_PROBE_STEP_OFFSET:-$PROBE_STEP_OFFSET}"
export WANDB_STEP_OFFSET="${WANDB_STEP_OFFSET:-$PROBE_STEP_OFFSET}"

if [ "$DIRECT_LOCAL_SAVE" = "1" ]; then
    export SSD_TARGET_WEIGHTS="$TRAIN_WEIGHTS"
fi

mkdir -p \
    "$SSD_CODE_TRAIN" \
    "$SSD_ROUTER_MEMORY" \
    "$SSD_ROUTER_MEMORY_EVAL" \
    "$SSD_SOURCE_WEIGHTS" \
    "$SSD_RESUME_WEIGHTS" \
    "$SSD_TARGET_WEIGHTS" \
    "$TRAIN_WEIGHTS" \
    "$LOG_DIR"
exec > >(
    tee -a "$RUN_LOG" | "$PYTHON_BIN" -u -c '
from datetime import datetime, timedelta
import re, sys
iter_re = re.compile(r"(\[[^]]+\]) iteration\s+(\d+)/\s*(\d+)")
ms_re = re.compile(r"elapsed time per iteration \(ms\):\s*([0-9.]+)")
loss_re = re.compile(r"lm loss:\s*([0-9.Ee+-]+)")
tflops_re = re.compile(r"throughput per GPU \(TFLOP/s/GPU\):\s*([0-9.]+)")
val_re = re.compile(r"validation loss at iteration\s+(\d+).*lm loss value:\s*([^|]+)")
save_re = re.compile(r"saving checkpoint at iteration\s+(\d+)")
keep_re = re.compile(r"shared-router hybrid|router memory|Router-memory|ERROR:|Traceback|failed \(exitcode|checkpoint at|probe |Expanded MoE checkpoint")

def parse_timestamp(value):
    try:
        return datetime.strptime(value.strip("[]"), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None

def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"

for line in sys.stdin:
    line = line.rstrip("\n")
    m = iter_re.search(line)
    if m:
        step = int(m.group(2))
        total = int(m.group(3))
        parts = [f"{m.group(1)} step {step}/{total}"]
        ms = ms_re.search(line)
        loss = loss_re.search(line)
        tflops = tflops_re.search(line)
        if ms:
            seconds_per_iter = float(ms.group(1)) / 1000.0
            remaining_seconds = max(0, total - step) * seconds_per_iter
            parts.append(f"{seconds_per_iter:.1f}s/it")
            if step < total:
                parts.append(f"ETA {format_duration(remaining_seconds)}")
                ts = parse_timestamp(m.group(1))
                if ts is not None:
                    end_at = ts + timedelta(seconds=remaining_seconds)
                    parts.append(f"end {end_at:%Y-%m-%d %H:%M:%S}")
        if loss:
            parts.append(f"lm loss {loss.group(1)}")
        if tflops:
            parts.append(f"GPU {tflops.group(1)} TFLOP/s")
        print(" | ".join(parts), flush=True)
        continue
    m = val_re.search(line)
    if m:
        print(f"validation step {m.group(1)} | lm loss {m.group(2).strip()}", flush=True)
        continue
    m = save_re.search(line)
    if m:
        print(f"saving checkpoint step {m.group(1)}", flush=True)
        continue
    if keep_re.search(line):
        print(line, flush=True)
'
) 2>&1

echo "shared-router hybrid continual run log: $RUN_LOG"
echo "shared-router hybrid continual gpu log: $GPU_LOG"
echo "shared-router hybrid continual metadata: $RUN_METADATA"
rsync -rlptD \
    --exclude 'logs/' \
    --exclude 'wandb/' \
    --exclude 'events.out.tfevents*' \
    --exclude 'progress.txt' \
    "$STAGE1_WEIGHTS_DIR/" "$SSD_SOURCE_WEIGHTS/"
if [ -n "$RESUME_FROM_WEIGHTS" ]; then
    if [ ! -f "$RESUME_FROM_WEIGHTS/latest_checkpointed_iteration.txt" ]; then
        echo "ERROR: resume checkpoint tracker not found: $RESUME_FROM_WEIGHTS/latest_checkpointed_iteration.txt" >&2
        exit 1
    fi
    rsync -rlptD \
        --exclude 'logs/' \
        --exclude 'wandb/' \
        --exclude 'events.out.tfevents*' \
        --exclude 'progress.txt' \
        "$RESUME_FROM_WEIGHTS/" "$SSD_RESUME_WEIGHTS/"
fi
rsync -rlptD --info=progress2 "$TRAIN_DATASET/" "$SSD_CODE_TRAIN/"
if router_memory_requested; then
    if ! compgen -G "$ROUTER_MEMORY_DATASET/*.bin" >/dev/null; then
        echo "ERROR: fixed router-memory dataset not found: $ROUTER_MEMORY_DATASET" >&2
        echo "Create it once with scripts/dataset/materialize_fixed_sample_stream.py." >&2
        exit 1
    fi
    rsync -rlptD --info=progress2 "$ROUTER_MEMORY_DATASET/" "$SSD_ROUTER_MEMORY/"
    if [ "$ROUTER_MEMORY_TEACHER_STUDENT_KL" != "1" ]; then
        if ! compgen -G "$ROUTER_MEMORY_EVAL_DATASET/*.bin" >/dev/null; then
            echo "ERROR: fixed router-memory eval dataset not found: $ROUTER_MEMORY_EVAL_DATASET" >&2
            echo "Create it once with scripts/dataset/materialize_fixed_sample_stream.py." >&2
            exit 1
        fi
        rsync -rlptD --info=progress2 "$ROUTER_MEMORY_EVAL_DATASET/" "$SSD_ROUTER_MEMORY_EVAL/"
    fi
fi

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
from megatron.core.datasets import indexed_dataset

dataset_dir = Path(os.environ['SSD_CODE_TRAIN'])
total_tokens = 0
total_documents = 0
shards = []
for idx_path in sorted(dataset_dir.glob('*.idx')):
    prefix = idx_path.with_suffix('')
    ds = indexed_dataset.IndexedDataset(str(prefix), multimodal=False, mmap=True)
    shard_tokens = int(ds.sequence_lengths.sum())
    shard_docs = int(ds.document_indices.shape[0] - 1)
    total_tokens += shard_tokens
    total_documents += shard_docs
    shards.append({'prefix': prefix.name, 'documents': shard_docs, 'tokens': shard_tokens})

metadata = {
    'partial_freeze_enabled': os.environ.get('SHARED_ROUTER_HYBRID_PARTIAL_FREEZE_MASK', '') != '',
    'stage': 'code_from_wiki_shared_router_hybrid_expand',
    'run_id': os.environ['RUN_ID'],
    'stage1_weights_dir': os.environ['STAGE1_WEIGHTS_DIR'],
    'resume_from_weights': os.environ.get('RESUME_FROM_WEIGHTS', ''),
    'dataset_name': os.environ['DATASET_NAME'],
    'dataset_source': os.environ['DATASET_SOURCE'],
    'train_dataset': {'path': str(dataset_dir), 'tokens': total_tokens, 'documents': total_documents, 'shards': shards},
    'train_iters': int(os.environ['TRAIN_ITERS']),
    'save_checkpoints': os.environ.get('SAVE_CHECKPOINTS', '1') == '1',
    'micro_batch_size': int(os.environ['MICRO_BATCH_SIZE']),
    'global_batch_size': int(os.environ['GLOBAL_BATCH_SIZE']),
    'num_layers': int(os.environ['NUM_LAYERS']),
    'hidden_size': int(os.environ['HIDDEN_SIZE']),
    'ffn_hidden_size': int(os.environ['FFN_HIDDEN_SIZE']),
    'num_query_groups': int(os.environ['NUM_QUERY_GROUPS']),
    'moe_ffn_hidden_size': int(os.environ['MOE_FFN_HIDDEN_SIZE']),
    'source_num_experts': int(os.environ['SOURCE_NUM_EXPERTS']),
    'target_num_experts': int(os.environ['NUM_EXPERTS']),
    'moe_router_topk': int(os.environ['MOE_ROUTER_TOPK']),
    'attn_lora_rank': int(os.environ['ATTN_LORA_RANK']),
    'attn_lora_alpha': float(os.environ['ATTN_LORA_ALPHA']),
    'attn_full_rank_lora_rank': int(os.environ.get('ATTN_FULL_RANK_LORA_RANK', '0')),
    'attn_full_rank_lora_alpha': float(os.environ.get('ATTN_FULL_RANK_LORA_ALPHA', '1.0')),
    'attn_full_rank_lora_targets': os.environ.get('ATTN_FULL_RANK_LORA_TARGETS', 'qkvo'),
    'attn_full_rank_lora_active_targets': os.environ.get('ATTN_FULL_RANK_LORA_ACTIVE_TARGETS', ''),
    'attn_lora_grouped_gemm': os.environ.get('ATTN_LORA_GROUPED_GEMM', '0') == '1',
    'shared_router_hybrid': True,
    'shared_router_train_mask_existing_experts': os.environ.get('SHARED_ROUTER_TRAIN_MASK_EXISTING_EXPERTS', '0') == '1',
    'shared_router_train_mask_existing_experts_from_num_experts': int(os.environ.get('SHARED_ROUTER_TRAIN_MASK_EXISTING_EXPERTS_FROM_NUM_EXPERTS', '0')),
    'train_new_experts_and_router_only': (
        os.environ.get('SHARED_ROUTER_HYBRID_TRAIN_ALL_EXPERTS_AND_ROUTER', '0') != '1'
        and os.environ.get('SHARED_ROUTER_HYBRID_PARTIAL_FREEZE_MASK', '') == ''
    ),
    'shared_router_hybrid_train_all_experts_and_router': os.environ.get('SHARED_ROUTER_HYBRID_TRAIN_ALL_EXPERTS_AND_ROUTER', '0') == '1',
    'shared_router_hybrid_partial_freeze_mask': os.environ.get('SHARED_ROUTER_HYBRID_PARTIAL_FREEZE_MASK', ''),
    'shared_router_hybrid_train_all_router_rows': (
        os.environ.get('SHARED_ROUTER_HYBRID_TRAIN_ALL_EXPERTS_AND_ROUTER', '0') == '1'
        or os.environ.get('SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS', '0') == '1'
    ),
    'router_memory_kl_coeff': float(os.environ.get('ROUTER_MEMORY_KL_COEFF', '0.0')),
    'router_memory_force_enable_zero_coeff': os.environ.get('ROUTER_MEMORY_FORCE_ENABLE_ZERO_COEFF', '0') == '1',
    'router_memory_fraction': float(os.environ.get('ROUTER_MEMORY_FRACTION', '0.0')),
    'router_memory_interval': int(os.environ.get('ROUTER_MEMORY_INTERVAL', '0')),
    'router_memory_dataset': os.environ.get('ROUTER_MEMORY_DATASET', ''),
    'router_memory_ssd_dataset': os.environ.get('SSD_ROUTER_MEMORY', ''),
    'router_memory_eval_dataset': os.environ.get('ROUTER_MEMORY_EVAL_DATASET', ''),
    'router_memory_eval_ssd_dataset': os.environ.get('SSD_ROUTER_MEMORY_EVAL', ''),
    'router_memory_eval_interval': int(os.environ.get('ROUTER_MEMORY_EVAL_INTERVAL', '0')),
    'router_memory_eval_iters': int(os.environ.get('ROUTER_MEMORY_EVAL_ITERS', '1')),
    'router_memory_teacher_student_kl': os.environ.get('ROUTER_MEMORY_TEACHER_STUDENT_KL', '0') == '1',
    'router_memory_teacher_student_kl_existing_experts_only': os.environ.get('ROUTER_MEMORY_TEACHER_STUDENT_KL_EXISTING_EXPERTS_ONLY', '0') == '1',
    'router_memory_joint_update': os.environ.get('ROUTER_MEMORY_JOINT_UPDATE', '0') == '1',
    'shared_router_hybrid_topk_with_all_new_experts': os.environ.get('SHARED_ROUTER_HYBRID_TOPK_WITH_ALL_NEW_EXPERTS', '0') == '1',
    'shared_router_hybrid_all_new_experts_from_num_experts': int(os.environ.get('SHARED_ROUTER_HYBRID_ALL_NEW_EXPERTS_FROM_NUM_EXPERTS', '0')),
    'wandb_log_checkpoints': os.environ.get('WANDB_LOG_CHECKPOINTS', '0') == '1',
    'router_kl_stop_step': os.environ.get('ROUTER_KL_STOP_STEP', ''),
    'router_kl_early_stop_enabled': os.environ.get('ROUTER_KL_EARLY_STOP_ENABLED', '0') == '1',
    'router_kl_early_stop_metric': os.environ.get('ROUTER_KL_EARLY_STOP_METRIC', 'fixed_probe_kl'),
    'router_kl_patience': int(os.environ.get('ROUTER_KL_PATIENCE', '3')),
    'router_kl_min_delta': float(os.environ.get('ROUTER_KL_MIN_DELTA', '0.01')),
    'router_kl_warmup_steps': int(os.environ.get('ROUTER_KL_WARMUP_STEPS', '300')),
    'router_kl_smoothing_window': int(os.environ.get('ROUTER_KL_SMOOTHING_WINDOW', '3')),
}
with open(os.environ['RUN_METADATA'], 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
PY

source "$MODEL_CONFIG_SCRIPT"

WANDB_ARGS=()
if [ -n "$WANDB_PROJECT" ]; then
    WANDB_ARGS+=(
        --wandb-project "$WANDB_PROJECT"
        --wandb-exp-name "$WANDB_EXP_NAME"
        --wandb-save-dir "$WANDB_SAVE_DIR"
        --wandb-step-offset "$WANDB_STEP_OFFSET"
        --wandb-run-id "$WANDB_RUN_ID"
        --wandb-resume "$WANDB_RESUME"
    )
    if [ "$WANDB_LOG_CHECKPOINTS" = "1" ]; then
        WANDB_ARGS+=(--wandb-log-checkpoints)
    fi
fi

LOG_STYLE_ARGS=()
if [ "$TRAIN_LOG_STEP_TIME_ONLY" = "1" ]; then
    LOG_STYLE_ARGS+=(--train-log-step-time-only)
fi

SAVE_ARGS=(
    --save "$SSD_TARGET_WEIGHTS"
    --save-interval "$SAVE_INTERVAL"
)
if [ "$SAVE_CHECKPOINTS" != "1" ]; then
    SAVE_ARGS+=(--skip-train-end-save)
fi

SHARED_ROUTER_ARGS=()
LOAD_WEIGHTS="$SSD_SOURCE_WEIGHTS"
SHARED_ROUTER_MODE_ARGS=(
    --shared-router-hybrid-expand-from-num-experts "$SOURCE_NUM_EXPERTS"
)
CHECKPOINT_LOAD_ARGS=(
    --no-load-optim
    --no-load-rng
    --finetune
)
if [ -n "$RESUME_FROM_WEIGHTS" ]; then
    LOAD_WEIGHTS="$SSD_RESUME_WEIGHTS"
    SHARED_ROUTER_MODE_ARGS=(
        --shared-router-hybrid-resume-from-num-experts "$SOURCE_NUM_EXPERTS"
    )
    CHECKPOINT_LOAD_ARGS=()
fi
if [ -n "$SHARED_ROUTER_HYBRID_PARTIAL_FREEZE_MASK" ]; then
    SHARED_ROUTER_ARGS+=(
        --shared-router-hybrid-partial-freeze-mask "$SHARED_ROUTER_HYBRID_PARTIAL_FREEZE_MASK"
    )
elif [ "$SHARED_ROUTER_HYBRID_TRAIN_ALL_EXPERTS_AND_ROUTER" = "1" ]; then
    SHARED_ROUTER_ARGS+=(--shared-router-hybrid-train-all-experts-and-router-only)
else
    SHARED_ROUTER_ARGS+=(--shared-router-hybrid-train-new-experts-and-router-only)
fi
if [ -z "$SHARED_ROUTER_HYBRID_PARTIAL_FREEZE_MASK" ] && [ "$SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS" = "1" ]; then
    SHARED_ROUTER_ARGS+=(--shared-router-hybrid-train-all-router-rows)
fi

INITIAL_VALID_ARGS=()
if [ "$RUN_INITIAL_VALID_EVAL" = "1" ]; then
    INITIAL_VALID_ARGS+=(--run-initial-valid-eval)
fi

INITIAL_PROBE_ARGS=()
if [ "$RUN_INITIAL_PROBE_EVAL" = "1" ]; then
    INITIAL_PROBE_ARGS+=(--run-initial-probe-eval)
fi

TERTIARY_PROBE_ARGS=()
if [ -n "$TERTIARY_PROBE_DATASET" ]; then
    TERTIARY_PROBE_ARGS+=(
        --tertiary-probe-name "$TERTIARY_PROBE_NAME"
        --tertiary-probe-eval-iters "$TERTIARY_PROBE_EVAL_ITERS"
        --tertiary-probe-eval-interval "$TERTIARY_PROBE_EVAL_INTERVAL"
        --tertiary-probe-step-offset "$TERTIARY_PROBE_STEP_OFFSET"
        --tertiary-probe-data-path $(build_data_path "$TERTIARY_PROBE_DATASET")
    )
fi

ROUTER_MEMORY_ARGS=()
if router_memory_requested; then
    ROUTER_MEMORY_ARGS+=(
        --router-memory-kl-coeff "$ROUTER_MEMORY_KL_COEFF"
        --router-memory-fraction "$ROUTER_MEMORY_FRACTION"
        --router-memory-interval "$ROUTER_MEMORY_INTERVAL"
        --router-memory-data-path $(build_data_path "$SSD_ROUTER_MEMORY")
        --router-memory-eval-data-path $(build_data_path "$SSD_ROUTER_MEMORY_EVAL")
        --router-memory-eval-interval "$ROUTER_MEMORY_EVAL_INTERVAL"
        --router-memory-eval-iters "$ROUTER_MEMORY_EVAL_ITERS"
        --router-kl-early-stop-metric "$ROUTER_KL_EARLY_STOP_METRIC"
        --router-kl-patience "$ROUTER_KL_PATIENCE"
        --router-kl-min-delta "$ROUTER_KL_MIN_DELTA"
        --router-kl-warmup-steps "$ROUTER_KL_WARMUP_STEPS"
        --router-kl-smoothing-window "$ROUTER_KL_SMOOTHING_WINDOW"
    )
    if [ -n "$ROUTER_KL_STOP_STEP" ]; then
        ROUTER_MEMORY_ARGS+=(--router-kl-stop-step "$ROUTER_KL_STOP_STEP")
    fi
    if [ "$ROUTER_KL_EARLY_STOP_ENABLED" = "1" ]; then
        ROUTER_MEMORY_ARGS+=(--router-kl-early-stop-enabled)
    fi
    if [ "$ROUTER_MEMORY_FORCE_ENABLE_ZERO_COEFF" = "1" ]; then
        ROUTER_MEMORY_ARGS+=(--router-memory-force-enable-zero-coeff)
    fi
    if [ "$ROUTER_MEMORY_TEACHER_STUDENT_KL" = "1" ]; then
        ROUTER_MEMORY_ARGS+=(--router-memory-teacher-student-kl)
        if [ -n "$RESUME_FROM_WEIGHTS" ]; then
            ROUTER_MEMORY_ARGS+=(--router-memory-teacher-load "$SSD_SOURCE_WEIGHTS")
        fi
    fi
    if [ "$ROUTER_MEMORY_TEACHER_STUDENT_KL_EXISTING_EXPERTS_ONLY" = "1" ]; then
        ROUTER_MEMORY_ARGS+=(--router-memory-teacher-student-kl-existing-experts-only)
    fi
    if [ "$ROUTER_MEMORY_JOINT_UPDATE" = "1" ]; then
        ROUTER_MEMORY_ARGS+=(--router-memory-joint-update)
    fi
fi

torchrun \
    --nproc_per_node "$NPROC_PER_NODE" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    Megatron-LM/pretrain_gpt.py \
    "${MODEL_ARGS[@]}" \
    --transformer-impl "$TRANSFORMER_IMPL" \
    --pipeline-model-parallel-size "$PIPELINE_MODEL_PARALLEL_SIZE" \
    --expert-model-parallel-size "$EXPERT_MODEL_PARALLEL_SIZE" \
    --distributed-timeout-minutes 30 \
    --no-persist-layer-norm \
    --bf16 \
    --micro-batch-size "$MICRO_BATCH_SIZE" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --lr "$LR" \
    --min-lr "$MIN_LR" \
    --lr-decay-style "$LR_DECAY_STYLE" \
    --lr-decay-iters "$LR_DECAY_ITERS" \
    --lr-warmup-fraction "$LR_WARMUP_FRACTION" \
    --lr-wsd-decay-iters "$LR_WSD_DECAY_ITERS" \
    --train-iters "$TRAIN_ITERS" \
    "${SHARED_ROUTER_MODE_ARGS[@]}" \
    "${SHARED_ROUTER_ARGS[@]}" \
    --seq-length "${SEQ_LENGTH:-512}" \
    --data-path $(build_data_path "$SSD_CODE_TRAIN") \
    --split "$DATASET_SPLIT" \
    --log-interval "$LOG_INTERVAL" \
    --log-throughput \
    --log-progress \
    "${LOG_STYLE_ARGS[@]}" \
    "${SAVE_ARGS[@]}" \
    --load "$LOAD_WEIGHTS" \
    --eval-interval "$EVAL_INTERVAL" \
    --tensorboard-dir "$SSD_TARGET_WEIGHTS" \
    "${CHECKPOINT_LOAD_ARGS[@]}" \
    "${INITIAL_VALID_ARGS[@]}" \
    --probe-name "$PROBE_NAME" \
    --probe-eval-iters "$PROBE_EVAL_ITERS" \
    --probe-eval-interval "$PROBE_EVAL_INTERVAL" \
    --probe-step-offset "$PROBE_STEP_OFFSET" \
    --probe-data-path $(build_data_path "$PROBE_DATASET") \
    "${INITIAL_PROBE_ARGS[@]}" \
    --secondary-probe-name "$SECONDARY_PROBE_NAME" \
    --secondary-probe-eval-iters "$SECONDARY_PROBE_EVAL_ITERS" \
    --secondary-probe-eval-interval "$SECONDARY_PROBE_EVAL_INTERVAL" \
    --secondary-probe-step-offset "$SECONDARY_PROBE_STEP_OFFSET" \
    --secondary-probe-data-path $(build_data_path "$SECONDARY_PROBE_DATASET") \
    "${TERTIARY_PROBE_ARGS[@]}" \
    "${ROUTER_MEMORY_ARGS[@]}" \
    "${WANDB_ARGS[@]}"

if [ "$SSD_TARGET_WEIGHTS" != "$TRAIN_WEIGHTS" ]; then
    rsync -rlptD "$SSD_TARGET_WEIGHTS/" "$TRAIN_WEIGHTS/"
fi
