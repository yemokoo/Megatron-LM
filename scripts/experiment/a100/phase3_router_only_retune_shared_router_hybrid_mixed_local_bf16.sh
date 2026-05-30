#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

export PYTHONPATH="$PROJECT_ROOT/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"

export RUN_ID="${RUN_ID:-phase3-router-only-retune-shared-router-hybrid-mixed-bf16-$(date -u +%Y%m%d-%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29761}"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
export DIRECT_LOCAL_SAVE="${DIRECT_LOCAL_SAVE:-1}"

export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:?TRAIN_WEIGHTS must point to the Phase 3 checkpoint copy}"
export LOG_DIR="${LOG_DIR:-$TRAIN_WEIGHTS/logs}"
export RUN_LOG="${RUN_LOG:-$LOG_DIR/phase3_run.log}"
export GPU_LOG="${GPU_LOG:-$LOG_DIR/phase3_gpu.log}"
export RUN_METADATA="${RUN_METADATA:-$LOG_DIR/phase3_run_metadata.json}"
export TRAIN_ROUTER_USAGE_LOG_INTERVAL="${TRAIN_ROUTER_USAGE_LOG_INTERVAL:-0}"
export TRAIN_ROUTER_USAGE_LOG_PATH="${TRAIN_ROUTER_USAGE_LOG_PATH:-$LOG_DIR/train_router_usage.jsonl}"
export SHARED_ROUTER_HYBRID_REINIT_ROUTER="${SHARED_ROUTER_HYBRID_REINIT_ROUTER:-0}"

export SSD_MOUNT="${LOCAL_SSD_ROOT}/${RUN_ID}"
export SSD_WIKI_TRAIN="${SSD_MOUNT}/dataset/wiki_train"
export SSD_CODE_TRAIN="${SSD_MOUNT}/dataset/code_train"
export SSD_TARGET_WEIGHTS="${SSD_MOUNT}/target_weights"

export NUM_LAYERS="${NUM_LAYERS:-9}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-5472}"
export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-8}"
export NUM_EXPERTS="${NUM_EXPERTS:-16}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_LAYER_FREQ="${MOE_LAYER_FREQ:-[0,1,1,1,1,1,1,1,1]}"
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.0}"
export MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.0}"
export ATTN_LORA_RANK="${ATTN_LORA_RANK:-256}"
export ATTN_LORA_ALPHA="${ATTN_LORA_ALPHA:-256}"
export ATTN_FULL_RANK_LORA_RANK="${ATTN_FULL_RANK_LORA_RANK:-256}"
export ATTN_FULL_RANK_LORA_ALPHA="${ATTN_FULL_RANK_LORA_ALPHA:-256}"
export ATTN_FULL_RANK_LORA_TARGETS="${ATTN_FULL_RANK_LORA_TARGETS:-qkvo}"
export ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="${ATTN_FULL_RANK_LORA_ACTIVE_TARGETS:-}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-72}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=1
export TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-local}"
export PRECISION="bf16"

if [ ! -f "$TRAIN_WEIGHTS/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: Phase 3 checkpoint copy is missing latest_checkpointed_iteration.txt: $TRAIN_WEIGHTS" >&2
    exit 1
fi

read_phase3_source_step() {
    if [ -f "$TRAIN_WEIGHTS/PHASE3_SOURCE.txt" ]; then
        local value
        value="$(grep -E '^source_step=' "$TRAIN_WEIGHTS/PHASE3_SOURCE.txt" | tail -1 | cut -d= -f2- || true)"
        if [ -n "$value" ]; then
            echo "$value"
            return
        fi
    fi
    tr -d '\n\r[:space:]' < "$TRAIN_WEIGHTS/latest_checkpointed_iteration.txt"
}

export RETUNE_ITERS="${RETUNE_ITERS:-1800}"
export SOURCE_STEP="${SOURCE_STEP:-$(read_phase3_source_step)}"
export TRAIN_ITERS="${TRAIN_ITERS:-$((SOURCE_STEP + RETUNE_ITERS))}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-600}"
export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-1}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-$TRAIN_ITERS}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export TRAIN_LOG_STEP_TIME_ONLY="${TRAIN_LOG_STEP_TIME_ONLY:-1}"
export LR="${LR:-3e-4}"
export MIN_LR="${MIN_LR:-3e-5}"
export LR_DECAY_STYLE="${LR_DECAY_STYLE:-WSD}"
export LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.0}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-$TRAIN_ITERS}"
export LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-$RETUNE_ITERS}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-configs/model/flame-shared-router-hybrid-experts.sh}"
export DATASET_SPLIT="${DATASET_SPLIT:-100,0,0}"

export TRAIN_DATASET_WIKI="${TRAIN_DATASET_WIKI:-$(dataset_dir_for_task wiki)}"
export TRAIN_DATASET_CODE="${TRAIN_DATASET_CODE:-$(dataset_dir_for_task code)}"
export DATASET_NAME="${DATASET_NAME:-wiki_code_mixed_exact}"
export DATASET_SOURCE="${DATASET_SOURCE:-Wikipedia exact train + Python code exact train}"
export PROBE_DATASET="${PROBE_DATASET:-$(probe_dir_for_task code)}"
export PROBE_NAME="${PROBE_NAME:-code_probe}"
export PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-25}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET:-$(probe_dir_for_task wiki)}"
export SECONDARY_PROBE_NAME="${SECONDARY_PROBE_NAME:-wiki_probe}"
export SECONDARY_PROBE_EVAL_ITERS="${SECONDARY_PROBE_EVAL_ITERS:-25}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}"
export PROBE_STEP_OFFSET="${PROBE_STEP_OFFSET:-0}"
export SECONDARY_PROBE_STEP_OFFSET="${SECONDARY_PROBE_STEP_OFFSET:-0}"
export RUN_INITIAL_PROBE_EVAL="${RUN_INITIAL_PROBE_EVAL:-1}"
export RUN_INITIAL_VALID_EVAL="${RUN_INITIAL_VALID_EVAL:-1}"

export WANDB_PROJECT="${WANDB_PROJECT:-}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-$RUN_ID}"
export WANDB_SAVE_DIR="${WANDB_SAVE_DIR:-$TRAIN_WEIGHTS/wandb}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$RUN_ID}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_STEP_OFFSET="${WANDB_STEP_OFFSET:-0}"
export WANDB_LOG_CHECKPOINTS="${WANDB_LOG_CHECKPOINTS:-0}"

export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"
export ATTN_LORA_GROUPED_GEMM="${ATTN_LORA_GROUPED_GEMM:-1}"
export MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

if [ "$DIRECT_LOCAL_SAVE" = "1" ]; then
    export SSD_TARGET_WEIGHTS="$TRAIN_WEIGHTS"
fi

mkdir -p "$SSD_WIKI_TRAIN" "$SSD_CODE_TRAIN" "$SSD_TARGET_WEIGHTS" "$TRAIN_WEIGHTS" "$LOG_DIR"
exec > >(
    tee -a "$RUN_LOG" | "$PYTHON_BIN" -u -c '
import datetime as dt
import re, sys
iter_re = re.compile(r"(\[[^]]+\]) iteration\s+(\d+)/\s*(\d+)")
ms_re = re.compile(r"elapsed time per iteration \(ms\):\s*([0-9.]+)")
tflop_re = re.compile(r"throughput per GPU \(TFLOP/s/GPU\):\s*([0-9.]+)")
loss_re = re.compile(r"\blm loss:\s*([0-9.E+-]+)")
val_re = re.compile(r"validation loss at iteration\s+(\d+).*lm loss value:\s*([^|]+)")
save_re = re.compile(r"saving checkpoint at iteration\s+(\d+)")
keep_re = re.compile(r"phase3|shared-router hybrid|router-only|ERROR:|Traceback|failed \(exitcode|checkpoint at|probe ")
recent_ms = []
for line in sys.stdin:
    line = line.rstrip("\n")
    m = iter_re.search(line)
    if m:
        step = int(m.group(2))
        total = int(m.group(3))
        parts = [f"{m.group(1)} step {step}/{total}"]
        ms = ms_re.search(line)
        if ms:
            ms_value = float(ms.group(1))
            recent_ms.append(ms_value)
            recent_ms[:] = recent_ms[-5:]
            avg_ms = sum(recent_ms) / len(recent_ms)
            eta = dt.datetime.now() + dt.timedelta(seconds=max(0, total - step) * avg_ms / 1000.0)
            parts.append(f"{ms_value:.1f} ms/iter")
            parts.append(f"avg5 {avg_ms:.1f} ms/iter")
            parts.append(f"eta {eta:%Y-%m-%d %H:%M:%S}")
        loss = loss_re.search(line)
        if loss:
            parts.append(f"lm loss {loss.group(1)}")
        tflop = tflop_re.search(line)
        if tflop:
            parts.append(f"GPU {tflop.group(1)} TFLOP/s")
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

echo "phase3 router-only mixed run log: $RUN_LOG"
echo "phase3 router-only mixed gpu log: $GPU_LOG"
echo "phase3 router-only mixed metadata: $RUN_METADATA"
rsync -rlptD --info=progress2 "$TRAIN_DATASET_WIKI/" "$SSD_WIKI_TRAIN/"
rsync -rlptD --info=progress2 "$TRAIN_DATASET_CODE/" "$SSD_CODE_TRAIN/"

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
from megatron.core.datasets import indexed_dataset

def summarize(dataset_dir: Path):
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
    return {'path': str(dataset_dir), 'tokens': total_tokens, 'documents': total_documents, 'shards': shards}

wiki = summarize(Path(os.environ['SSD_WIKI_TRAIN']))
code = summarize(Path(os.environ['SSD_CODE_TRAIN']))
metadata = {
    'stage': 'phase3_router_only_retune_shared_router_hybrid_mixed',
    'run_id': os.environ['RUN_ID'],
    'train_weights': os.environ['TRAIN_WEIGHTS'],
    'source_step': int(os.environ['SOURCE_STEP']),
    'retune_iters': int(os.environ['RETUNE_ITERS']),
    'target_iteration': int(os.environ['TRAIN_ITERS']),
    'dataset_name': os.environ['DATASET_NAME'],
    'dataset_source': os.environ['DATASET_SOURCE'],
    'wiki_train': wiki,
    'code_train': code,
    'combined_tokens': wiki['tokens'] + code['tokens'],
    'trainable': 'shared-router weights only',
    'frozen': 'all FFN experts, attention LoRA experts, dense trunk, embeddings, and output weights',
    'loss': 'standard final language-modeling loss',
    'shared_router_hybrid_reinit_router': os.environ['SHARED_ROUTER_HYBRID_REINIT_ROUTER'] == '1',
    'train_router_usage_log_interval': int(os.environ['TRAIN_ROUTER_USAGE_LOG_INTERVAL']),
    'train_router_usage_log_path': os.environ['TRAIN_ROUTER_USAGE_LOG_PATH'],
    'moe_aux_loss_coeff': float(os.environ['MOE_AUX_LOSS_COEFF']),
    'moe_z_loss_coeff': float(os.environ['MOE_Z_LOSS_COEFF']),
    'micro_batch_size': int(os.environ['MICRO_BATCH_SIZE']),
    'global_batch_size': int(os.environ['GLOBAL_BATCH_SIZE']),
    'num_layers': int(os.environ['NUM_LAYERS']),
    'hidden_size': int(os.environ['HIDDEN_SIZE']),
    'ffn_hidden_size': int(os.environ['FFN_HIDDEN_SIZE']),
    'moe_ffn_hidden_size': int(os.environ['MOE_FFN_HIDDEN_SIZE']),
    'source_num_experts': int(os.environ['SOURCE_NUM_EXPERTS']),
    'target_num_experts': int(os.environ['NUM_EXPERTS']),
    'moe_router_topk': int(os.environ['MOE_ROUTER_TOPK']),
    'attn_lora_rank': int(os.environ['ATTN_LORA_RANK']),
    'attn_full_rank_lora_rank': int(os.environ['ATTN_FULL_RANK_LORA_RANK']),
    'attn_full_rank_lora_targets': os.environ['ATTN_FULL_RANK_LORA_TARGETS'],
    'precision': os.environ['PRECISION'],
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

REINIT_ROUTER_ARGS=()
if [ "$SHARED_ROUTER_HYBRID_REINIT_ROUTER" = "1" ]; then
    REINIT_ROUTER_ARGS+=(--shared-router-hybrid-reinit-router)
fi

SAVE_ARGS=(
    --save "$SSD_TARGET_WEIGHTS"
    --save-interval "$SAVE_INTERVAL"
)
if [ "$SAVE_CHECKPOINTS" != "1" ]; then
    SAVE_ARGS+=(--skip-train-end-save)
fi

INITIAL_VALID_ARGS=()
if [ "$RUN_INITIAL_VALID_EVAL" = "1" ]; then
    INITIAL_VALID_ARGS+=(--run-initial-valid-eval)
fi

INITIAL_PROBE_ARGS=()
if [ "$RUN_INITIAL_PROBE_EVAL" = "1" ]; then
    INITIAL_PROBE_ARGS+=(--run-initial-probe-eval)
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
    --shared-router-hybrid-resume-from-num-experts "$SOURCE_NUM_EXPERTS" \
    --shared-router-hybrid-train-router-only \
    "${REINIT_ROUTER_ARGS[@]}" \
    --train-router-usage-log-interval "$TRAIN_ROUTER_USAGE_LOG_INTERVAL" \
    --train-router-usage-log-path "$TRAIN_ROUTER_USAGE_LOG_PATH" \
    --train-router-usage-num-existing-experts "$SOURCE_NUM_EXPERTS" \
    --seq-length "${SEQ_LENGTH:-512}" \
    --data-path $(build_data_path "$SSD_WIKI_TRAIN" "$SSD_CODE_TRAIN") \
    --split "$DATASET_SPLIT" \
    --log-interval "$LOG_INTERVAL" \
    --log-throughput \
    --log-progress \
    "${LOG_STYLE_ARGS[@]}" \
    "${SAVE_ARGS[@]}" \
    --load "$TRAIN_WEIGHTS" \
    --no-load-optim \
    --no-load-rng \
    --eval-interval "$EVAL_INTERVAL" \
    --tensorboard-dir "$SSD_TARGET_WEIGHTS" \
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
    "${WANDB_ARGS[@]}"

if [ "$SSD_TARGET_WEIGHTS" != "$TRAIN_WEIGHTS" ]; then
    rsync -rlptD "$SSD_TARGET_WEIGHTS/" "$TRAIN_WEIGHTS/"
fi
