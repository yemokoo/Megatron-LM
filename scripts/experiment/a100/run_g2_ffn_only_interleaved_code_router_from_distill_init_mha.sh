#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
A100_SCRIPTS_DIR="$SCRIPT_DIR"
source "$SCRIPT_DIR/common.sh"

MODE="${1:-logits}"
if [ "$MODE" != "logits" ]; then
    echo "usage: $0 [logits]" >&2
    exit 1
fi

export SOURCE_TASK=wiki
export TARGET_TASK=code
export FREEZE_SHARED=1
export TRAIN_ATTENTION_WITH_NEW_EXPERTS=0
export FREEZE_DENSE_ATTENTION_LORA_WITH_NEW_EXPERTS=1
export TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY=1
export LOAD_EXPANDED_SOURCE=1

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export G2_ROOT="${G2_ROOT:-$LOCAL_WEIGHTS/a100/mha/g2-checkpoints}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

export DISTILL_SOURCE_MB="${DISTILL_SOURCE_MB:-48}"
export DISTILL_SOURCE_ITERS="${DISTILL_SOURCE_ITERS:-1800}"
export DISTILL_SOURCE_RUN_ID="${DISTILL_SOURCE_RUN_ID:-g2-ffn-only-e8to16-code-expert-init-logits-wiki-distill-mha-a100-bf16-mb${DISTILL_SOURCE_MB}-${DISTILL_SOURCE_ITERS}}"
export SOURCE_WEIGHTS_DIR="${SOURCE_WEIGHTS_DIR:-$G2_ROOT/code/expansion_distill_init/$DISTILL_SOURCE_RUN_ID}"
export SOURCE_RUN_ID="$DISTILL_SOURCE_RUN_ID"
export SOURCE_RUN_SUBDIR="a100/mha/g2-checkpoints/code/expansion_distill_init"
export SOURCE_REQUIRED_ITERS="$DISTILL_SOURCE_ITERS"

if [ ! -f "$SOURCE_WEIGHTS_DIR/latest_checkpointed_iteration.txt" ]; then
    echo "[ERROR] logits wiki-distill checkpoint not found: $SOURCE_WEIGHTS_DIR" >&2
    exit 1
fi
source_step="$(tr -d '\n\r[:space:]' < "$SOURCE_WEIGHTS_DIR/latest_checkpointed_iteration.txt")"
if [ "$source_step" != "$DISTILL_SOURCE_ITERS" ]; then
    echo "[ERROR] expected source iteration $DISTILL_SOURCE_ITERS, got $source_step: $SOURCE_WEIGHTS_DIR" >&2
    exit 1
fi

export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-8}"
export NUM_EXPERTS="${NUM_EXPERTS:-16}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-scripts/experiment/a100/flame-moe-bf16-no-shared.sh}"

export MOE_INTERLEAVE_CODE_TOTAL_STEPS="${MOE_INTERLEAVE_CODE_TOTAL_STEPS:-1800}"
export MOE_INTERLEAVE_CODE_STEPS="${MOE_INTERLEAVE_CODE_STEPS:-50}"
export MOE_INTERLEAVE_ROUTER_STEPS="${MOE_INTERLEAVE_ROUTER_STEPS:-50}"
export MOE_INTERLEAVE_ROUTER_AFTER_FINAL="${MOE_INTERLEAVE_ROUTER_AFTER_FINAL:-1}"

if [ "$MOE_INTERLEAVE_CODE_TOTAL_STEPS" -le 0 ] ||
   [ "$MOE_INTERLEAVE_CODE_STEPS" -le 0 ] ||
   [ "$MOE_INTERLEAVE_ROUTER_STEPS" -le 0 ]; then
    echo "[ERROR] interleave step counts must all be positive" >&2
    exit 1
fi

num_code_chunks=$(((MOE_INTERLEAVE_CODE_TOTAL_STEPS + MOE_INTERLEAVE_CODE_STEPS - 1) / MOE_INTERLEAVE_CODE_STEPS))
num_router_chunks="$num_code_chunks"
if [ "$MOE_INTERLEAVE_ROUTER_AFTER_FINAL" != "1" ]; then
    num_router_chunks=$((num_router_chunks - 1))
fi
export TRAIN_ITERS=$((MOE_INTERLEAVE_CODE_TOTAL_STEPS + num_router_chunks * MOE_INTERLEAVE_ROUTER_STEPS))

export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-72}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
batch_unit=$((MICRO_BATCH_SIZE * NPROC_PER_NODE))
if [ $((GLOBAL_BATCH_SIZE % batch_unit)) -ne 0 ]; then
    echo "[ERROR] global batch $GLOBAL_BATCH_SIZE must be divisible by mb*gpus=$batch_unit" >&2
    exit 1
fi
export NUM_MICROBATCHES=$((GLOBAL_BATCH_SIZE / batch_unit))
export LR="${LR:-3e-4}"
export MIN_LR="${MIN_LR:-3e-5}"
export MOE_INTERLEAVE_CODE_LR="${MOE_INTERLEAVE_CODE_LR:-$LR}"
export MOE_INTERLEAVE_CODE_MIN_LR="${MOE_INTERLEAVE_CODE_MIN_LR:-$MIN_LR}"
export MOE_INTERLEAVE_ROUTER_LR="${MOE_INTERLEAVE_ROUTER_LR:-$LR}"
export MOE_INTERLEAVE_ROUTER_MIN_LR="${MOE_INTERLEAVE_ROUTER_MIN_LR:-$MIN_LR}"
export MOE_INTERLEAVE_CODE_AUX_LOSS_COEFF="${MOE_INTERLEAVE_CODE_AUX_LOSS_COEFF:-0.01}"
export MOE_INTERLEAVE_CODE_Z_LOSS_COEFF="${MOE_INTERLEAVE_CODE_Z_LOSS_COEFF:-0.001}"
export MOE_INTERLEAVE_ROUTER_AUX_LOSS_COEFF="${MOE_INTERLEAVE_ROUTER_AUX_LOSS_COEFF:-0.0}"
export MOE_INTERLEAVE_ROUTER_Z_LOSS_COEFF="${MOE_INTERLEAVE_ROUTER_Z_LOSS_COEFF:-0.0}"

# Keep one rolling model-only recovery checkpoint, plus the final persistent checkpoint.
export SAVE_INTERVAL="${SAVE_INTERVAL:-$TRAIN_ITERS}"
export RECOVERY_SAVE_INTERVAL="${RECOVERY_SAVE_INTERVAL:-600}"
export NO_SAVE_OPTIM=1
export EVAL_INTERVAL="${EVAL_INTERVAL:-$TRAIN_ITERS}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}"
export RUN_INITIAL_PROBE_EVAL="${RUN_INITIAL_PROBE_EVAL:-1}"
export RUN_INITIAL_VALID_EVAL="${RUN_INITIAL_VALID_EVAL:-0}"

export ENABLE_OLD_MODEL_KL=0
export OLD_MODEL_KL_COEFF=0.0
export MOE_EXPANSION_DISTILL_MODE=none
export LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND=0

export STAGE_DIR_NAME="a100/mha/g2-checkpoints/code/interleaved_router_finetune_distill_init"
export RUN_ID="${RUN_ID:-g2-ffn-only-interleaved-code${MOE_INTERLEAVE_CODE_STEPS}-router${MOE_INTERLEAVE_ROUTER_STEPS}-from-logits-wiki-distill-mb${MICRO_BATCH_SIZE}-code${MOE_INTERLEAVE_CODE_TOTAL_STEPS}}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$G2_ROOT/code/interleaved_router_finetune_distill_init/$RUN_ID}"
export RUN_LOG="${RUN_LOG:-$TRAIN_WEIGHTS/logs/run.log}"
export DIRECT_LOCAL_SAVE="${DIRECT_LOCAL_SAVE:-1}"
export RESUME_CONTINUAL_FROM_TRAIN_WEIGHTS="${RESUME_CONTINUAL_FROM_TRAIN_WEIGHTS:-1}"

export PROBE_STEP_OFFSET="${PROBE_STEP_OFFSET:-$DISTILL_SOURCE_ITERS}"
export SECONDARY_PROBE_STEP_OFFSET="${SECONDARY_PROBE_STEP_OFFSET:-$DISTILL_SOURCE_ITERS}"
export WANDB_STEP_OFFSET="${WANDB_STEP_OFFSET:-$DISTILL_SOURCE_ITERS}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2 FFN-only interleaved Code/router from logits wiki-distill}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$RUN_ID}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_LOG_CHECKPOINTS=0

echo "[CONFIG] source=$SOURCE_WEIGHTS_DIR (iteration $source_step)"
echo "[CONFIG] output=$TRAIN_WEIGHTS"
echo "[CONFIG] one process: Code $MOE_INTERLEAVE_CODE_STEPS / router $MOE_INTERLEAVE_ROUTER_STEPS"
echo "[CONFIG] Code total=$MOE_INTERLEAVE_CODE_TOTAL_STEPS, total optimizer steps=$TRAIN_ITERS"
echo "[CONFIG] gpus=$CUDA_VISIBLE_DEVICES nproc=$NPROC_PER_NODE"
echo "[CONFIG] mb=$MICRO_BATCH_SIZE gbs=$GLOBAL_BATCH_SIZE microbatches=$NUM_MICROBATCHES"
echo "[CONFIG] recovery every $RECOVERY_SAVE_INTERVAL; keep one rolling + one final"
echo "[CONFIG] phase optimizers persist in memory; optimizer state is not checkpointed"

if [ "${PLAN_ONLY:-0}" = "1" ]; then
    exit 0
fi

exec bash "$A100_SCRIPTS_DIR/run_continual_moe_a100_bf16.sh"
