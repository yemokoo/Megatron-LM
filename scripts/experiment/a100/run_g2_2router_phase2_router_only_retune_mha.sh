#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_LOG_CHECKPOINTS="${WANDB_LOG_CHECKPOINTS:-0}"
export RETUNE_ITERS="${RETUNE_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-600}"
export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-1}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-72}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export G2_2R_ROOT="${G2_2R_ROOT:-$PROJECT_ROOT/.local/weights/a100/mha/g2-2router}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-EleutherAI/pythia-12b}"

export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-8}"
export NUM_EXPERTS="${NUM_EXPERTS:-16}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.0}"
export MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.0}"
export ATTN_LORA_RANK="${ATTN_LORA_RANK:-256}"
export ATTN_LORA_ALPHA="${ATTN_LORA_ALPHA:-$ATTN_LORA_RANK}"
export ATTN_FULL_RANK_LORA_RANK="${ATTN_FULL_RANK_LORA_RANK:-256}"
export ATTN_FULL_RANK_LORA_ALPHA="${ATTN_FULL_RANK_LORA_ALPHA:-$ATTN_FULL_RANK_LORA_RANK}"
export ATTN_FULL_RANK_LORA_TARGETS="${ATTN_FULL_RANK_LORA_TARGETS:-qkvo}"
export ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="${ATTN_FULL_RANK_LORA_ACTIVE_TARGETS:-}"
export ROUTER_MEMORY_KL_COEFF=0.0
export ROUTER_MEMORY_INTERVAL=0
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"
export ATTN_LORA_GROUPED_GEMM="${ATTN_LORA_GROUPED_GEMM:-1}"
export MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-configs/model/flame-two-router-hybrid-experts.sh}"

RUN_ID="${RUN_ID:-g2-2router-exp2-phase2-router-only-retune-wikicode-from-new-experts-new-router-rows-no-reinit-mb72-1800}"
WEIGHTS="${TRAIN_WEIGHTS:-$G2_2R_ROOT/code/phase2/$RUN_ID}"

if [ ! -f "$WEIGHTS/latest_checkpointed_iteration.txt" ]; then
    echo "[ERROR] missing Phase 2 copy: $WEIGHTS" >&2
    echo "Run: scripts/experiment/a100/prepare_g2_2router_phase2_copy.sh" >&2
    exit 1
fi

BASE_STEP=""
if [ -f "$WEIGHTS/PHASE2_SOURCE.txt" ]; then
    BASE_STEP="$(grep -E '^source_step=' "$WEIGHTS/PHASE2_SOURCE.txt" | tail -1 | cut -d= -f2- || true)"
elif [ -f "$WEIGHTS/PHASE3_SOURCE.txt" ]; then
    BASE_STEP="$(grep -E '^source_step=' "$WEIGHTS/PHASE3_SOURCE.txt" | tail -1 | cut -d= -f2- || true)"
fi
BASE_STEP="${BASE_STEP:-$(tr -d '\n\r[:space:]' < "$WEIGHTS/latest_checkpointed_iteration.txt")}"
LATEST_STEP="$(tr -d '\n\r[:space:]' < "$WEIGHTS/latest_checkpointed_iteration.txt")"
TARGET_STEP=$((BASE_STEP + RETUNE_ITERS))

if [ "$LATEST_STEP" -ge "$TARGET_STEP" ]; then
    echo "[SKIP] already at or past target: $WEIGHTS latest=$LATEST_STEP target=$TARGET_STEP"
    exit 0
fi

echo "[CONFIG] G2-2router Phase 2 router-only retune"
echo "[CONFIG] base_step=${BASE_STEP}, latest_step=${LATEST_STEP}, retune_iters=${RETUNE_ITERS}, target_step=${TARGET_STEP}"
echo "[CONFIG] data=wiki train + code train, loss=single LM loss"
echo "[CONFIG] trainable=attention router + FFN router only"
echo "[CONFIG] frozen=all experts, dense trunk, embeddings, output weights"
echo "[CONFIG] weights=${WEIGHTS}"

env \
    RUN_ID="$RUN_ID" \
    TRAIN_WEIGHTS="$WEIGHTS" \
    SOURCE_STEP="$BASE_STEP" \
    TRAIN_ITERS="$TARGET_STEP" \
    WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2-2router - exp2 phase2 router-only retune wiki+code}" \
    WANDB_RUN_ID="${WANDB_RUN_ID:-$RUN_ID}" \
    MASTER_PORT="${MASTER_PORT:-29764}" \
    MODEL_CONFIG_SCRIPT="$MODEL_CONFIG_SCRIPT" \
    bash "$SCRIPT_DIR/run_guarded_training.sh" \
        bash "$SCRIPT_DIR/phase3_router_only_retune_shared_router_hybrid_mixed_local_bf16.sh"
