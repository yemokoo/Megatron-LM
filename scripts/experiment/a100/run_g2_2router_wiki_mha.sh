#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-600}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-$TRAIN_ITERS}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export USE_GUARD="${USE_GUARD:-1}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-72}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export BASE_STAGE_DIR="${BASE_STAGE_DIR:-$PROJECT_ROOT/.local/weights/a100/mha/g2-2router}"

export RUN_ID="${RUN_ID:-g2-2router-top4-e8-ffn352-r256-wiki-independent-router-qkvo-mha-a100-bf16-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$BASE_STAGE_DIR/wiki/$RUN_ID}"

export NUM_EXPERTS="${NUM_EXPERTS:-8}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export ATTN_LORA_RANK="${ATTN_LORA_RANK:-256}"
export ATTN_LORA_ALPHA="${ATTN_LORA_ALPHA:-$ATTN_LORA_RANK}"
export ATTN_FULL_RANK_LORA_RANK="${ATTN_FULL_RANK_LORA_RANK:-256}"
export ATTN_FULL_RANK_LORA_ALPHA="${ATTN_FULL_RANK_LORA_ALPHA:-$ATTN_FULL_RANK_LORA_RANK}"
export ATTN_FULL_RANK_LORA_TARGETS="${ATTN_FULL_RANK_LORA_TARGETS:-qkvo}"
export ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="${ATTN_FULL_RANK_LORA_ACTIVE_TARGETS:-}"
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"
export ATTN_LORA_GROUPED_GEMM="${ATTN_LORA_GROUPED_GEMM:-1}"
export MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-configs/model/flame-two-router-hybrid-experts.sh}"

is_completed() {
    local run_dir="$1"
    local latest_file="${run_dir}/latest_checkpointed_iteration.txt"
    [ -f "$latest_file" ] && [ "$(tr -d '\n\r[:space:]' < "$latest_file")" = "$TRAIN_ITERS" ]
}

if is_completed "$TRAIN_WEIGHTS"; then
    echo "[SKIP] already completed: $RUN_ID"
    exit 0
fi

echo "[CONFIG] G2-2router wiki training"
echo "[CONFIG] steps=${TRAIN_ITERS}, mb=${MICRO_BATCH_SIZE}, gbs=${GLOBAL_BATCH_SIZE}"
echo "[CONFIG] routers: attention router from LN_attn(x), FFN router from LN_ffn(h)"
echo "[CONFIG] weights=${TRAIN_WEIGHTS}"

env \
    WANDB_MODE="$WANDB_MODE" \
    DIRECT_LOCAL_SAVE=1 \
    RUN_ID="$RUN_ID" \
    TRAIN_WEIGHTS="$TRAIN_WEIGHTS" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2-2router wiki - independent attn/ffn routers}" \
    MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" \
    GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
    TRAIN_ITERS="$TRAIN_ITERS" \
    SAVE_INTERVAL="$SAVE_INTERVAL" \
    EVAL_INTERVAL="$EVAL_INTERVAL" \
    LOG_INTERVAL="$LOG_INTERVAL" \
    NUM_EXPERTS="$NUM_EXPERTS" \
    MOE_ROUTER_TOPK="$MOE_ROUTER_TOPK" \
    MOE_FFN_HIDDEN_SIZE="$MOE_FFN_HIDDEN_SIZE" \
    ATTN_LORA_RANK="$ATTN_LORA_RANK" \
    ATTN_LORA_ALPHA="$ATTN_LORA_ALPHA" \
    ATTN_FULL_RANK_LORA_RANK="$ATTN_FULL_RANK_LORA_RANK" \
    ATTN_FULL_RANK_LORA_ALPHA="$ATTN_FULL_RANK_LORA_ALPHA" \
    ATTN_FULL_RANK_LORA_TARGETS="$ATTN_FULL_RANK_LORA_TARGETS" \
    ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="$ATTN_FULL_RANK_LORA_ACTIVE_TARGETS" \
    MOE_GROUPED_GEMM="$MOE_GROUPED_GEMM" \
    ATTN_LORA_GROUPED_GEMM="$ATTN_LORA_GROUPED_GEMM" \
    MOE_ROUTER_DTYPE="$MOE_ROUTER_DTYPE" \
    MODEL_CONFIG_SCRIPT="$MODEL_CONFIG_SCRIPT" \
    MASTER_PORT="${MASTER_PORT:-29751}" \
    USE_GUARD="$USE_GUARD" \
    SCRIPT_DIR="$SCRIPT_DIR" \
    bash -lc 'if [ "$USE_GUARD" = "1" ]; then exec "$SCRIPT_DIR/run_guarded_training.sh" bash "$SCRIPT_DIR/pretrain_wiki_shared_router_hybrid_local_bf16.sh"; else exec bash "$SCRIPT_DIR/pretrain_wiki_shared_router_hybrid_local_bf16.sh"; fi'

echo "[DONE] G2-2router wiki training $(date)"
