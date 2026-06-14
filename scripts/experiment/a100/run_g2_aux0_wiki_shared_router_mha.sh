#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_LOG_CHECKPOINTS="${WANDB_LOG_CHECKPOINTS:-0}"
export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-$TRAIN_ITERS}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-$TRAIN_ITERS}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-72}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export BASE_STAGE_DIR="${BASE_STAGE_DIR:-$PROJECT_ROOT/.local/weights/a100/mha/g2-checkpoints}"

export RUN_ID="${RUN_ID:-g2-aux0-top4-e8-ffn352-r256-wiki-shared-router-qkvo-mha-a100-bf16-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$BASE_STAGE_DIR/wiki_aux0/$RUN_ID}"

export NUM_EXPERTS="${NUM_EXPERTS:-8}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export ATTN_LORA_RANK="${ATTN_LORA_RANK:-256}"
export ATTN_LORA_ALPHA="${ATTN_LORA_ALPHA:-$ATTN_LORA_RANK}"
export ATTN_FULL_RANK_LORA_RANK="${ATTN_FULL_RANK_LORA_RANK:-256}"
export ATTN_FULL_RANK_LORA_ALPHA="${ATTN_FULL_RANK_LORA_ALPHA:-$ATTN_FULL_RANK_LORA_RANK}"
export ATTN_FULL_RANK_LORA_TARGETS="${ATTN_FULL_RANK_LORA_TARGETS:-qkvo}"
export ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="${ATTN_FULL_RANK_LORA_ACTIVE_TARGETS:-}"

# Experiment 3 variable control: remove only load-balancing aux loss.
# Keep z-loss at the normal G2 value to avoid conflating specialization with logit-scale drift.
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.0}"
export MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.001}"
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"
export ATTN_LORA_GROUPED_GEMM="${ATTN_LORA_GROUPED_GEMM:-1}"
export MOE_PERMUTE_FUSION="${MOE_PERMUTE_FUSION:-0}"
export MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}"

is_completed() {
    local run_dir="$1"
    local latest_file="${run_dir}/latest_checkpointed_iteration.txt"
    [ -f "$latest_file" ] && [ "$(tr -d '\n\r[:space:]' < "$latest_file")" = "$TRAIN_ITERS" ]
}

if is_completed "$TRAIN_WEIGHTS"; then
    echo "[SKIP] already completed: $RUN_ID"
    exit 0
fi

echo "[CONFIG] G2 wiki shared-router training with aux loss disabled"
echo "[CONFIG] steps=${TRAIN_ITERS}, mb=${MICRO_BATCH_SIZE}, gbs=${GLOBAL_BATCH_SIZE}"
echo "[CONFIG] experts=${NUM_EXPERTS}, topk=${MOE_ROUTER_TOPK}, ffn_hidden=${MOE_FFN_HIDDEN_SIZE}, attn_rank=${ATTN_LORA_RANK}"
echo "[CONFIG] aux/z loss=${MOE_AUX_LOSS_COEFF}/${MOE_Z_LOSS_COEFF}"
echo "[CONFIG] weights=${TRAIN_WEIGHTS}"

exec env \
    WANDB_MODE="$WANDB_MODE" \
    DIRECT_LOCAL_SAVE=1 \
    WANDB_LOG_CHECKPOINTS="$WANDB_LOG_CHECKPOINTS" \
    RUN_ID="$RUN_ID" \
    TRAIN_WEIGHTS="$TRAIN_WEIGHTS" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2 - exp3 aux0 wiki shared-router qkvo}" \
    PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}" \
    SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}" \
    MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" \
    GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
    TRAIN_ITERS="$TRAIN_ITERS" \
    SAVE_INTERVAL="$SAVE_INTERVAL" \
    EVAL_INTERVAL="$EVAL_INTERVAL" \
    LOG_INTERVAL="$LOG_INTERVAL" \
    NUM_EXPERTS="$NUM_EXPERTS" \
    MOE_ROUTER_TOPK="$MOE_ROUTER_TOPK" \
    MOE_FFN_HIDDEN_SIZE="$MOE_FFN_HIDDEN_SIZE" \
    MOE_AUX_LOSS_COEFF="$MOE_AUX_LOSS_COEFF" \
    MOE_Z_LOSS_COEFF="$MOE_Z_LOSS_COEFF" \
    ATTN_LORA_RANK="$ATTN_LORA_RANK" \
    ATTN_LORA_ALPHA="$ATTN_LORA_ALPHA" \
    ATTN_FULL_RANK_LORA_RANK="$ATTN_FULL_RANK_LORA_RANK" \
    ATTN_FULL_RANK_LORA_ALPHA="$ATTN_FULL_RANK_LORA_ALPHA" \
    ATTN_FULL_RANK_LORA_TARGETS="$ATTN_FULL_RANK_LORA_TARGETS" \
    ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="$ATTN_FULL_RANK_LORA_ACTIVE_TARGETS" \
    MOE_GROUPED_GEMM="$MOE_GROUPED_GEMM" \
    ATTN_LORA_GROUPED_GEMM="$ATTN_LORA_GROUPED_GEMM" \
    MOE_PERMUTE_FUSION="$MOE_PERMUTE_FUSION" \
    MOE_ROUTER_DTYPE="$MOE_ROUTER_DTYPE" \
    MASTER_PORT="${MASTER_PORT:-29794}" \
    bash "$SCRIPT_DIR/run_guarded_training.sh" \
        bash "$SCRIPT_DIR/wiki_e6_fullrank_qkvo_expert_mha_a100_bf16.sh"
