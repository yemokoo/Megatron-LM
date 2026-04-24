#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

run_and_pause() {
    local name="$1"
    shift
    echo "[START] $name $(date)"
    "$@"
    echo "[END] $name $(date)"
    sleep "${PAUSE_SECONDS:-300}"
}

export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
export A2_SOURCE_WEIGHTS_DIR="${A2_SOURCE_WEIGHTS_DIR:-$PROJECT_ROOT/.local/weights/a100/mha/wiki-a-moe-bf16/a2-wiki-ffn-moe-mha-a100-bf16-mb96-1800}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-4}"
export NUM_EXPERTS="${NUM_EXPERTS:-7}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-2}"

run_and_pause "a2_conversation_freeze" env \
    WANDB_MODE=offline \
    DIRECT_LOCAL_SAVE=1 \
    RUN_ID="${FREEZE_RUN_ID:-a2-wiki-to-conversation-ffn-moe-freeze-mha-a100-bf16-mb${FREEZE_MICRO_BATCH_SIZE:-96}-1800}" \
    TRAIN_WEIGHTS="${FREEZE_TRAIN_WEIGHTS:-$PROJECT_ROOT/.local/weights/a100/mha/wiki-to-conversation-moe-bf16-freeze/a2-wiki-to-conversation-ffn-moe-freeze-mha-a100-bf16-mb${FREEZE_MICRO_BATCH_SIZE:-96}-1800}" \
    SOURCE_WEIGHTS_DIR="$A2_SOURCE_WEIGHTS_DIR" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_EXP_NAME="${FREEZE_WANDB_EXP_NAME:-A2 - Wiki to Conversation FFN MoE Freeze MHA mb${FREEZE_MICRO_BATCH_SIZE:-96}}" \
    MICRO_BATCH_SIZE="${FREEZE_MICRO_BATCH_SIZE:-96}" \
    GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
    TRAIN_ITERS="$TRAIN_ITERS" \
    SOURCE_NUM_EXPERTS="$SOURCE_NUM_EXPERTS" \
    NUM_EXPERTS="$NUM_EXPERTS" \
    MOE_ROUTER_TOPK="$MOE_ROUTER_TOPK" \
    MASTER_PORT="${FREEZE_MASTER_PORT:-29621}" \
    "$SCRIPT_DIR/run_guarded_training.sh" \
    bash "$SCRIPT_DIR/wiki_to_conversation_freeze_a2_mha_a100_bf16.sh"

run_and_pause "a2_conversation_attn_unfreeze" env \
    WANDB_MODE=offline \
    DIRECT_LOCAL_SAVE=1 \
    RUN_ID="${ATTN_RUN_ID:-a2-wiki-to-conversation-ffn-moe-attn-unfreeze-kl1.0-mha-a100-bf16-mb${ATTN_MICRO_BATCH_SIZE:-64}-1800}" \
    TRAIN_WEIGHTS="${ATTN_TRAIN_WEIGHTS:-$PROJECT_ROOT/.local/weights/a100/mha/wiki-to-conversation-moe-bf16-attn-unfreeze/a2-wiki-to-conversation-ffn-moe-attn-unfreeze-kl1.0-mha-a100-bf16-mb${ATTN_MICRO_BATCH_SIZE:-64}-1800}" \
    SOURCE_WEIGHTS_DIR="$A2_SOURCE_WEIGHTS_DIR" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_EXP_NAME="${ATTN_WANDB_EXP_NAME:-A2 - Wiki to Conversation FFN MoE Attention Unfreeze KL1.0 MHA mb${ATTN_MICRO_BATCH_SIZE:-64}}" \
    MICRO_BATCH_SIZE="${ATTN_MICRO_BATCH_SIZE:-64}" \
    GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
    TRAIN_ITERS="$TRAIN_ITERS" \
    SOURCE_NUM_EXPERTS="$SOURCE_NUM_EXPERTS" \
    NUM_EXPERTS="$NUM_EXPERTS" \
    MOE_ROUTER_TOPK="$MOE_ROUTER_TOPK" \
    ENABLE_OLD_MODEL_KL=1 \
    OLD_MODEL_KL_COEFF="${OLD_MODEL_KL_COEFF:-1.0}" \
    OLD_MODEL_KL_TEMPERATURE="${OLD_MODEL_KL_TEMPERATURE:-1.0}" \
    MASTER_PORT="${ATTN_MASTER_PORT:-29622}" \
    "$SCRIPT_DIR/run_guarded_training.sh" \
    bash "$SCRIPT_DIR/wiki_to_conversation_attn_unfreeze_a2_mha_a100_bf16.sh"

echo "[ALL DONE] $(date)"
