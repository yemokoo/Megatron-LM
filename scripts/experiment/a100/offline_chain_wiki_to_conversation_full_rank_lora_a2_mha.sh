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

run_and_pause "a2_conversation_full_rank_lora_freeze" env \
    WANDB_MODE=offline \
    DIRECT_LOCAL_SAVE=1 \
    RUN_ID="${FULL_RANK_RUN_ID:-a2-wiki-to-conversation-ffn-moe-full-rank-lora-freeze-mha-a100-bf16-mb${FULL_RANK_MICRO_BATCH_SIZE:-32}-1800}" \
    TRAIN_WEIGHTS="${FULL_RANK_TRAIN_WEIGHTS:-$PROJECT_ROOT/.local/weights/a100/mha/wiki-to-conversation-full-rank-lora-freeze/a2-wiki-to-conversation-ffn-moe-full-rank-lora-freeze-mha-a100-bf16-mb${FULL_RANK_MICRO_BATCH_SIZE:-32}-1800}" \
    SOURCE_WEIGHTS_DIR="$A2_SOURCE_WEIGHTS_DIR" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_EXP_NAME="${FULL_RANK_WANDB_EXP_NAME:-A2 - Wiki to Conversation FFN MoE Full-Rank LoRA Freeze MHA mb${FULL_RANK_MICRO_BATCH_SIZE:-32}}" \
    LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND=0 \
    RUN_INITIAL_PROBE_EVAL=1 \
    PROBE_EVAL_INTERVAL=100 \
    SECONDARY_PROBE_EVAL_INTERVAL=100 \
    MICRO_BATCH_SIZE="${FULL_RANK_MICRO_BATCH_SIZE:-32}" \
    GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
    TRAIN_ITERS="$TRAIN_ITERS" \
    SOURCE_NUM_EXPERTS="$SOURCE_NUM_EXPERTS" \
    NUM_EXPERTS="$NUM_EXPERTS" \
    MOE_ROUTER_TOPK="$MOE_ROUTER_TOPK" \
    ENABLE_OLD_MODEL_KL=0 \
    OLD_MODEL_KL_COEFF=0.0 \
    OLD_MODEL_KL_TEMPERATURE=1.0 \
    ATTN_FULL_RANK_LORA_RANK="${ATTN_FULL_RANK_LORA_RANK:-1024}" \
    ATTN_FULL_RANK_LORA_ALPHA="${ATTN_FULL_RANK_LORA_ALPHA:-1024}" \
    ATTN_FULL_RANK_LORA_TARGETS="${ATTN_FULL_RANK_LORA_TARGETS:-qkvo}" \
    ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="${ATTN_FULL_RANK_LORA_ACTIVE_TARGETS:-}" \
    MASTER_PORT="${FULL_RANK_MASTER_PORT:-29623}" \
    "$SCRIPT_DIR/run_guarded_training.sh" \
    bash "$SCRIPT_DIR/wiki_to_conversation_full_rank_lora_freeze_a2_mha_a100_bf16.sh"

echo "[ALL DONE] $(date)"
