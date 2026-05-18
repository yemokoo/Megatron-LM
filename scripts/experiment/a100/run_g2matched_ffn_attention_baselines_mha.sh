#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-$TRAIN_ITERS}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-$TRAIN_ITERS}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}"
export WIKI_MICRO_BATCH_SIZE="${WIKI_MICRO_BATCH_SIZE:-128}"
export CODE_MICRO_BATCH_SIZE="${CODE_MICRO_BATCH_SIZE:-96}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export ATTN_FULL_RANK_LORA_RANK="${ATTN_FULL_RANK_LORA_RANK:-1024}"
export PAUSE_SECONDS="${PAUSE_SECONDS:-0}"

BASE_WEIGHTS_DIR="${BASE_WEIGHTS_DIR:-$PROJECT_ROOT/.local/weights/a100/mha}"
WIKI_RUN_ID="${WIKI_RUN_ID:-g2matched-top4-e8-ffn352-wiki-ffn-moe-mha-a100-bf16-mb${WIKI_MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"
ATTN_FREEZE_RUN_ID="${ATTN_FREEZE_RUN_ID:-g2matched-top4-e8to16-ffn352-wiki-to-code-ffn-moe-attn-freeze-mha-a100-bf16-mb${CODE_MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"
ATTN_LORA_RUN_ID="${ATTN_LORA_RUN_ID:-g2matched-top4-e8to16-ffn352-wiki-to-code-ffn-moe-attn-fullrank-qkvo-r${ATTN_FULL_RANK_LORA_RANK}-mha-a100-bf16-mb${CODE_MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"

WIKI_WEIGHTS="$BASE_WEIGHTS_DIR/wiki-a-moe-g2matched-bf16/$WIKI_RUN_ID"
ATTN_FREEZE_WEIGHTS="$BASE_WEIGHTS_DIR/g2matched-ffn-moe-attn-freeze-bf16/$ATTN_FREEZE_RUN_ID"
ATTN_LORA_WEIGHTS="$BASE_WEIGHTS_DIR/g2matched-ffn-moe-attn-full-rank-lora-bf16/$ATTN_LORA_RUN_ID"

is_completed() {
    local run_dir="$1"
    local latest_file="${run_dir}/latest_checkpointed_iteration.txt"

    [ -f "$latest_file" ] && [ "$(tr -d '\n\r[:space:]' < "$latest_file")" = "$TRAIN_ITERS" ]
}

pause_after_stage() {
    if [ "$PAUSE_SECONDS" -gt 0 ]; then
        echo "[PAUSE] sleeping ${PAUSE_SECONDS}s before next stage $(date)"
        sleep "$PAUSE_SECONDS"
    fi
}

echo "[CONFIG] G2-matched FFN attention baselines"
echo "[CONFIG] common: topk=4, source experts=8, target experts=16, moe_ffn_hidden=352"
echo "[CONFIG] wiki micro_batch_size=$WIKI_MICRO_BATCH_SIZE, code micro_batch_size=$CODE_MICRO_BATCH_SIZE"
echo "[CONFIG] wiki=$WIKI_WEIGHTS"
echo "[CONFIG] attn_freeze=$ATTN_FREEZE_WEIGHTS"
echo "[CONFIG] attn_fullrank_lora=$ATTN_LORA_WEIGHTS"

if is_completed "$WIKI_WEIGHTS"; then
    echo "[SKIP] wiki source already completed: $WIKI_RUN_ID"
else
    echo "[START] wiki source $(date)"
    env \
        WANDB_MODE="$WANDB_MODE" \
        DIRECT_LOCAL_SAVE=1 \
        RUN_ID="$WIKI_RUN_ID" \
        TRAIN_WEIGHTS="$WIKI_WEIGHTS" \
        WANDB_PROJECT="$WANDB_PROJECT" \
        MICRO_BATCH_SIZE="$WIKI_MICRO_BATCH_SIZE" \
        GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
        TRAIN_ITERS="$TRAIN_ITERS" \
        SAVE_INTERVAL="$SAVE_INTERVAL" \
        EVAL_INTERVAL="$EVAL_INTERVAL" \
        LOG_INTERVAL="$LOG_INTERVAL" \
        PROBE_EVAL_INTERVAL="$PROBE_EVAL_INTERVAL" \
        SECONDARY_PROBE_EVAL_INTERVAL="$SECONDARY_PROBE_EVAL_INTERVAL" \
        bash "$SCRIPT_DIR/run_guarded_training.sh" \
        bash "$SCRIPT_DIR/wiki_ffn_moe_g2matched_mha_a100_bf16.sh"
    echo "[END] wiki source $(date)"
    pause_after_stage
fi

if is_completed "$ATTN_FREEZE_WEIGHTS"; then
    echo "[SKIP] attention-freeze continual already completed: $ATTN_FREEZE_RUN_ID"
else
    echo "[START] attention-freeze continual $(date)"
    env \
        WANDB_MODE="$WANDB_MODE" \
        DIRECT_LOCAL_SAVE=1 \
        RUN_ID="$ATTN_FREEZE_RUN_ID" \
        TRAIN_WEIGHTS="$ATTN_FREEZE_WEIGHTS" \
        SOURCE_WEIGHTS_DIR="$WIKI_WEIGHTS" \
        WANDB_PROJECT="$WANDB_PROJECT" \
        MICRO_BATCH_SIZE="$CODE_MICRO_BATCH_SIZE" \
        GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
        TRAIN_ITERS="$TRAIN_ITERS" \
        SAVE_INTERVAL="$SAVE_INTERVAL" \
        EVAL_INTERVAL="$EVAL_INTERVAL" \
        LOG_INTERVAL="$LOG_INTERVAL" \
        PROBE_EVAL_INTERVAL="$PROBE_EVAL_INTERVAL" \
        SECONDARY_PROBE_EVAL_INTERVAL="$SECONDARY_PROBE_EVAL_INTERVAL" \
        bash "$SCRIPT_DIR/run_guarded_training.sh" \
        bash "$SCRIPT_DIR/code_from_wiki_ffn_moe_g2matched_attn_freeze_mha_a100_bf16.sh"
    echo "[END] attention-freeze continual $(date)"
    pause_after_stage
fi

if is_completed "$ATTN_LORA_WEIGHTS"; then
    echo "[SKIP] attention full-rank LoRA continual already completed: $ATTN_LORA_RUN_ID"
else
    echo "[START] attention full-rank LoRA continual $(date)"
    env \
        WANDB_MODE="$WANDB_MODE" \
        DIRECT_LOCAL_SAVE=1 \
        RUN_ID="$ATTN_LORA_RUN_ID" \
        TRAIN_WEIGHTS="$ATTN_LORA_WEIGHTS" \
        SOURCE_WEIGHTS_DIR="$WIKI_WEIGHTS" \
        WANDB_PROJECT="$WANDB_PROJECT" \
        MICRO_BATCH_SIZE="$CODE_MICRO_BATCH_SIZE" \
        GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
        TRAIN_ITERS="$TRAIN_ITERS" \
        SAVE_INTERVAL="$SAVE_INTERVAL" \
        EVAL_INTERVAL="$EVAL_INTERVAL" \
        LOG_INTERVAL="$LOG_INTERVAL" \
        PROBE_EVAL_INTERVAL="$PROBE_EVAL_INTERVAL" \
        SECONDARY_PROBE_EVAL_INTERVAL="$SECONDARY_PROBE_EVAL_INTERVAL" \
        ATTN_FULL_RANK_LORA_RANK="$ATTN_FULL_RANK_LORA_RANK" \
        ATTN_FULL_RANK_LORA_ALPHA="${ATTN_FULL_RANK_LORA_ALPHA:-$ATTN_FULL_RANK_LORA_RANK}" \
        bash "$SCRIPT_DIR/run_guarded_training.sh" \
        bash "$SCRIPT_DIR/code_from_wiki_ffn_moe_g2matched_attn_full_rank_lora_mha_a100_bf16.sh"
    echo "[END] attention full-rank LoRA continual $(date)"
fi

echo "[ALL DONE] $(date)"
