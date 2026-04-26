#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
export A2_SOURCE_WEIGHTS_DIR="${A2_SOURCE_WEIGHTS_DIR:-$PROJECT_ROOT/.local/weights/a100/mha/wiki-a-moe-bf16/a2-wiki-ffn-moe-mha-a100-bf16-mb96-1800}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1800}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-96}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-4}"
export NUM_EXPERTS="${NUM_EXPERTS:-7}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-2}"
export PAUSE_SECONDS="${PAUSE_SECONDS:-1200}"
export RANKS="${RANKS:-256 768 640 896 128 64 32 16}"
export BASE_STAGE_DIR="${BASE_STAGE_DIR:-$PROJECT_ROOT/.local/weights/a100/mha/f-attn-full-rank-lora-bf16-freeze}"

run_rank() {
    local rank="$1"
    local run_id="f2-r${rank}-wiki-to-code-ffn-moe-freeze-attn-full-rank-lora-qkvo-mha-a100-bf16-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}"
    local train_weights="${BASE_STAGE_DIR}/${run_id}"
    local latest_file="${train_weights}/latest_checkpointed_iteration.txt"

    if [ -f "$latest_file" ] && [ "$(tr -d '\n\r[:space:]' < "$latest_file")" = "$TRAIN_ITERS" ]; then
        echo "[SKIP] rank=${rank} already completed at ${train_weights}"
        return 0
    fi

    echo "[START] rank=${rank} run_id=${run_id} $(date)"
    env \
        WANDB_MODE=offline \
        DIRECT_LOCAL_SAVE=1 \
        RUN_ID="$run_id" \
        TRAIN_WEIGHTS="$train_weights" \
        SOURCE_WEIGHTS_DIR="$A2_SOURCE_WEIGHTS_DIR" \
        WANDB_PROJECT="$WANDB_PROJECT" \
        WANDB_EXP_NAME="F - Wiki to Code ${rank} Rank Freeze" \
        LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND=0 \
        RUN_INITIAL_PROBE_EVAL=1 \
        PROBE_EVAL_INTERVAL=100 \
        SECONDARY_PROBE_EVAL_INTERVAL=100 \
        MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" \
        GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
        TRAIN_ITERS="$TRAIN_ITERS" \
        SAVE_INTERVAL="$SAVE_INTERVAL" \
        SOURCE_NUM_EXPERTS="$SOURCE_NUM_EXPERTS" \
        NUM_EXPERTS="$NUM_EXPERTS" \
        MOE_ROUTER_TOPK="$MOE_ROUTER_TOPK" \
        ENABLE_OLD_MODEL_KL=0 \
        OLD_MODEL_KL_COEFF=0.0 \
        OLD_MODEL_KL_TEMPERATURE=1.0 \
        ATTN_FULL_RANK_LORA_RANK="$rank" \
        ATTN_FULL_RANK_LORA_ALPHA="$rank" \
        ATTN_FULL_RANK_LORA_TARGETS=qkvo \
        ATTN_FULL_RANK_LORA_ACTIVE_TARGETS= \
        MASTER_PORT="${MASTER_PORT:-29625}" \
        "$SCRIPT_DIR/run_guarded_training.sh" \
        bash "$SCRIPT_DIR/code_from_wiki_f2_attn_full_rank_lora_freeze_shared_mha_a100_bf16.sh"
    echo "[END] rank=${rank} $(date)"
}

rank_count="$(wc -w <<< "$RANKS" | tr -d '[:space:]')"
rank_index=0
for rank in $RANKS; do
    rank_index=$((rank_index + 1))
    run_rank "$rank"

    if [ "$rank_index" -lt "$rank_count" ]; then
        echo "[PAUSE] sleeping ${PAUSE_SECONDS}s before next rank $(date)"
        sleep "$PAUSE_SECONDS"
    fi
done

echo "[ALL DONE] $(date)"
echo "Offline W&B runs are under:"
for rank in $RANKS; do
    run_id="f2-r${rank}-wiki-to-code-ffn-moe-freeze-attn-full-rank-lora-qkvo-mha-a100-bf16-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}"
    echo "  ${BASE_STAGE_DIR}/${run_id}/wandb/wandb/offline-run-*-${run_id}"
done
