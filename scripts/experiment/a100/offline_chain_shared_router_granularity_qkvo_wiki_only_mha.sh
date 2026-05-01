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
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-96}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export PAUSE_SECONDS="${PAUSE_SECONDS:-1200}"
export BASE_STAGE_DIR="${BASE_STAGE_DIR:-$PROJECT_ROOT/.local/weights/a100/mha/shared-router-granularity-qkvo/wiki}"

is_completed() {
    local run_dir="$1"
    local latest_file="${run_dir}/latest_checkpointed_iteration.txt"

    [ -f "$latest_file" ] && [ "$(tr -d '\n\r[:space:]' < "$latest_file")" = "$TRAIN_ITERS" ]
}

pause_after_stage() {
    if [ "$PAUSE_SECONDS" -gt 0 ]; then
        echo "[PAUSE] sleeping ${PAUSE_SECONDS}s before next wiki run $(date)"
        sleep "$PAUSE_SECONDS"
    fi
}

run_wiki() {
    local exp_id="$1"
    local topk="$2"
    local moe_ffn_hidden="$3"
    local attn_rank="$4"
    local num_experts="$5"
    local master_port="$6"

    local tag="g${exp_id}-top${topk}-e${num_experts}-ffn${moe_ffn_hidden}-r${attn_rank}"
    local run_id="${tag}-wiki-shared-router-qkvo-mha-a100-bf16-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}"
    local train_weights="${BASE_STAGE_DIR}/${run_id}"

    echo "[CONFIG] exp=${exp_id} topk=${topk} ffn_hidden=${moe_ffn_hidden} attn_rank=${attn_rank} experts=${num_experts}"
    echo "[CONFIG] wiki=${train_weights}"

    if is_completed "$train_weights"; then
        echo "[SKIP] wiki already completed: ${run_id}"
        return 0
    fi

    echo "[START] wiki exp=${exp_id} $(date)"
    env \
        WANDB_MODE="$WANDB_MODE" \
        DIRECT_LOCAL_SAVE=1 \
        RUN_ID="$run_id" \
        TRAIN_WEIGHTS="$train_weights" \
        WANDB_PROJECT="$WANDB_PROJECT" \
        WANDB_EXP_NAME="G${exp_id} - shared router qkvo top${topk} ffn${moe_ffn_hidden} r${attn_rank} - wiki" \
        MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" \
        GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
        TRAIN_ITERS="$TRAIN_ITERS" \
        SAVE_INTERVAL="$SAVE_INTERVAL" \
        EVAL_INTERVAL="$EVAL_INTERVAL" \
        PROBE_EVAL_INTERVAL=100 \
        SECONDARY_PROBE_EVAL_INTERVAL=100 \
        NUM_EXPERTS="$num_experts" \
        MOE_ROUTER_TOPK="$topk" \
        MOE_FFN_HIDDEN_SIZE="$moe_ffn_hidden" \
        ATTN_LORA_RANK="$attn_rank" \
        ATTN_LORA_ALPHA="$attn_rank" \
        ATTN_FULL_RANK_LORA_RANK="$attn_rank" \
        ATTN_FULL_RANK_LORA_ALPHA="$attn_rank" \
        ATTN_FULL_RANK_LORA_TARGETS=qkvo \
        ATTN_FULL_RANK_LORA_ACTIVE_TARGETS= \
        MASTER_PORT="$master_port" \
        "$SCRIPT_DIR/run_guarded_training.sh" \
        bash "$SCRIPT_DIR/wiki_e6_fullrank_qkvo_expert_mha_a100_bf16.sh"
    echo "[END] wiki exp=${exp_id} $(date)"
}

run_wiki 1 2 704 512 4 29681
pause_after_stage
run_wiki 2 4 352 256 8 29682
pause_after_stage
run_wiki 3 8 176 128 16 29683
pause_after_stage
run_wiki 4 16 88 64 32 29684

echo "[ALL DONE] $(date)"
echo "Offline W&B runs are under:"
echo "  ${BASE_STAGE_DIR}/*/wandb/wandb/offline-run-*"
