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
export WIKI_MICRO_BATCH_SIZE="${WIKI_MICRO_BATCH_SIZE:-96}"
export CODE_MICRO_BATCH_SIZE="${CODE_MICRO_BATCH_SIZE:-96}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export PAUSE_SECONDS="${PAUSE_SECONDS:-1200}"
export BASE_STAGE_DIR="${BASE_STAGE_DIR:-$PROJECT_ROOT/.local/weights/a100/mha/shared-router-granularity-qkvo}"

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

run_experiment() {
    local exp_id="$1"
    local topk="$2"
    local moe_ffn_hidden="$3"
    local attn_rank="$4"
    local source_experts="$5"
    local target_experts="$6"
    local wiki_port="$7"
    local code_port="$8"

    local tag="g${exp_id}-top${topk}-e${source_experts}to${target_experts}-ffn${moe_ffn_hidden}-r${attn_rank}"
    local wiki_run_id="${tag}-wiki-shared-router-qkvo-mha-a100-bf16-mb${WIKI_MICRO_BATCH_SIZE}-${TRAIN_ITERS}"
    local code_run_id="${tag}-wiki-to-code-shared-router-qkvo-mha-a100-bf16-mb${CODE_MICRO_BATCH_SIZE}-${TRAIN_ITERS}"
    local wiki_weights="${BASE_STAGE_DIR}/wiki/${wiki_run_id}"
    local code_weights="${BASE_STAGE_DIR}/code/${code_run_id}"

    echo "[CONFIG] exp=${exp_id} topk=${topk} ffn_hidden=${moe_ffn_hidden} attn_rank=${attn_rank} experts=${source_experts}->${target_experts}"
    echo "[CONFIG] wiki=${wiki_weights}"
    echo "[CONFIG] code=${code_weights}"

    if is_completed "$wiki_weights"; then
        echo "[SKIP] wiki already completed: ${wiki_run_id}"
    else
        echo "[START] wiki exp=${exp_id} $(date)"
        env \
            WANDB_MODE="$WANDB_MODE" \
            DIRECT_LOCAL_SAVE=1 \
            RUN_ID="$wiki_run_id" \
            TRAIN_WEIGHTS="$wiki_weights" \
            WANDB_PROJECT="$WANDB_PROJECT" \
            WANDB_EXP_NAME="G${exp_id} - shared router qkvo top${topk} ffn${moe_ffn_hidden} r${attn_rank} - wiki" \
            MICRO_BATCH_SIZE="$WIKI_MICRO_BATCH_SIZE" \
            GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
            TRAIN_ITERS="$TRAIN_ITERS" \
            SAVE_INTERVAL="$SAVE_INTERVAL" \
            EVAL_INTERVAL="$EVAL_INTERVAL" \
            PROBE_EVAL_INTERVAL=100 \
            SECONDARY_PROBE_EVAL_INTERVAL=100 \
            NUM_EXPERTS="$source_experts" \
            MOE_ROUTER_TOPK="$topk" \
            MOE_FFN_HIDDEN_SIZE="$moe_ffn_hidden" \
            ATTN_LORA_RANK="$attn_rank" \
            ATTN_LORA_ALPHA="$attn_rank" \
            ATTN_FULL_RANK_LORA_RANK="$attn_rank" \
            ATTN_FULL_RANK_LORA_ALPHA="$attn_rank" \
            ATTN_FULL_RANK_LORA_TARGETS=qkvo \
            ATTN_FULL_RANK_LORA_ACTIVE_TARGETS= \
            MASTER_PORT="$wiki_port" \
            "$SCRIPT_DIR/run_guarded_training.sh" \
            bash "$SCRIPT_DIR/wiki_e6_fullrank_qkvo_expert_mha_a100_bf16.sh"
        echo "[END] wiki exp=${exp_id} $(date)"
        pause_after_stage
    fi

    if is_completed "$code_weights"; then
        echo "[SKIP] code already completed: ${code_run_id}"
    else
        echo "[START] code exp=${exp_id} $(date)"
        env \
            WANDB_MODE="$WANDB_MODE" \
            DIRECT_LOCAL_SAVE=1 \
            RUN_ID="$code_run_id" \
            TRAIN_WEIGHTS="$code_weights" \
            STAGE1_WEIGHTS_DIR="$wiki_weights" \
            WANDB_PROJECT="$WANDB_PROJECT" \
            WANDB_EXP_NAME="G${exp_id} - shared router qkvo top${topk} ffn${moe_ffn_hidden} r${attn_rank} - wiki to code" \
            LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND=0 \
            RUN_INITIAL_PROBE_EVAL=1 \
            PROBE_EVAL_INTERVAL=100 \
            SECONDARY_PROBE_EVAL_INTERVAL=100 \
            MICRO_BATCH_SIZE="$CODE_MICRO_BATCH_SIZE" \
            GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
            TRAIN_ITERS="$TRAIN_ITERS" \
            SAVE_INTERVAL="$SAVE_INTERVAL" \
            EVAL_INTERVAL="$EVAL_INTERVAL" \
            SOURCE_NUM_EXPERTS="$source_experts" \
            NUM_EXPERTS="$target_experts" \
            MOE_ROUTER_TOPK="$topk" \
            MOE_FFN_HIDDEN_SIZE="$moe_ffn_hidden" \
            ENABLE_OLD_MODEL_KL=0 \
            OLD_MODEL_KL_COEFF=0.0 \
            OLD_MODEL_KL_TEMPERATURE=1.0 \
            ATTN_LORA_RANK="$attn_rank" \
            ATTN_LORA_ALPHA="$attn_rank" \
            ATTN_FULL_RANK_LORA_RANK="$attn_rank" \
            ATTN_FULL_RANK_LORA_ALPHA="$attn_rank" \
            ATTN_FULL_RANK_LORA_TARGETS=qkvo \
            ATTN_FULL_RANK_LORA_ACTIVE_TARGETS= \
            MASTER_PORT="$code_port" \
            "$SCRIPT_DIR/run_guarded_training.sh" \
            bash "$SCRIPT_DIR/code_from_wiki_e6_fullrank_qkvo_expert_mha_a100_bf16.sh"
        echo "[END] code exp=${exp_id} $(date)"
        if [ "$exp_id" -lt 4 ]; then
            pause_after_stage
        fi
    fi
}

run_experiment 1 2 704 512 4 8 29671 29672
run_experiment 2 4 352 256 8 16 29673 29674
run_experiment 3 8 176 128 16 32 29675 29676
run_experiment 4 16 88 64 32 64 29677 29678

echo "[ALL DONE] $(date)"
echo "Offline W&B runs are under:"
echo "  ${BASE_STAGE_DIR}/wiki/*/wandb/wandb/offline-run-*"
echo "  ${BASE_STAGE_DIR}/code/*/wandb/wandb/offline-run-*"
