#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-300}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1800}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export PAUSE_SECONDS="${PAUSE_SECONDS:-180}"
export BASE_STAGE_DIR="${BASE_STAGE_DIR:-$PROJECT_ROOT/.local/weights/a100/mha/shared-router-granularity-qkvo}"

export CODE_MICRO_BATCH_SIZE_G1="${CODE_MICRO_BATCH_SIZE_G1:-96}"
export CODE_MICRO_BATCH_SIZE_G2="${CODE_MICRO_BATCH_SIZE_G2:-72}"
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"
export ATTN_LORA_GROUPED_GEMM="${ATTN_LORA_GROUPED_GEMM:-1}"
export MOE_PERMUTE_FUSION="${MOE_PERMUTE_FUSION:-0}"
export MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}"
export SHARED_ROUTER_TRAIN_MASK_EXISTING_EXPERTS=1

is_completed() {
    local run_dir="$1"
    local latest_file="${run_dir}/latest_checkpointed_iteration.txt"

    [ -f "$latest_file" ] && [ "$(tr -d '\n\r[:space:]' < "$latest_file")" = "$TRAIN_ITERS" ]
}

pause_after_stage() {
    if [ "$PAUSE_SECONDS" -gt 0 ]; then
        echo "[PAUSE] sleeping ${PAUSE_SECONDS}s before next code run $(date)"
        sleep "$PAUSE_SECONDS"
    fi
}

run_code_experiment() {
    local exp_id="$1"
    local topk="$2"
    local moe_ffn_hidden="$3"
    local attn_rank="$4"
    local source_experts="$5"
    local target_experts="$6"
    local code_micro_batch_size="$7"
    local code_port="$8"
    local wiki_run_id="$9"

    local code_tag="g${exp_id}-top${topk}-e${source_experts}to${target_experts}-ffn${moe_ffn_hidden}-r${attn_rank}"
    local code_run_id="${code_tag}-wiki-to-code-shared-router-qkvo-mha-a100-bf16-mb${code_micro_batch_size}-code-train-mask-wiki-experts-${TRAIN_ITERS}"
    local wiki_weights="${BASE_STAGE_DIR}/wiki/${wiki_run_id}"
    local code_weights="${BASE_STAGE_DIR}/code/${code_run_id}"

    echo "[CONFIG] exp=${exp_id} topk=${topk} ffn_hidden=${moe_ffn_hidden} attn_rank=${attn_rank} experts=${source_experts}->${target_experts} code_mb=${code_micro_batch_size}"
    echo "[CONFIG] train mask existing experts: 0..$((source_experts - 1))"
    echo "[CONFIG] eval/probe experts: all ${target_experts} experts enabled"
    echo "[CONFIG] wiki=${wiki_weights}"
    echo "[CONFIG] code=${code_weights}"

    if ! is_completed "$wiki_weights"; then
        echo "[ERROR] missing completed wiki checkpoint ${TRAIN_ITERS}: ${wiki_weights}" >&2
        exit 1
    fi

    if is_completed "$code_weights"; then
        echo "[SKIP] code already completed: ${code_run_id}"
        return
    fi

    echo "[START] code exp=${exp_id} $(date)"
    env \
        WANDB_MODE="$WANDB_MODE" \
        DIRECT_LOCAL_SAVE=1 \
        RUN_ID="$code_run_id" \
        TRAIN_WEIGHTS="$code_weights" \
        STAGE1_WEIGHTS_DIR="$wiki_weights" \
        WANDB_PROJECT="$WANDB_PROJECT" \
        WANDB_EXP_NAME="G${exp_id} - code train masks wiki experts top${topk}" \
        LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND=0 \
        RUN_INITIAL_PROBE_EVAL=1 \
        PROBE_EVAL_INTERVAL=100 \
        SECONDARY_PROBE_EVAL_INTERVAL=100 \
        MICRO_BATCH_SIZE="$code_micro_batch_size" \
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
        ATTN_LORA_RANK="$attn_rank" \
        ATTN_LORA_ALPHA="$attn_rank" \
        ATTN_FULL_RANK_LORA_RANK="$attn_rank" \
        ATTN_FULL_RANK_LORA_ALPHA="$attn_rank" \
        ATTN_FULL_RANK_LORA_TARGETS=qkvo \
        ATTN_FULL_RANK_LORA_ACTIVE_TARGETS= \
        ATTN_LORA_GROUPED_GEMM="$ATTN_LORA_GROUPED_GEMM" \
        MOE_GROUPED_GEMM="$MOE_GROUPED_GEMM" \
        MOE_PERMUTE_FUSION="$MOE_PERMUTE_FUSION" \
        MOE_ROUTER_DTYPE="$MOE_ROUTER_DTYPE" \
        SHARED_ROUTER_TRAIN_MASK_EXISTING_EXPERTS=1 \
        SHARED_ROUTER_TRAIN_MASK_EXISTING_EXPERTS_FROM_NUM_EXPERTS="$source_experts" \
        MASTER_PORT="$code_port" \
        "$SCRIPT_DIR/run_guarded_training.sh" \
        bash "$SCRIPT_DIR/code_from_wiki_e6_fullrank_qkvo_expert_mha_a100_bf16.sh"
    echo "[END] code exp=${exp_id} $(date)"
}

run_code_experiment \
    1 2 704 512 4 8 "$CODE_MICRO_BATCH_SIZE_G1" 29692 \
    "${WIKI_G1_RUN_ID:-g1-top2-e4-ffn704-r512-wiki-shared-router-qkvo-mha-a100-bf16-mb72-1800}"
pause_after_stage

run_code_experiment \
    2 4 352 256 8 16 "$CODE_MICRO_BATCH_SIZE_G2" 29694 \
    "${WIKI_G2_RUN_ID:-g2-top4-e8-ffn352-r256-wiki-shared-router-qkvo-mha-a100-bf16-mb72-1800}"

echo "[ALL CODE DONE] $(date)"
