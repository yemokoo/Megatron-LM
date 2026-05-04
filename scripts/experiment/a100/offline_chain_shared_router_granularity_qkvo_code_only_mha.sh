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
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export PAUSE_SECONDS="${PAUSE_SECONDS:-1200}"
export BASE_STAGE_DIR="${BASE_STAGE_DIR:-$PROJECT_ROOT/.local/weights/a100/mha/shared-router-granularity-qkvo}"

export CODE_MICRO_BATCH_SIZE_G1="${CODE_MICRO_BATCH_SIZE_G1:-96}"
export CODE_MICRO_BATCH_SIZE_G2="${CODE_MICRO_BATCH_SIZE_G2:-72}"
export CODE_MICRO_BATCH_SIZE_G3="${CODE_MICRO_BATCH_SIZE_G3:-48}"
export CODE_MICRO_BATCH_SIZE_G4="${CODE_MICRO_BATCH_SIZE_G4:-24}"

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
    local code_run_id="${code_tag}-wiki-to-code-shared-router-qkvo-mha-a100-bf16-mb${code_micro_batch_size}-${TRAIN_ITERS}"
    local wiki_weights="${BASE_STAGE_DIR}/wiki/${wiki_run_id}"
    local code_weights="${BASE_STAGE_DIR}/code/${code_run_id}"

    echo "[CONFIG] exp=${exp_id} topk=${topk} ffn_hidden=${moe_ffn_hidden} attn_rank=${attn_rank} experts=${source_experts}->${target_experts} code_mb=${code_micro_batch_size}"
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
        WANDB_EXP_NAME="G${exp_id} - shared router qkvo top${topk} ffn${moe_ffn_hidden} r${attn_rank} - wiki to code" \
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
        OLD_MODEL_KL_TEMPERATURE=1.0 \
        ATTN_LORA_RANK="$attn_rank" \
        ATTN_LORA_ALPHA="$attn_rank" \
        ATTN_FULL_RANK_LORA_RANK="$attn_rank" \
        ATTN_FULL_RANK_LORA_ALPHA="$attn_rank" \
        ATTN_FULL_RANK_LORA_TARGETS=qkvo \
        ATTN_FULL_RANK_LORA_ACTIVE_TARGETS= \
        MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-0}" \
        MOE_PERMUTE_FUSION="${MOE_PERMUTE_FUSION:-0}" \
        MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}" \
        MASTER_PORT="$code_port" \
        "$SCRIPT_DIR/run_guarded_training.sh" \
        bash "$SCRIPT_DIR/code_from_wiki_e6_fullrank_qkvo_expert_mha_a100_bf16.sh"
    echo "[END] code exp=${exp_id} $(date)"
}

run_code_experiment \
    1 2 704 512 4 8 "$CODE_MICRO_BATCH_SIZE_G1" 29672 \
    "${WIKI_G1_RUN_ID:-g1-top2-e4-ffn704-r512-wiki-shared-router-qkvo-mha-a100-bf16-mb72-1800}"
pause_after_stage

run_code_experiment \
    2 4 352 256 8 16 "$CODE_MICRO_BATCH_SIZE_G2" 29674 \
    "${WIKI_G2_RUN_ID:-g2-top4-e8-ffn352-r256-wiki-shared-router-qkvo-mha-a100-bf16-mb72-1800}"
pause_after_stage

run_code_experiment \
    3 8 176 128 16 32 "$CODE_MICRO_BATCH_SIZE_G3" 29676 \
    "${WIKI_G3_RUN_ID:-g3-top8-e16-ffn176-r128-wiki-shared-router-qkvo-mha-a100-bf16-mb48-1800}"
pause_after_stage

run_code_experiment \
    4 16 88 64 32 64 "$CODE_MICRO_BATCH_SIZE_G4" 29678 \
    "${WIKI_G4_RUN_ID:-g4-top16-e32-ffn88-r64-wiki-shared-router-qkvo-mha-a100-bf16-mb32-groupedgemm-1800}"

echo "[ALL CODE DONE] $(date)"
echo "Offline W&B runs are under:"
echo "  ${BASE_STAGE_DIR}/code/*/wandb/wandb/offline-run-*"
