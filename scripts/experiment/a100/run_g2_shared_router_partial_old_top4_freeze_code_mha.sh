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
export STAGE1_REQUIRED_ITERS="${STAGE1_REQUIRED_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-600}"
export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-1}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-$TRAIN_ITERS}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export USE_GUARD="${USE_GUARD:-1}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-72}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"

export G2_ROOT="${G2_ROOT:-$PROJECT_ROOT/.local/weights/a100/mha/g2-checkpoints}"
export RUN_ID="${RUN_ID:-g2-exp4-top4oldfreeze-e8to16-ffn352-r256-wiki-to-code-layerwise-top4-old-freeze-qkvo-mha-a100-bf16-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"
export STAGE1_WEIGHTS_DIR="${STAGE1_WEIGHTS_DIR:-$G2_ROOT/wiki/g2-top4-e8-ffn352-r256-wiki-shared-router-qkvo-mha-a100-bf16-mb72-1800}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$G2_ROOT/code/phase1/$RUN_ID}"

export SHARED_ROUTER_HYBRID_PARTIAL_FREEZE_MASK="${SHARED_ROUTER_HYBRID_PARTIAL_FREEZE_MASK:-$PROJECT_ROOT/analysis_outputs/g2_wiki_router_softmax_importance_1m/router_softmax_top4_freeze_mask.json}"

export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-8}"
export NUM_EXPERTS="${NUM_EXPERTS:-16}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export ATTN_LORA_RANK="${ATTN_LORA_RANK:-256}"
export ATTN_LORA_ALPHA="${ATTN_LORA_ALPHA:-$ATTN_LORA_RANK}"
export ATTN_FULL_RANK_LORA_RANK="${ATTN_FULL_RANK_LORA_RANK:-256}"
export ATTN_FULL_RANK_LORA_ALPHA="${ATTN_FULL_RANK_LORA_ALPHA:-$ATTN_FULL_RANK_LORA_RANK}"
export ATTN_FULL_RANK_LORA_TARGETS="${ATTN_FULL_RANK_LORA_TARGETS:-qkvo}"
export ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="${ATTN_FULL_RANK_LORA_ACTIVE_TARGETS:-}"

export SHARED_ROUTER_HYBRID_TRAIN_ALL_EXPERTS_AND_ROUTER=0
export SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS=0
export ENABLE_OLD_MODEL_KL=0
export OLD_MODEL_KL_COEFF=0.0
export OLD_MODEL_KL_TEMPERATURE=1.0
export ROUTER_MEMORY_KL_COEFF="${ROUTER_MEMORY_KL_COEFF:-0.0}"
export ROUTER_MEMORY_INTERVAL="${ROUTER_MEMORY_INTERVAL:-0}"
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.01}"
export MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.001}"
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"
export ATTN_LORA_GROUPED_GEMM="${ATTN_LORA_GROUPED_GEMM:-1}"
export MOE_PERMUTE_FUSION="${MOE_PERMUTE_FUSION:-0}"
export MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}"
export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-configs/model/flame-shared-router-hybrid-experts.sh}"

is_completed() {
    local run_dir="$1"
    local latest_file="${run_dir}/latest_checkpointed_iteration.txt"
    [ -f "$latest_file" ] && [ "$(tr -d '\n\r[:space:]' < "$latest_file")" = "$TRAIN_ITERS" ]
}

if [ ! -f "$SHARED_ROUTER_HYBRID_PARTIAL_FREEZE_MASK" ]; then
    echo "[ERROR] missing partial-freeze mask: $SHARED_ROUTER_HYBRID_PARTIAL_FREEZE_MASK" >&2
    echo "[HINT] build it first with plot_wiki_router_softmax_importance.py using --router-softmax-max-tokens 1048576." >&2
    exit 1
fi

if ! is_completed "$STAGE1_WEIGHTS_DIR"; then
    echo "[ERROR] missing completed G2 wiki checkpoint: $STAGE1_WEIGHTS_DIR" >&2
    exit 1
fi

if is_completed "$TRAIN_WEIGHTS"; then
    echo "[SKIP] already completed: $RUN_ID"
    exit 0
fi

echo "[CONFIG] G2 shared-router partial old-expert freeze code training"
echo "[CONFIG] code steps=${TRAIN_ITERS}, mb=${MICRO_BATCH_SIZE}, gbs=${GLOBAL_BATCH_SIZE}"
echo "[CONFIG] selection=layer-wise top4 old/wiki experts from pre-TopK router softmax"
echo "[CONFIG] frozen: selected old router rows + selected old FFN experts + selected old attention experts + dense trunk"
echo "[CONFIG] trainable: unselected old router/expert rows + all new code router/expert rows"
echo "[CONFIG] aux/z loss=${MOE_AUX_LOSS_COEFF}/${MOE_Z_LOSS_COEFF}"
echo "[CONFIG] wiki=${STAGE1_WEIGHTS_DIR}"
echo "[CONFIG] mask=${SHARED_ROUTER_HYBRID_PARTIAL_FREEZE_MASK}"
echo "[CONFIG] code=${TRAIN_WEIGHTS}"

env \
    WANDB_MODE="$WANDB_MODE" \
    DIRECT_LOCAL_SAVE=1 \
    WANDB_LOG_CHECKPOINTS="$WANDB_LOG_CHECKPOINTS" \
    RUN_ID="$RUN_ID" \
    TRAIN_WEIGHTS="$TRAIN_WEIGHTS" \
    STAGE1_WEIGHTS_DIR="$STAGE1_WEIGHTS_DIR" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2 - exp4 layerwise top4 old expert freeze - wiki to code}" \
    RUN_INITIAL_PROBE_EVAL="${RUN_INITIAL_PROBE_EVAL:-1}" \
    RUN_INITIAL_VALID_EVAL="${RUN_INITIAL_VALID_EVAL:-1}" \
    PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}" \
    SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}" \
    MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" \
    GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
    TRAIN_ITERS="$TRAIN_ITERS" \
    SOURCE_REQUIRED_ITERS="$STAGE1_REQUIRED_ITERS" \
    SAVE_INTERVAL="$SAVE_INTERVAL" \
    SAVE_CHECKPOINTS="$SAVE_CHECKPOINTS" \
    EVAL_INTERVAL="$EVAL_INTERVAL" \
    LOG_INTERVAL="$LOG_INTERVAL" \
    SOURCE_NUM_EXPERTS="$SOURCE_NUM_EXPERTS" \
    NUM_EXPERTS="$NUM_EXPERTS" \
    MOE_ROUTER_TOPK="$MOE_ROUTER_TOPK" \
    MOE_FFN_HIDDEN_SIZE="$MOE_FFN_HIDDEN_SIZE" \
    ENABLE_OLD_MODEL_KL="$ENABLE_OLD_MODEL_KL" \
    OLD_MODEL_KL_COEFF="$OLD_MODEL_KL_COEFF" \
    OLD_MODEL_KL_TEMPERATURE="$OLD_MODEL_KL_TEMPERATURE" \
    ATTN_LORA_RANK="$ATTN_LORA_RANK" \
    ATTN_LORA_ALPHA="$ATTN_LORA_ALPHA" \
    ATTN_FULL_RANK_LORA_RANK="$ATTN_FULL_RANK_LORA_RANK" \
    ATTN_FULL_RANK_LORA_ALPHA="$ATTN_FULL_RANK_LORA_ALPHA" \
    ATTN_FULL_RANK_LORA_TARGETS="$ATTN_FULL_RANK_LORA_TARGETS" \
    ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="$ATTN_FULL_RANK_LORA_ACTIVE_TARGETS" \
    SHARED_ROUTER_HYBRID_PARTIAL_FREEZE_MASK="$SHARED_ROUTER_HYBRID_PARTIAL_FREEZE_MASK" \
    SHARED_ROUTER_HYBRID_TRAIN_ALL_EXPERTS_AND_ROUTER="$SHARED_ROUTER_HYBRID_TRAIN_ALL_EXPERTS_AND_ROUTER" \
    SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS="$SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS" \
    ROUTER_MEMORY_KL_COEFF="$ROUTER_MEMORY_KL_COEFF" \
    ROUTER_MEMORY_INTERVAL="$ROUTER_MEMORY_INTERVAL" \
    MOE_AUX_LOSS_COEFF="$MOE_AUX_LOSS_COEFF" \
    MOE_Z_LOSS_COEFF="$MOE_Z_LOSS_COEFF" \
    MOE_GROUPED_GEMM="$MOE_GROUPED_GEMM" \
    ATTN_LORA_GROUPED_GEMM="$ATTN_LORA_GROUPED_GEMM" \
    MOE_PERMUTE_FUSION="$MOE_PERMUTE_FUSION" \
    MOE_ROUTER_DTYPE="$MOE_ROUTER_DTYPE" \
    NUM_QUERY_GROUPS="$NUM_QUERY_GROUPS" \
    MODEL_CONFIG_SCRIPT="$MODEL_CONFIG_SCRIPT" \
    MASTER_PORT="${MASTER_PORT:-29791}" \
    USE_GUARD="$USE_GUARD" \
    bash -lc 'if [ "$USE_GUARD" = "1" ]; then exec scripts/experiment/a100/run_guarded_training.sh bash scripts/experiment/continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh; else exec bash scripts/experiment/continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh; fi'

echo "[DONE] G2 shared-router partial old-expert freeze code training $(date)"
