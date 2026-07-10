#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/common.sh"

MODE="${1:-${MOE_EXPANSION_DISTILL_MODE:-logits}}"
case "$MODE" in
    logits|logits_hidden|logits_hidden_router)
        ;;
    *)
        echo "usage: $0 [logits|logits_hidden|logits_hidden_router]" >&2
        exit 1
        ;;
esac

# Wiki 8-expert shared-router teacher -> 16-expert student. FFN and routed
# QKVO LoRA experts use the same layer-wise router. During this initialization
# stage only the newly added expert parameters and router rows are trainable.
export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export G2_ROOT="${G2_ROOT:-$LOCAL_WEIGHTS/a100/mha/g2-checkpoints}"

export STAGE1_WEIGHTS_DIR="${STAGE1_WEIGHTS_DIR:-$G2_ROOT/wiki/g2-top4-e8-ffn352-r256-wiki-shared-router-qkvo-mha-a100-bf16-mb72-1800}"
export SOURCE_REQUIRED_ITERS="${SOURCE_REQUIRED_ITERS:-1800}"
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
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-configs/model/flame-shared-router-hybrid-experts.sh}"

export SHARED_ROUTER_HYBRID_TRAIN_ALL_EXPERTS_AND_ROUTER=0
export SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS=0
export SHARED_ROUTER_HYBRID_FREEZE_PREEXISTING_ONLY=0
export SHARED_ROUTER_HYBRID_PARTIAL_FREEZE_MASK=""
export SHARED_ROUTER_HYBRID_TOPK_WITH_ALL_NEW_EXPERTS=0

export TRAIN_DATASET="${TRAIN_DATASET:-$(dataset_dir_for_task wiki)}"
export DATASET_NAME="${DATASET_NAME:-wiki_train_shared_router_expansion_distill}"
export DATASET_SOURCE="${DATASET_SOURCE:-Wiki train data for shared-router pre-Code expansion distillation}"
export PROBE_DATASET="${PROBE_DATASET:-$(probe_dir_for_task wiki)}"
export PROBE_NAME="${PROBE_NAME:-wiki_probe}"
export SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET:-$(probe_dir_for_task code)}"
export SECONDARY_PROBE_NAME="${SECONDARY_PROBE_NAME:-code_probe}"

export ENABLE_OLD_MODEL_KL=1
export OLD_MODEL_KL_COEFF="${OLD_MODEL_KL_COEFF:-1.0}"
export OLD_MODEL_KL_TEMPERATURE="${OLD_MODEL_KL_TEMPERATURE:-1.0}"
export MOE_EXPANSION_DISTILL_MODE="$MODE"
export MOE_EXPANSION_DISTILL_LM_LOSS_COEFF="${MOE_EXPANSION_DISTILL_LM_LOSS_COEFF:-0.0}"
export MOE_EXPANSION_DISTILL_HIDDEN_MSE_COEFF="${MOE_EXPANSION_DISTILL_HIDDEN_MSE_COEFF:-1.0}"
export MOE_EXPANSION_DISTILL_ROUTER_KL_COEFF="${MOE_EXPANSION_DISTILL_ROUTER_KL_COEFF:-1.0}"
export MOE_EXPANSION_DISTILL_HIDDEN_LAYERS="${MOE_EXPANSION_DISTILL_HIDDEN_LAYERS:-all}"
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.0}"
export MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.0}"

export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-600}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-$TRAIN_ITERS}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export TRAIN_LOG_STEP_TIME_ONLY="${TRAIN_LOG_STEP_TIME_ONLY:-0}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}"
export RUN_INITIAL_PROBE_EVAL="${RUN_INITIAL_PROBE_EVAL:-1}"
export RUN_INITIAL_VALID_EVAL="${RUN_INITIAL_VALID_EVAL:-1}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-48}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export SEED="${SEED:-1234}"
export DIRECT_LOCAL_SAVE="${DIRECT_LOCAL_SAVE:-1}"
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"
export ATTN_LORA_GROUPED_GEMM="${ATTN_LORA_GROUPED_GEMM:-1}"
export MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}"

SAFE_MODE="${MODE//_/-}"
export RUN_ID="${RUN_ID:-g2-shared-router-e8to16-code-expert-init-${SAFE_MODE}-wiki-distill-qkvo-mha-a100-bf16-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$G2_ROOT/code/shared_router_expansion_distill_init/$RUN_ID}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2 FFN+Attention experts - code expert init ${MODE} - wiki distill}"

if [ ! -f "$STAGE1_WEIGHTS_DIR/latest_checkpointed_iteration.txt" ]; then
    echo "[ERROR] missing Wiki shared-router source: $STAGE1_WEIGHTS_DIR" >&2
    exit 1
fi
SOURCE_STEP="$(tr -d '\n\r[:space:]' < "$STAGE1_WEIGHTS_DIR/latest_checkpointed_iteration.txt")"
if [ "$SOURCE_STEP" -lt "$SOURCE_REQUIRED_ITERS" ]; then
    echo "[ERROR] incomplete Wiki shared-router source: step=$SOURCE_STEP required=$SOURCE_REQUIRED_ITERS" >&2
    exit 1
fi
if [ -f "$TRAIN_WEIGHTS/latest_checkpointed_iteration.txt" ]; then
    TARGET_STEP="$(tr -d '\n\r[:space:]' < "$TRAIN_WEIGHTS/latest_checkpointed_iteration.txt")"
    if [ "$TARGET_STEP" = "$TRAIN_ITERS" ]; then
        echo "[SKIP] completed shared-router expansion distill: $RUN_ID"
        exit 0
    fi
fi

echo "[CONFIG] G2 shared-router FFN+Attention pre-Code expansion distill"
echo "[CONFIG] mode=$MOE_EXPANSION_DISTILL_MODE"
echo "[CONFIG] source=$STAGE1_WEIGHTS_DIR"
echo "[CONFIG] target=$TRAIN_WEIGHTS"
echo "[CONFIG] experts: FFN 8->16 + QKVO LoRA 8->16, shared Top-$MOE_ROUTER_TOPK router"
echo "[CONFIG] trainable=new FFN experts + new QKVO LoRA experts + new shared-router rows"
echo "[CONFIG] frozen=old FFN/attention experts + old router rows + dense/shared trunk"
echo "[CONFIG] losses: logit_kl=$OLD_MODEL_KL_COEFF hidden_mse=$MOE_EXPANSION_DISTILL_HIDDEN_MSE_COEFF router_kl=$MOE_EXPANSION_DISTILL_ROUTER_KL_COEFF lm_coeff=$MOE_EXPANSION_DISTILL_LM_LOSS_COEFF aux=$MOE_AUX_LOSS_COEFF z=$MOE_Z_LOSS_COEFF"
echo "[CONFIG] data=$TRAIN_DATASET"

exec bash "$PROJECT_ROOT/scripts/experiment/continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh"
