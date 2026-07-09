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

# Pre-code expert initialization for the FFN-only G2 baseline:
#   wiki 8-expert teacher -> expand to 16 experts -> distill on wiki data
# The resulting checkpoint is intended to be used as the source for normal Code
# training, so this stage uses only teacher-student alignment losses.
export SOURCE_TASK=wiki
export TARGET_TASK=code
export FREEZE_SHARED=1
export TRAIN_ATTENTION_WITH_NEW_EXPERTS=0
export FREEZE_DENSE_ATTENTION_LORA_WITH_NEW_EXPERTS=1
export TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY=1

export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export G2_ROOT="${G2_ROOT:-$LOCAL_WEIGHTS/a100/mha/g2-checkpoints}"

export SOURCE_RUN_SUBDIR="${SOURCE_RUN_SUBDIR:-a100/mha/wiki-a-moe-g2matched-bf16}"
export SOURCE_RUN_ID="${SOURCE_RUN_ID:-g2matched-top4-e8-ffn352-wiki-ffn-moe-mha-a100-bf16-mb128-1800}"
export SOURCE_WEIGHTS_DIR="${SOURCE_WEIGHTS_DIR:-}"
if [ -z "$SOURCE_WEIGHTS_DIR" ]; then
    REGISTRY_SOURCE_WEIGHTS_DIR="$G2_ROOT/wiki/$SOURCE_RUN_ID"
    if [ -d "$REGISTRY_SOURCE_WEIGHTS_DIR" ]; then
        export SOURCE_WEIGHTS_DIR="$REGISTRY_SOURCE_WEIGHTS_DIR"
    fi
fi

export STAGE_NAME="${STAGE_NAME:-g2_ffn_only_code_expert_${MODE}_init}"
export STAGE_LABEL="${STAGE_LABEL:-g2_ffn_only_code_expert_${MODE}_init}"
export STAGE_DIR_NAME="${STAGE_DIR_NAME:-a100/mha/g2-checkpoints/code/expansion_distill_init}"

export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-8}"
export NUM_EXPERTS="${NUM_EXPERTS:-16}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-scripts/experiment/a100/flame-moe-bf16-no-shared.sh}"

export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-96}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export TRAIN_ITERS="${TRAIN_ITERS:-300}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-$TRAIN_ITERS}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-$TRAIN_ITERS}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}"
export RUN_INITIAL_PROBE_EVAL="${RUN_INITIAL_PROBE_EVAL:-1}"
export LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND="${LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND:-1}"

# Distillation data is intentionally Wiki, not Code: this aligns the expanded
# 16-expert student to the 8-expert Wiki teacher before Code adaptation starts.
export TRAIN_DATASET="${TRAIN_DATASET:-$(dataset_dir_for_task wiki)}"
export DATASET_NAME="${DATASET_NAME:-wiki_train_expansion_distill}"
export DATASET_SOURCE="${DATASET_SOURCE:-Wiki train data for pre-Code expert initialization distillation}"
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

SAFE_MODE="${MODE//_/-}"
export RUN_ID="${RUN_ID:-g2-ffn-only-e8to16-code-expert-init-${SAFE_MODE}-wiki-distill-mha-a100-bf16-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$G2_ROOT/code/expansion_distill_init/$RUN_ID}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2 FFN-only - code expert init ${MODE} - wiki distill}"

echo "[CONFIG] G2 FFN-only pre-Code expert initialization distill"
echo "[CONFIG] mode=$MOE_EXPANSION_DISTILL_MODE"
echo "[CONFIG] source=${SOURCE_WEIGHTS_DIR:-$LOCAL_WEIGHTS/$SOURCE_RUN_SUBDIR/$SOURCE_RUN_ID}"
echo "[CONFIG] target=$TRAIN_WEIGHTS"
echo "[CONFIG] trainable=new FFN experts + new router rows only"
echo "[CONFIG] losses: logit_kl=$OLD_MODEL_KL_COEFF hidden_mse=$MOE_EXPANSION_DISTILL_HIDDEN_MSE_COEFF router_kl=$MOE_EXPANSION_DISTILL_ROUTER_KL_COEFF lm_coeff=$MOE_EXPANSION_DISTILL_LM_LOSS_COEFF aux=$MOE_AUX_LOSS_COEFF z=$MOE_Z_LOSS_COEFF"
echo "[CONFIG] data=$TRAIN_DATASET"

exec bash "$SCRIPT_DIR/run_continual_moe_a100_bf16.sh"
