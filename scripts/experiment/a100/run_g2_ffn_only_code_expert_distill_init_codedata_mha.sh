#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# Preserve this script's dir: sourcing common.sh (-> activate_kt_env.sh) clobbers
# SCRIPT_DIR, so keep our own handle for the final exec.
A100_SCRIPTS_DIR="$SCRIPT_DIR"
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

# ---------------------------------------------------------------------------
# CODE-DATA pre-Code expert initialization (ablation variant of the wiki-init
# stage, run_g2_ffn_only_code_expert_distill_init_mha.sh).
#
#   wiki 8-expert teacher -> expand to 16 experts -> distill on CODE data
#
# ONLY ONE thing changes vs the wiki-init stage: the distillation data
# (wiki -> code). Teacher (wiki 8-expert), freeze mask (new FFN experts + new
# router rows only), objective (KL to teacher [+ hidden MSE / router KL]), and
# LM loss coeff = 0 are all IDENTICAL. LM loss is still 0, so this does NOT
# learn code labels; it anchors the expanded model to the wiki teacher's
# function over the CODE input distribution — i.e. it pre-conditions the new
# experts + router on the downstream (code) input statistics without yet
# learning the code task. Probes are kept wiki-primary / code-secondary so the
# result is directly comparable to the wiki-init lineage.
# ---------------------------------------------------------------------------
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

export STAGE_NAME="${STAGE_NAME:-g2_ffn_only_code_expert_${MODE}_codeinit}"
export STAGE_LABEL="${STAGE_LABEL:-g2_ffn_only_code_expert_${MODE}_codeinit}"
export STAGE_DIR_NAME="${STAGE_DIR_NAME:-a100/mha/g2-checkpoints/code/expansion_distill_init}"

export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-8}"
export NUM_EXPERTS="${NUM_EXPERTS:-16}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-scripts/experiment/a100/flame-moe-bf16-no-shared.sh}"

export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-96}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-600}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-$TRAIN_ITERS}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}"
export RUN_INITIAL_PROBE_EVAL="${RUN_INITIAL_PROBE_EVAL:-1}"
export LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND="${LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND:-1}"

# *** THE ONLY EXPERIMENTAL CHANGE vs wiki-init: distill on CODE data ***
# The teacher is still the wiki 8-expert model; only the data over which the
# student is aligned to it changes from wiki to code.
export TRAIN_DATASET="${TRAIN_DATASET:-$(dataset_dir_for_task code)}"
export DATASET_NAME="${DATASET_NAME:-code_train_expansion_distill}"
export DATASET_SOURCE="${DATASET_SOURCE:-Code train data for pre-Code expert initialization distillation (code-init ablation)}"
# Probes kept wiki-primary / code-secondary, identical to the wiki-init lineage,
# so wiki preservation and code behaviour are read on the same metrics.
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
# Distinct "code-distill" tag so this never collides with the wiki-init lineage
# ("wiki-distill") under the shared expansion_distill_init/ directory.
export RUN_ID="${RUN_ID:-g2-ffn-only-e8to16-code-expert-init-${SAFE_MODE}-code-distill-mha-a100-bf16-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$G2_ROOT/code/expansion_distill_init/$RUN_ID}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2 FFN-only - code expert init ${MODE} - CODE distill}"

echo "[CONFIG] G2 FFN-only pre-Code expert initialization distill (CODE-DATA ablation)"
echo "[CONFIG] mode=$MOE_EXPANSION_DISTILL_MODE"
echo "[CONFIG] teacher/source=${SOURCE_WEIGHTS_DIR:-$LOCAL_WEIGHTS/$SOURCE_RUN_SUBDIR/$SOURCE_RUN_ID} (wiki 8-expert)"
echo "[CONFIG] target=$TRAIN_WEIGHTS"
echo "[CONFIG] trainable=new FFN experts + new router rows only"
echo "[CONFIG] losses: logit_kl=$OLD_MODEL_KL_COEFF hidden_mse=$MOE_EXPANSION_DISTILL_HIDDEN_MSE_COEFF router_kl=$MOE_EXPANSION_DISTILL_ROUTER_KL_COEFF lm_coeff=$MOE_EXPANSION_DISTILL_LM_LOSS_COEFF aux=$MOE_AUX_LOSS_COEFF z=$MOE_Z_LOSS_COEFF"
echo "[CONFIG] distill data=CODE  (vs wiki-init which distills on wiki)  -> $TRAIN_DATASET"

exec bash "$A100_SCRIPTS_DIR/run_continual_moe_a100_bf16.sh"
