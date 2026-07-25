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
SAFE_MODE="${MODE//_/-}"

# Stage B of the FFN-only pre-Code expert-init experiment:
#   Stage A (run_g2_ffn_only_code_expert_distill_init_mha.sh) produced a 16-expert
#   checkpoint by distilling the expanded student from the 8-expert Wiki teacher.
#   Here we take that already-expanded checkpoint and run NORMAL Code training on
#   it, WITHOUT re-expanding: fresh finetune (iteration 0), new FFN experts + new
#   router rows trainable, everything else frozen. This is directly comparable to
#   the random-init baseline
#   (code_from_wiki_ffn_moe_g2matched_attn_freeze_mha_a100_bf16.sh); the only
#   difference is the starting 16-expert initialization.
export SOURCE_TASK=wiki
export TARGET_TASK=code
export FREEZE_SHARED=1
export TRAIN_ATTENTION_WITH_NEW_EXPERTS=0
export FREEZE_DENSE_ATTENTION_LORA_WITH_NEW_EXPERTS=1
export TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY=1

# Load the already-expanded distill-init checkpoint as a fresh finetune (no
# re-expansion). SOURCE_NUM_EXPERTS is the freeze boundary: experts [0:8] (Wiki)
# stay frozen, experts [8:16] (new) + their router rows train.
export LOAD_EXPANDED_SOURCE=1

export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export G2_ROOT="${G2_ROOT:-$LOCAL_WEIGHTS/a100/mha/g2-checkpoints}"

# --- Source = Stage A distill-init checkpoint (16 experts) ---
# Rebuild the Stage A run id so the default source path tracks the distill runner.
export DISTILL_SOURCE_MB="${DISTILL_SOURCE_MB:-96}"
export DISTILL_SOURCE_ITERS="${DISTILL_SOURCE_ITERS:-1800}"
export DISTILL_SOURCE_RUN_ID="${DISTILL_SOURCE_RUN_ID:-g2-ffn-only-e8to16-code-expert-init-${SAFE_MODE}-wiki-distill-mha-a100-bf16-mb${DISTILL_SOURCE_MB}-${DISTILL_SOURCE_ITERS}}"
export SOURCE_RUN_SUBDIR="${SOURCE_RUN_SUBDIR:-a100/mha/g2-checkpoints/code/expansion_distill_init}"
export SOURCE_RUN_ID="${SOURCE_RUN_ID:-$DISTILL_SOURCE_RUN_ID}"
export SOURCE_WEIGHTS_DIR="${SOURCE_WEIGHTS_DIR:-$G2_ROOT/code/expansion_distill_init/$SOURCE_RUN_ID}"
if [ ! -f "$SOURCE_WEIGHTS_DIR/latest_checkpointed_iteration.txt" ]; then
    echo "[ERROR] distill-init checkpoint not found: $SOURCE_WEIGHTS_DIR" >&2
    echo "        Run Stage A first: run_g2_ffn_only_code_expert_distill_init_mha.sh $MODE" >&2
    exit 1
fi

export STAGE_DIR_NAME="${STAGE_DIR_NAME:-a100/mha/g2-checkpoints/code/from_distill_init}"

export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-8}"
export NUM_EXPERTS="${NUM_EXPERTS:-16}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-scripts/experiment/a100/flame-moe-bf16-no-shared.sh}"

# Match the random-init baseline exactly for a fair comparison.
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-96}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-600}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-$TRAIN_ITERS}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}"
export RUN_INITIAL_PROBE_EVAL="${RUN_INITIAL_PROBE_EVAL:-1}"
# No expansion happens here, so there is no separate pre-expand source baseline.
export LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND="${LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND:-0}"

# Normal Code training: no KL / expansion-distill objectives.
export ENABLE_OLD_MODEL_KL=0
export OLD_MODEL_KL_COEFF=0.0
export MOE_EXPANSION_DISTILL_MODE=none
# aux/z loss coeffs intentionally left unset -> inherit model-config defaults
# (0.01 / 0.001), identical to the random-init baseline.

export RUN_ID="${RUN_ID:-g2-ffn-only-e8to16-code-from-distill-init-${SAFE_MODE}-mha-a100-bf16-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$G2_ROOT/code/from_distill_init/$RUN_ID}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2 FFN-only - code from ${MODE} distill-init}"

echo "[CONFIG] G2 FFN-only Stage B: Code training from distill-init"
echo "[CONFIG] mode=$MODE"
echo "[CONFIG] source(distill-init 16e)=$SOURCE_WEIGHTS_DIR"
echo "[CONFIG] target=$TRAIN_WEIGHTS"
echo "[CONFIG] load=fresh-finetune, no re-expansion (LOAD_EXPANDED_SOURCE=1)"
echo "[CONFIG] trainable=new FFN experts + new router rows only"
echo "[CONFIG] iters=$TRAIN_ITERS mb=$MICRO_BATCH_SIZE gbs=$GLOBAL_BATCH_SIZE (baseline-matched, KL off)"

exec bash "$A100_SCRIPTS_DIR/run_continual_moe_a100_bf16.sh"
