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
SAFE_MODE="${MODE//_/-}"

# Stage B of the G2 shared-router (FFN + QKVO attention experts) pre-Code
# expert-init experiment:
#   Stage A (run_g2_shared_router_code_expert_distill_init_mha.sh) produced a
#   16-expert checkpoint by distilling the expanded student from the 8-expert
#   Wiki shared-router teacher. Here we take that already-expanded checkpoint and
#   run NORMAL Code training on it, WITHOUT re-expanding: fresh finetune
#   (iteration 0), new FFN experts + new QKVO attention experts + new shared
#   router rows trainable, everything else frozen. This is directly comparable to
#   the random-init baseline (run_g2_shared_router_new_experts_all_router_code_
#   mha.sh with new-experts-and-router-only freeze); the only difference is the
#   starting 16-expert initialization.
export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export G2_ROOT="${G2_ROOT:-$LOCAL_WEIGHTS/a100/mha/g2-checkpoints}"

# Model config source: the 8-expert Wiki shared-router checkpoint (architecture /
# metadata). Weights are actually loaded from the 16-expert distill-init via the
# resume path below.
export STAGE1_WEIGHTS_DIR="${STAGE1_WEIGHTS_DIR:-$G2_ROOT/wiki/g2-top4-e8-ffn352-r256-wiki-shared-router-qkvo-mha-a100-bf16-mb72-1800}"
export SOURCE_REQUIRED_ITERS="${SOURCE_REQUIRED_ITERS:-1800}"

# --- Source weights = Stage A distill-init checkpoint (16 experts) ---
# Rebuild the Stage A run id so the default source path tracks the distill runner
# (Stage A default micro-batch is 48).
export DISTILL_SOURCE_MB="${DISTILL_SOURCE_MB:-48}"
export DISTILL_SOURCE_ITERS="${DISTILL_SOURCE_ITERS:-1800}"
export DISTILL_SOURCE_RUN_ID="${DISTILL_SOURCE_RUN_ID:-g2-shared-router-e8to16-code-expert-init-${SAFE_MODE}-wiki-distill-qkvo-mha-a100-bf16-mb${DISTILL_SOURCE_MB}-${DISTILL_SOURCE_ITERS}}"
export DISTILL_SOURCE_ROOT="${DISTILL_SOURCE_ROOT:-$G2_ROOT/code/shared_router_expansion_distill_init}"
export RESUME_FROM_WEIGHTS="${RESUME_FROM_WEIGHTS:-$DISTILL_SOURCE_ROOT/$DISTILL_SOURCE_RUN_ID}"
# Load the 16-expert distill-init as a fresh finetune (iteration 0, no optimizer
# state, no re-expansion). The resume-from-num-experts boundary keeps experts
# [0:8] (Wiki) frozen and trains experts [8:16] (new) + their router rows.
export RESUME_LOAD_OPTIM="${RESUME_LOAD_OPTIM:-0}"
export RESUME_RESET_ITERATION="${RESUME_RESET_ITERATION:-1}"

if [ ! -f "$STAGE1_WEIGHTS_DIR/latest_checkpointed_iteration.txt" ]; then
    echo "[ERROR] missing Wiki shared-router config source: $STAGE1_WEIGHTS_DIR" >&2
    exit 1
fi
if [ ! -f "$RESUME_FROM_WEIGHTS/latest_checkpointed_iteration.txt" ]; then
    echo "[ERROR] distill-init checkpoint not found: $RESUME_FROM_WEIGHTS" >&2
    echo "        Run Stage A first: run_g2_shared_router_code_expert_distill_init_mha.sh $MODE" >&2
    exit 1
fi

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

# Trainable = new FFN experts + new QKVO attention experts + new shared-router
# rows only (backbone default appends --shared-router-hybrid-train-new-experts-
# and-router-only when neither all-experts nor a partial mask is requested).
export SHARED_ROUTER_HYBRID_TRAIN_ALL_EXPERTS_AND_ROUTER=0
export SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS=0
export SHARED_ROUTER_HYBRID_FREEZE_PREEXISTING_ONLY=0
export SHARED_ROUTER_HYBRID_PARTIAL_FREEZE_MASK=""
export SHARED_ROUTER_HYBRID_TOPK_WITH_ALL_NEW_EXPERTS=0

# Normal Code training: no KL / expansion-distill objectives.
export ENABLE_OLD_MODEL_KL=0
export OLD_MODEL_KL_COEFF=0.0
export OLD_MODEL_KL_TEMPERATURE=1.0
export MOE_EXPANSION_DISTILL_MODE=none
# aux/z loss coeffs intentionally left unset -> inherit model-config defaults,
# identical to the shared-router random-init code baseline.

# Data = code (backbone default). Primary probe=code, secondary probe=wiki, so
# the Wiki-probe trajectory (collapse) is tracked as the secondary metric.
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-72}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-600}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-$TRAIN_ITERS}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}"
export RUN_INITIAL_PROBE_EVAL="${RUN_INITIAL_PROBE_EVAL:-1}"
export RUN_INITIAL_VALID_EVAL="${RUN_INITIAL_VALID_EVAL:-1}"
export SEED="${SEED:-1234}"
export DIRECT_LOCAL_SAVE="${DIRECT_LOCAL_SAVE:-1}"
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"
export ATTN_LORA_GROUPED_GEMM="${ATTN_LORA_GROUPED_GEMM:-1}"
export MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}"

export RUN_ID="${RUN_ID:-g2-shared-router-e8to16-code-from-distill-init-${SAFE_MODE}-qkvo-mha-a100-bf16-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$G2_ROOT/code/shared_router_from_distill_init/$RUN_ID}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2 shared-router - code from ${MODE} distill-init}"

if [ -f "$TRAIN_WEIGHTS/latest_checkpointed_iteration.txt" ]; then
    TARGET_STEP="$(tr -d '\n\r[:space:]' < "$TRAIN_WEIGHTS/latest_checkpointed_iteration.txt")"
    if [ "$TARGET_STEP" = "$TRAIN_ITERS" ]; then
        echo "[SKIP] completed shared-router Stage B: $RUN_ID"
        exit 0
    fi
fi

echo "[CONFIG] G2 shared-router Stage B: Code training from distill-init"
echo "[CONFIG] mode=$MODE"
echo "[CONFIG] config-source(8e wiki)=$STAGE1_WEIGHTS_DIR"
echo "[CONFIG] weights-source(distill-init 16e)=$RESUME_FROM_WEIGHTS"
echo "[CONFIG] target=$TRAIN_WEIGHTS"
echo "[CONFIG] load=fresh-finetune resume, no re-expansion (RESUME_RESET_ITERATION=1, RESUME_LOAD_OPTIM=0)"
echo "[CONFIG] trainable=new FFN experts + new QKVO attention experts + new shared-router rows only"
echo "[CONFIG] iters=$TRAIN_ITERS mb=$MICRO_BATCH_SIZE gbs=$GLOBAL_BATCH_SIZE (baseline-matched, KL off)"

exec bash "$PROJECT_ROOT/scripts/experiment/continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh"
