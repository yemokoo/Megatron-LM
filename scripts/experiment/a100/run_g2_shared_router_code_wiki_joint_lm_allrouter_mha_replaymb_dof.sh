#!/bin/bash
set -euo pipefail

A100_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$A100_SCRIPTS_DIR/../../.." && pwd)"
source "$A100_SCRIPTS_DIR/common.sh"

MODE="${1:-${MOE_EXPANSION_DISTILL_MODE_SOURCE:-logits}}"
case "$MODE" in
    logits|logits_hidden|logits_hidden_router)
        ;;
    *)
        echo "usage: $0 [logits|logits_hidden|logits_hidden_router]" >&2
        exit 2
        ;;
esac
SAFE_MODE="${MODE//_/-}"

# Phase 1 for the G2 shared-router hybrid experiment.
# Load an already-expanded 16-expert KD-init checkpoint, then train on Code while
# accumulating a 1:1 Wiki replay LM backward before each optimizer update.
# Code LM trains the newly added FFN/QKVO experts and every shared-router row.
# Wiki replay contributes LM gradients to every router row only; its non-router
# gradients are discarded before the single aggregated optimizer update.
export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export G2_ROOT="${G2_ROOT:-$LOCAL_WEIGHTS/a100/mha/g2-checkpoints}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-$PROJECT_ROOT/.local/models/pythia-12b-tokenizer}"

export STAGE1_WEIGHTS_DIR="${STAGE1_WEIGHTS_DIR:-$G2_ROOT/wiki/g2-top4-e8-ffn352-r256-wiki-shared-router-qkvo-mha-a100-bf16-mb72-1800}"
export SOURCE_REQUIRED_ITERS="${SOURCE_REQUIRED_ITERS:-1800}"

export DISTILL_SOURCE_MB="${DISTILL_SOURCE_MB:-48}"
export DISTILL_SOURCE_ITERS="${DISTILL_SOURCE_ITERS:-600}"
export DISTILL_SOURCE_RUN_ID="${DISTILL_SOURCE_RUN_ID:-g2-shared-router-e8to16-code-expert-init-${SAFE_MODE}-wiki-distill-qkvo-mha-a100-bf16-mb${DISTILL_SOURCE_MB}-${DISTILL_SOURCE_ITERS}}"
export DISTILL_SOURCE_ROOT="${DISTILL_SOURCE_ROOT:-$G2_ROOT/code/shared_router_expansion_distill_init}"
export RESUME_FROM_WEIGHTS="${RESUME_FROM_WEIGHTS:-$DISTILL_SOURCE_ROOT/$DISTILL_SOURCE_RUN_ID}"
export DISTILL_SOURCE_REQUIRED_ITERS="${DISTILL_SOURCE_REQUIRED_ITERS:-$DISTILL_SOURCE_ITERS}"
export RESUME_LOAD_OPTIM=0
export RESUME_RESET_ITERATION=1

export SOURCE_NUM_EXPERTS=8
export NUM_EXPERTS=16
export MOE_ROUTER_TOPK=4
# DoF 스윕: expert 폭을 env 로 개방 (미지정 시 기존 352 그대로)
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export NUM_QUERY_GROUPS=16

export ATTN_LORA_RANK="${ATTN_LORA_RANK:-256}"
export ATTN_LORA_ALPHA="${ATTN_LORA_ALPHA:-$ATTN_LORA_RANK}"
export ATTN_FULL_RANK_LORA_RANK="${ATTN_FULL_RANK_LORA_RANK:-256}"
export ATTN_FULL_RANK_LORA_ALPHA="${ATTN_FULL_RANK_LORA_ALPHA:-$ATTN_FULL_RANK_LORA_RANK}"
export ATTN_FULL_RANK_LORA_TARGETS="${ATTN_FULL_RANK_LORA_TARGETS:-qkvo}"
export ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="${ATTN_FULL_RANK_LORA_ACTIVE_TARGETS:-}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-configs/model/flame-shared-router-hybrid-experts.sh}"

# Old FFN/attention experts remain frozen. Unlike the KD-init stage, phase 1
# trains every shared-router row from the Code + Wiki LM gradients.
export SHARED_ROUTER_HYBRID_TRAIN_ALL_EXPERTS_AND_ROUTER=0
export SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS=1
export SHARED_ROUTER_HYBRID_FREEZE_PREEXISTING_ONLY=0
export SHARED_ROUTER_HYBRID_PARTIAL_FREEZE_MASK=""
export SHARED_ROUTER_HYBRID_TOPK_WITH_ALL_NEW_EXPERTS=0

export TRAIN_DATASET="${TRAIN_DATASET:-$(dataset_dir_for_task code)}"
export DATASET_NAME="${DATASET_NAME:-code_train_with_wiki_joint_replay}"
export DATASET_SOURCE="${DATASET_SOURCE:-Code primary LM + Wiki 1:1 joint replay LM}"
export MOE_JOINT_REPLAY_LM=1
export JOINT_REPLAY_DATASET="${JOINT_REPLAY_DATASET:-$(dataset_dir_for_task wiki)}"
export JOINT_REPLAY_DATA_WEIGHT_MODE="${JOINT_REPLAY_DATA_WEIGHT_MODE:-equal_dataset}"

# Phase 1 is pure LM training; expansion KD is disabled after initialization.
export ENABLE_OLD_MODEL_KL=0
export OLD_MODEL_KL_COEFF=0.0
export OLD_MODEL_KL_TEMPERATURE=1.0
export MOE_EXPANSION_DISTILL_MODE=none
export ROUTER_MEMORY_KL_COEFF=0.0
export ROUTER_MEMORY_INTERVAL=0
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.01}"
export MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.001}"

export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
# Ramp training is an ablation. The canonical 1-phase run uses the scheduled LR
# immediately for both new expert families and for the router.
export MOE_NEW_EXPERT_LR_RAMP_STEPS=0
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-96}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-$TRAIN_ITERS}"
export LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-$((TRAIN_ITERS / 10))}"
export LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.01}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-600}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-$TRAIN_ITERS}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export TRAIN_LOG_STEP_TIME_ONLY="${TRAIN_LOG_STEP_TIME_ONLY:-0}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}"
export PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-25}"
export SECONDARY_PROBE_EVAL_ITERS="${SECONDARY_PROBE_EVAL_ITERS:-25}"
export PROBE_MICRO_BATCH_SIZE="${PROBE_MICRO_BATCH_SIZE:-32}"
export RUN_INITIAL_PROBE_EVAL="${RUN_INITIAL_PROBE_EVAL:-1}"
export RUN_INITIAL_VALID_EVAL="${RUN_INITIAL_VALID_EVAL:-1}"
export PROBE_DATASET="${PROBE_DATASET:-$(probe_dir_for_task code)}"
export PROBE_NAME="${PROBE_NAME:-code_probe}"
export SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET:-$(probe_dir_for_task wiki)}"
export SECONDARY_PROBE_NAME="${SECONDARY_PROBE_NAME:-wiki_probe}"
export PROBE_STEP_OFFSET="${PROBE_STEP_OFFSET:-$((SOURCE_REQUIRED_ITERS + DISTILL_SOURCE_REQUIRED_ITERS))}"
export SECONDARY_PROBE_STEP_OFFSET="${SECONDARY_PROBE_STEP_OFFSET:-$PROBE_STEP_OFFSET}"
export WANDB_STEP_OFFSET="${WANDB_STEP_OFFSET:-$PROBE_STEP_OFFSET}"
export DIRECT_LOCAL_SAVE="${DIRECT_LOCAL_SAVE:-1}"
export NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-1}"
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"
export ATTN_LORA_GROUPED_GEMM="${ATTN_LORA_GROUPED_GEMM:-1}"
export MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}"

export RUN_ID="${RUN_ID:-g2-shared-router-code-wiki-joint-lm-allrouter-${SAFE_MODE}-init-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$G2_ROOT/code/shared_router_joint_lm_replay/$RUN_ID}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2 shared-router Code+Wiki 1-phase all-router}"

if [ "${PLAN_ONLY:-0}" = "1" ]; then
    echo "[PLAN] source=$RESUME_FROM_WEIGHTS"
    echo "[PLAN] target=$TRAIN_WEIGHTS"
    echo "[PLAN] experts=8->16 topk=4 FFN352 QKVO-rank256"
    echo "[PLAN] primary=$TRAIN_DATASET replay=$JOINT_REPLAY_DATASET ratio=1:1"
    echo "[PLAN] Code gradients=new FFN + new QKVO LoRA + all shared-router rows"
    echo "[PLAN] Wiki gradients=all shared-router rows only"
    echo "[PLAN] train_iters=$TRAIN_ITERS new_expert_lr_ramp=off"
    exit 0
fi

resume_tracker="$RESUME_FROM_WEIGHTS/latest_checkpointed_iteration.txt"
if [ ! -f "$resume_tracker" ]; then
    echo "ERROR: shared-router KD-init checkpoint is missing: $resume_tracker" >&2
    exit 1
fi
resume_step="$(tr -d '[:space:]' < "$resume_tracker")"
if [ "$resume_step" != "$DISTILL_SOURCE_REQUIRED_ITERS" ]; then
    echo "ERROR: expected KD-init step $DISTILL_SOURCE_REQUIRED_ITERS, got $resume_step: $RESUME_FROM_WEIGHTS" >&2
    exit 1
fi

for dataset_spec in "Code:$TRAIN_DATASET" "Wiki replay:$JOINT_REPLAY_DATASET"; do
    label="${dataset_spec%%:*}"
    dataset_dir="${dataset_spec#*:}"
    if ! compgen -G "$dataset_dir/*.bin" >/dev/null || ! compgen -G "$dataset_dir/*.idx" >/dev/null; then
        echo "ERROR: missing $label .bin/.idx files: $dataset_dir" >&2
        exit 1
    fi
done

echo "[CONFIG] G2 shared-router phase 1: Code + Wiki 1:1 joint LM"
echo "[CONFIG] source=$RESUME_FROM_WEIGHTS (step $resume_step)"
echo "[CONFIG] target=$TRAIN_WEIGHTS"
echo "[CONFIG] trainable=new FFN experts + new QKVO LoRA experts + all shared-router rows"
echo "[CONFIG] frozen=old FFN/QKVO experts + dense/shared trunk"
echo "[CONFIG] gradients=Code(new FFN + new QKVO + all router) + Wiki(all router only)"
echo "[CONFIG] new-expert LR ramp=off (canonical 1-phase)"
echo "[CONFIG] primary Code=$TRAIN_DATASET"
echo "[CONFIG] replay Wiki=$JOINT_REPLAY_DATASET"

exec bash "$PROJECT_ROOT/scripts/experiment/continual_shared_router_hybrid_replaymb_local_bf16.sh"
