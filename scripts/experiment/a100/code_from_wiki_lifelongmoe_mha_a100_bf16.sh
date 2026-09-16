#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# G2-matched FFN-MoE continual baseline.
# Only the newly added FFN experts and router rows are trained; attention stays frozen.
export SOURCE_TASK=wiki
export TARGET_TASK=code
# Lifelong-MoE trains the shared dense/attention parameters (theta_d) and
# regularizes them with Online L2 instead of freezing them.
export FREEZE_SHARED=0
export TRAIN_ATTENTION_WITH_NEW_EXPERTS=0
export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"

export SOURCE_RUN_SUBDIR="${SOURCE_RUN_SUBDIR:-a100/mha/wiki-a-moe-g2matched-bf16}"
export STAGE_DIR_NAME="${STAGE_DIR_NAME:-a100/mha/lifelongmoe}"
export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-8}"
export NUM_EXPERTS="${NUM_EXPERTS:-16}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-96}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-$TRAIN_ITERS}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-$TRAIN_ITERS}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}"
export LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND="${LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND:-0}"
export RUN_INITIAL_PROBE_EVAL="${RUN_INITIAL_PROBE_EVAL:-1}"
# Lifelong-MoE regularization = output-level knowledge distillation against
# the frozen previous-task model:  L = L_Perp + lambda_KL * L_KL  (paper Eq. 4).
# The teacher is the pre-expansion checkpoint the stage script already loads.
export ENABLE_OLD_MODEL_KL="${ENABLE_OLD_MODEL_KL:-1}"
export OLD_MODEL_KL_COEFF="${LIFELONG_KL_COEFF:-1.0}"
export OLD_MODEL_KL_TEMPERATURE="${LIFELONG_KL_TEMPERATURE:-1.0}"
# Online L2 is the paper's *baseline* regularizer, kept available for ablation.
export LIFELONG_L2_COEFF="${LIFELONG_L2_COEFF:-0.0}"

export RUN_ID="${RUN_ID:-lifelongmoe-e8to16-code-kl${OLD_MODEL_KL_COEFF}-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$LOCAL_WEIGHTS/$STAGE_DIR_NAME/$RUN_ID}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2-matched FFN MoE top4 e8to16 ffn352 attn-freeze - wiki to code}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-scripts/experiment/a100/flame-moe-bf16-no-shared-groupedgemm.sh}"

export TRAIN_ENTRY="${TRAIN_ENTRY:-pretrain_gpt_lifelongmoe.py}"
exec bash "$SCRIPT_DIR/run_continual_moe_a100_bf16_lprjoint.sh"
