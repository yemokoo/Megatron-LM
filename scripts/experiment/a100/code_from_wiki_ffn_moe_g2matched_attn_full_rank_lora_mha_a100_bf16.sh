#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# G2-matched FFN-MoE continual baseline with a single dense full-rank LoRA branch
# on attention. Shared/base attention weights stay frozen; LoRA params are trainable.
export SOURCE_TASK=wiki
export TARGET_TASK=code
export FREEZE_SHARED=1
export TRAIN_ATTENTION_WITH_NEW_EXPERTS=0
export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"

export SOURCE_RUN_SUBDIR="${SOURCE_RUN_SUBDIR:-a100/mha/wiki-a-moe-g2matched-bf16}"
export STAGE_DIR_NAME="${STAGE_DIR_NAME:-a100/mha/g2matched-ffn-moe-attn-full-rank-lora-bf16}"
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
export ENABLE_OLD_MODEL_KL=0
export OLD_MODEL_KL_COEFF=0.0
export OLD_MODEL_KL_TEMPERATURE=1.0

export ATTN_FULL_RANK_LORA_RANK="${ATTN_FULL_RANK_LORA_RANK:-1024}"
export ATTN_FULL_RANK_LORA_ALPHA="${ATTN_FULL_RANK_LORA_ALPHA:-$ATTN_FULL_RANK_LORA_RANK}"
export ATTN_FULL_RANK_LORA_TARGETS="${ATTN_FULL_RANK_LORA_TARGETS:-qkvo}"
export ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="${ATTN_FULL_RANK_LORA_ACTIVE_TARGETS:-}"

export RUN_ID="${RUN_ID:-g2matched-top4-e8to16-ffn352-wiki-to-code-ffn-moe-attn-fullrank-qkvo-r${ATTN_FULL_RANK_LORA_RANK}-mha-a100-bf16-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$LOCAL_WEIGHTS/$STAGE_DIR_NAME/$RUN_ID}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2-matched FFN MoE top4 e8to16 ffn352 attn full-rank LoRA qkvo r${ATTN_FULL_RANK_LORA_RANK} - wiki to code}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-configs/model/flame-attn-full-rank-lora.sh}"

exec bash "$SCRIPT_DIR/run_continual_moe_a100_bf16.sh"
