#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# FFN-only MoE wiki source model matched to the G2 shared-router granularity:
# top-k 4, 8 source experts, expert FFN hidden size 352.
export TASK_NAME=wiki
export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"

export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export NUM_EXPERTS="${NUM_EXPERTS:-8}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-128}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-$TRAIN_ITERS}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-$TRAIN_ITERS}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}"

export STAGE_NAME="${STAGE_NAME:-wiki_a_g2matched_ffn_moe}"
export RUN_ID="${RUN_ID:-g2matched-top4-e8-ffn352-wiki-ffn-moe-mha-a100-bf16-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$LOCAL_WEIGHTS/a100/mha/wiki-a-moe-g2matched-bf16/$RUN_ID}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2-matched FFN MoE top4 e8 ffn352 - wiki}"

exec bash "$SCRIPT_DIR/run_base_moe_a100_bf16.sh"
