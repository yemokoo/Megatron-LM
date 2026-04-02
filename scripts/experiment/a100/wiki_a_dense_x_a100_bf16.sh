#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export TASK_NAME=wiki
export TASK_LABEL="${TASK_LABEL:-wiki_a_dense_x}"
export RUN_ID="${RUN_ID:-wiki-a-moe-dense-x-a100-bf16-$(date -u +%Y%m%d-%H%M%S)}"

# Remove the dense first layer and make every transformer layer an MoE layer.
export MOE_LAYER_FREQ="${MOE_LAYER_FREQ:-[1]*9}"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$LOCAL_WEIGHTS/a100/additional/wiki-a-moe-dense-x-bf16/$RUN_ID}"

# Keep the same workspace/project convention while making the run name explicit.
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-A-moe-wiki-dense-x-top${MOE_ROUTER_TOPK:-2}-mb${MICRO_BATCH_SIZE:-8}-${TRAIN_ITERS:-1800}}"
export STAGE_NAME="${STAGE_NAME:-wiki_a_dense_x}"

exec bash "$SCRIPT_DIR/run_base_moe_a100_bf16.sh"
