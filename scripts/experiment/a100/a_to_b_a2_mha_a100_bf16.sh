#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export SOURCE_TASK=wiki
export TARGET_TASK=code
export FREEZE_SHARED=0
export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export SOURCE_RUN_SUBDIR="${SOURCE_RUN_SUBDIR:-a100/mha/wiki-a-moe-bf16}"
export STAGE_DIR_NAME="${STAGE_DIR_NAME:-a100/mha/a-to-b-moe-bf16}"
export RUN_ID="${RUN_ID:-a2-wiki-to-code-ffn-moe-unfreeze-mha-a100-bf16-$(date -u +%Y%m%d-%H%M%S)}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-A2 - Wiki to Code FFN MoE Unfreeze MHA}"

exec bash "$SCRIPT_DIR/run_continual_moe_a100_bf16.sh"
