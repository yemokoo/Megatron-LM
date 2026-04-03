#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export STAGE1_SUBDIR="${STAGE1_SUBDIR:-a100/mha/wiki-shared-router-hybrid-pretrain-local}"
export RUN_ID="${RUN_ID:-e2-wiki-to-code-ffn-attn-lora-single-router-moe-mha-a100-bf16-$(date -u +%Y%m%d-%H%M%S)}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$LOCAL_WEIGHTS/a100/mha/code-from-wiki-shared-router-hybrid-expand-local/$RUN_ID}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-E2 - Wiki to Code FFN+Attn LoRA Single Router MoE MHA}"

exec bash "$PROJECT_ROOT/scripts/experiment/continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh"
