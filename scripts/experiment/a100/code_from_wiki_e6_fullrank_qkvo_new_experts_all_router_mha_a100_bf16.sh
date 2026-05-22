#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export STAGE1_SUBDIR="${STAGE1_SUBDIR:-a100/mha/wiki-shared-router-hybrid-pretrain-local}"
export ATTN_FULL_RANK_LORA_RANK="${ATTN_FULL_RANK_LORA_RANK:-1024}"
export ATTN_FULL_RANK_LORA_ALPHA="${ATTN_FULL_RANK_LORA_ALPHA:-1024}"
export ATTN_FULL_RANK_LORA_TARGETS="${ATTN_FULL_RANK_LORA_TARGETS:-qkvo}"
export SHARED_ROUTER_HYBRID_TRAIN_ALL_EXPERTS_AND_ROUTER=0
export SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS=1
export RUN_ID="${RUN_ID:-e6-wiki-to-code-ffn-attn-fullrank-qkvo-new-experts-all-router-single-router-moe-mha-a100-bf16-$(date -u +%Y%m%d-%H%M%S)}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$LOCAL_WEIGHTS/a100/mha/code-from-wiki-shared-router-hybrid-new-experts-all-router-local/$RUN_ID}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-E6 - Wiki to Code FFN+Attn Full-Rank QKVO New Experts+All Router Single Router MoE MHA}"

exec bash "$PROJECT_ROOT/scripts/experiment/continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh"
