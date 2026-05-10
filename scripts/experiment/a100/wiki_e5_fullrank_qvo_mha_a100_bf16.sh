#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
export ATTN_LORA_RANK="${ATTN_LORA_RANK:-1024}"
export ATTN_LORA_ALPHA="${ATTN_LORA_ALPHA:-$ATTN_LORA_RANK}"
export ATTN_LORA_INCLUDE_PROJ="${ATTN_LORA_INCLUDE_PROJ:-1}"
export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export RUN_ID="${RUN_ID:-e5-wiki-ffn-attn-lora-qvo-r${ATTN_LORA_RANK}-single-router-moe-mha-a100-bf16-$(date -u +%Y%m%d-%H%M%S)}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$LOCAL_WEIGHTS/a100/mha/wiki-shared-router-hybrid-pretrain-local/$RUN_ID}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-E5 - FullRank QVO LoRA Expert - wiki}"

exec bash "$SCRIPT_DIR/pretrain_wiki_shared_router_hybrid_local_bf16.sh"
