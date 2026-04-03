#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export SOURCE_TASK=wiki
export TARGET_TASK=code
export FREEZE_SHARED=0
export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
export ATTN_FULL_RANK_LORA_RANK="${ATTN_FULL_RANK_LORA_RANK:-1024}"
export ATTN_FULL_RANK_LORA_ALPHA="${ATTN_FULL_RANK_LORA_ALPHA:-1024}"
export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export SOURCE_RUN_SUBDIR="${SOURCE_RUN_SUBDIR:-a100/mha/wiki-a-moe-bf16}"
export STAGE_DIR_NAME="${STAGE_DIR_NAME:-a100/mha/f-attn-full-rank-lora-bf16}"
export RUN_ID="${RUN_ID:-f1-wiki-to-code-ffn-moe-unfreeze-attn-full-rank-lora-mha-a100-bf16-$(date -u +%Y%m%d-%H%M%S)}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-F - Wiki to Code FFN MoE Unfreeze + Attn Full-Rank LoRA MHA}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-configs/model/flame-attn-full-rank-lora.sh}"

exec bash "$SCRIPT_DIR/run_continual_moe_a100_bf16.sh"
