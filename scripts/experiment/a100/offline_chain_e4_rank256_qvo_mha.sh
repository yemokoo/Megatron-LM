#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

run_and_pause() {
    local name="$1"
    shift
    echo "[START] $name $(date)"
    "$@"
    echo "[END] $name $(date)"
    sleep 300
}

export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
export E4_ATTN_LORA_RANK="${E4_ATTN_LORA_RANK:-16}"
export E4_ATTN_LORA_ALPHA="${E4_ATTN_LORA_ALPHA:-$E4_ATTN_LORA_RANK}"
export E4_WIKI_MICRO_BATCH_SIZE="${E4_WIKI_MICRO_BATCH_SIZE:-96}"
export E4_CODE_MICRO_BATCH_SIZE="${E4_CODE_MICRO_BATCH_SIZE:-48}"
export E4_WIKI_RUN_ID="${E4_WIKI_RUN_ID:-e4-wiki-ffn-attn-lora-qvo-r${E4_ATTN_LORA_RANK}-single-router-moe-mha-a100-bf16-mb${E4_WIKI_MICRO_BATCH_SIZE}-1800}"
export E4_CODE_RUN_ID="${E4_CODE_RUN_ID:-e4-wiki-to-code-ffn-attn-lora-qvo-r${E4_ATTN_LORA_RANK}-single-router-moe-mha-a100-bf16-mb${E4_CODE_MICRO_BATCH_SIZE}-1800}"
export E4_WIKI_TRAIN_WEIGHTS="${E4_WIKI_TRAIN_WEIGHTS:-$PROJECT_ROOT/.local/weights/a100/mha/wiki-shared-router-hybrid-pretrain-local/$E4_WIKI_RUN_ID}"
export E4_CODE_TRAIN_WEIGHTS="${E4_CODE_TRAIN_WEIGHTS:-$PROJECT_ROOT/.local/weights/a100/mha/code-from-wiki-shared-router-hybrid-expand-local/$E4_CODE_RUN_ID}"
export E4_SOURCE_WEIGHTS_DIR="${E4_SOURCE_WEIGHTS_DIR:-$E4_WIKI_TRAIN_WEIGHTS}"

run_and_pause "1_E4_wiki" env \
    WANDB_MODE=offline \
    DIRECT_LOCAL_SAVE=1 \
    RUN_ID="$E4_WIKI_RUN_ID" \
    TRAIN_WEIGHTS="$E4_WIKI_TRAIN_WEIGHTS" \
    WANDB_PROJECT=flame-continual-top2-qv-lora \
    WANDB_EXP_NAME="E4 - rank${E4_ATTN_LORA_RANK} qvo - wiki" \
    MICRO_BATCH_SIZE="$E4_WIKI_MICRO_BATCH_SIZE" \
    GLOBAL_BATCH_SIZE=2304 \
    TRAIN_ITERS=1800 \
    MOE_ROUTER_TOPK=2 \
    ATTN_LORA_RANK="$E4_ATTN_LORA_RANK" \
    ATTN_LORA_ALPHA="$E4_ATTN_LORA_ALPHA" \
    ATTN_LORA_INCLUDE_PROJ=1 \
    MASTER_PORT=29631 \
    "$SCRIPT_DIR/run_guarded_training.sh" \
    bash "$SCRIPT_DIR/wiki_e2_mha_a100_bf16.sh"

run_and_pause "2_E4_code" env \
    WANDB_MODE=offline \
    DIRECT_LOCAL_SAVE=1 \
    RUN_ID="$E4_CODE_RUN_ID" \
    TRAIN_WEIGHTS="$E4_CODE_TRAIN_WEIGHTS" \
    SOURCE_WEIGHTS_DIR="$E4_SOURCE_WEIGHTS_DIR" \
    WANDB_PROJECT=flame-continual-top2-qv-lora \
    WANDB_EXP_NAME="E4 - rank${E4_ATTN_LORA_RANK} qvo - wiki to code" \
    MICRO_BATCH_SIZE="$E4_CODE_MICRO_BATCH_SIZE" \
    GLOBAL_BATCH_SIZE=2304 \
    TRAIN_ITERS=1800 \
    SOURCE_NUM_EXPERTS=4 \
    NUM_EXPERTS=7 \
    MOE_ROUTER_TOPK=2 \
    ATTN_LORA_RANK="$E4_ATTN_LORA_RANK" \
    ATTN_LORA_ALPHA="$E4_ATTN_LORA_ALPHA" \
    ATTN_LORA_INCLUDE_PROJ=1 \
    MASTER_PORT=29632 \
    "$SCRIPT_DIR/run_guarded_training.sh" \
    bash "$SCRIPT_DIR/code_from_wiki_e2_mha_a100_bf16.sh"

echo "[ALL DONE] $(date)"
