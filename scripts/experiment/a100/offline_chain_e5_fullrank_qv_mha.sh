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
export E5_ATTN_LORA_RANK="${E5_ATTN_LORA_RANK:-1024}"
export E5_ATTN_LORA_ALPHA="${E5_ATTN_LORA_ALPHA:-$E5_ATTN_LORA_RANK}"
export E5_WIKI_MICRO_BATCH_SIZE="${E5_WIKI_MICRO_BATCH_SIZE:-64}"
export E5_CODE_MICRO_BATCH_SIZE="${E5_CODE_MICRO_BATCH_SIZE:-64}"
export E5_WIKI_RUN_ID="${E5_WIKI_RUN_ID:-e5-wiki-ffn-attn-lora-qv-r${E5_ATTN_LORA_RANK}-single-router-moe-mha-a100-bf16-mb${E5_WIKI_MICRO_BATCH_SIZE}-1800}"
export E5_CODE_RUN_ID="${E5_CODE_RUN_ID:-e5-wiki-to-code-ffn-attn-lora-qv-r${E5_ATTN_LORA_RANK}-single-router-moe-mha-a100-bf16-mb${E5_CODE_MICRO_BATCH_SIZE}-1800}"
export E5_WIKI_TRAIN_WEIGHTS="${E5_WIKI_TRAIN_WEIGHTS:-$PROJECT_ROOT/.local/weights/a100/mha/wiki-shared-router-hybrid-pretrain-local/$E5_WIKI_RUN_ID}"
export E5_CODE_TRAIN_WEIGHTS="${E5_CODE_TRAIN_WEIGHTS:-$PROJECT_ROOT/.local/weights/a100/mha/code-from-wiki-shared-router-hybrid-expand-local/$E5_CODE_RUN_ID}"
export E5_SOURCE_WEIGHTS_DIR="${E5_SOURCE_WEIGHTS_DIR:-$E5_WIKI_TRAIN_WEIGHTS}"

run_and_pause "1_E5_wiki" env \
    WANDB_MODE=offline \
    DIRECT_LOCAL_SAVE=1 \
    RUN_ID="$E5_WIKI_RUN_ID" \
    TRAIN_WEIGHTS="$E5_WIKI_TRAIN_WEIGHTS" \
    WANDB_PROJECT=flame-continual-top2-qv-lora \
    WANDB_EXP_NAME="E5 - fullrank qv - wiki" \
    MICRO_BATCH_SIZE="$E5_WIKI_MICRO_BATCH_SIZE" \
    GLOBAL_BATCH_SIZE=2304 \
    TRAIN_ITERS=1800 \
    MOE_ROUTER_TOPK=2 \
    ATTN_LORA_RANK="$E5_ATTN_LORA_RANK" \
    ATTN_LORA_ALPHA="$E5_ATTN_LORA_ALPHA" \
    MASTER_PORT=29661 \
    "$SCRIPT_DIR/run_guarded_training.sh" \
    bash "$SCRIPT_DIR/wiki_e5_fullrank_qv_mha_a100_bf16.sh"

run_and_pause "2_E5_code" env \
    WANDB_MODE=offline \
    DIRECT_LOCAL_SAVE=1 \
    RUN_ID="$E5_CODE_RUN_ID" \
    TRAIN_WEIGHTS="$E5_CODE_TRAIN_WEIGHTS" \
    SOURCE_WEIGHTS_DIR="$E5_SOURCE_WEIGHTS_DIR" \
    WANDB_PROJECT=flame-continual-top2-qv-lora \
    WANDB_EXP_NAME="E5 - fullrank qv - wiki to code" \
    MICRO_BATCH_SIZE="$E5_CODE_MICRO_BATCH_SIZE" \
    GLOBAL_BATCH_SIZE=2304 \
    TRAIN_ITERS=1800 \
    SOURCE_NUM_EXPERTS=4 \
    NUM_EXPERTS=7 \
    MOE_ROUTER_TOPK=2 \
    ATTN_LORA_RANK="$E5_ATTN_LORA_RANK" \
    ATTN_LORA_ALPHA="$E5_ATTN_LORA_ALPHA" \
    MASTER_PORT=29662 \
    "$SCRIPT_DIR/run_guarded_training.sh" \
    bash "$SCRIPT_DIR/code_from_wiki_e5_fullrank_qv_mha_a100_bf16.sh"

echo "[ALL DONE] $(date)"
