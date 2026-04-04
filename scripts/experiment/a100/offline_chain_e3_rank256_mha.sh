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
export E3_SOURCE_WEIGHTS_DIR="${E3_SOURCE_WEIGHTS_DIR:-$PROJECT_ROOT/.local/weights/a100/mha/wiki-shared-router-hybrid-pretrain-local/e3-wiki-ffn-attn-lora-r256-single-router-moe-mha-a100-bf16-mb96-1800}"

run_and_pause "1_E3_wiki" env \
    WANDB_MODE=offline \
    DIRECT_LOCAL_SAVE=1 \
    RUN_ID=e3-wiki-ffn-attn-lora-r256-single-router-moe-mha-a100-bf16-mb96-1800 \
    TRAIN_WEIGHTS="$PROJECT_ROOT/.local/weights/a100/mha/wiki-shared-router-hybrid-pretrain-local/e3-wiki-ffn-attn-lora-r256-single-router-moe-mha-a100-bf16-mb96-1800" \
    WANDB_PROJECT=flame-continual-top2-qv-lora \
    WANDB_EXP_NAME="E3 - rank256 wiki" \
    MICRO_BATCH_SIZE=96 \
    GLOBAL_BATCH_SIZE=2304 \
    TRAIN_ITERS=1800 \
    MOE_ROUTER_TOPK=2 \
    ATTN_LORA_RANK=256 \
    ATTN_LORA_ALPHA=256 \
    MASTER_PORT=29621 \
    "$SCRIPT_DIR/run_guarded_training.sh" \
    bash "$SCRIPT_DIR/wiki_e2_mha_a100_bf16.sh"

run_and_pause "2_E3_code" env \
    WANDB_MODE=offline \
    DIRECT_LOCAL_SAVE=1 \
    RUN_ID=e3-wiki-to-code-ffn-attn-lora-r256-single-router-moe-mha-a100-bf16-mb48-1800 \
    TRAIN_WEIGHTS="$PROJECT_ROOT/.local/weights/a100/mha/code-from-wiki-shared-router-hybrid-expand-local/e3-wiki-to-code-ffn-attn-lora-r256-single-router-moe-mha-a100-bf16-mb48-1800" \
    SOURCE_WEIGHTS_DIR="$E3_SOURCE_WEIGHTS_DIR" \
    WANDB_PROJECT=flame-continual-top2-qv-lora \
    WANDB_EXP_NAME="E3 - rank256 wiki to code" \
    MICRO_BATCH_SIZE=48 \
    GLOBAL_BATCH_SIZE=2304 \
    TRAIN_ITERS=1800 \
    SOURCE_NUM_EXPERTS=4 \
    NUM_EXPERTS=7 \
    MOE_ROUTER_TOPK=2 \
    ATTN_LORA_RANK=256 \
    ATTN_LORA_ALPHA=256 \
    MASTER_PORT=29622 \
    "$SCRIPT_DIR/run_guarded_training.sh" \
    bash "$SCRIPT_DIR/code_from_wiki_e2_mha_a100_bf16.sh"

echo "[ALL DONE] $(date)"
