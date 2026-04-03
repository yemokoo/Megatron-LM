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
export A2_SOURCE_WEIGHTS_DIR="${A2_SOURCE_WEIGHTS_DIR:-$PROJECT_ROOT/.local/weights/a100/mha/wiki-a-moe-bf16/a2-wiki-ffn-moe-mha-a100-bf16-mb96-1800}"
export E2_SOURCE_WEIGHTS_DIR="${E2_SOURCE_WEIGHTS_DIR:-$PROJECT_ROOT/.local/weights/a100/mha/wiki-shared-router-hybrid-pretrain-local/e2-wiki-ffn-attn-lora-single-router-moe-mha-a100-bf16-mb96-1800}"

run_and_pause "2_A2_unfreeze" env \
    WANDB_MODE=offline \
    DIRECT_LOCAL_SAVE=1 \
    RUN_ID=a2-wiki-to-code-ffn-moe-unfreeze-mha-kd1.0-a100-bf16-mb48-1800 \
    TRAIN_WEIGHTS="$PROJECT_ROOT/.local/weights/a100/mha/a-to-b-moe-bf16/a2-wiki-to-code-ffn-moe-unfreeze-mha-kd1.0-a100-bf16-mb48-1800" \
    SOURCE_WEIGHTS_DIR="$A2_SOURCE_WEIGHTS_DIR" \
    WANDB_PROJECT=flame-continual-top2-qv-lora \
    WANDB_EXP_NAME="A2 - Wiki to Code FFN MoE Unfreeze MHA KD1.0 mb48" \
    MICRO_BATCH_SIZE=48 \
    GLOBAL_BATCH_SIZE=2304 \
    TRAIN_ITERS=1800 \
    SOURCE_NUM_EXPERTS=4 \
    NUM_EXPERTS=7 \
    MOE_ROUTER_TOPK=2 \
    OLD_MODEL_KL_COEFF=1.0 \
    OLD_MODEL_KL_TEMPERATURE=1.0 \
    MASTER_PORT=29611 \
    "$SCRIPT_DIR/run_guarded_training.sh" \
    bash "$SCRIPT_DIR/a_to_b_a2_mha_a100_bf16.sh"

run_and_pause "3_A2_freeze" env \
    WANDB_MODE=offline \
    DIRECT_LOCAL_SAVE=1 \
    RUN_ID=a2-wiki-to-code-ffn-moe-freeze-mha-a100-bf16-mb96-1800 \
    TRAIN_WEIGHTS="$PROJECT_ROOT/.local/weights/a100/mha/a-to-b-moe-bf16-freeze/a2-wiki-to-code-ffn-moe-freeze-mha-a100-bf16-mb96-1800" \
    SOURCE_WEIGHTS_DIR="$A2_SOURCE_WEIGHTS_DIR" \
    WANDB_PROJECT=flame-continual-top2-qv-lora \
    WANDB_EXP_NAME="A2 - Wiki to Code FFN MoE Freeze MHA mb96" \
    MICRO_BATCH_SIZE=96 \
    GLOBAL_BATCH_SIZE=2304 \
    TRAIN_ITERS=1800 \
    SOURCE_NUM_EXPERTS=4 \
    NUM_EXPERTS=7 \
    MOE_ROUTER_TOPK=2 \
    MASTER_PORT=29612 \
    "$SCRIPT_DIR/run_guarded_training.sh" \
    bash "$SCRIPT_DIR/a_to_b_freeze_a2_mha_a100_bf16.sh"

run_and_pause "4_E2_wiki" env \
    WANDB_MODE=offline \
    DIRECT_LOCAL_SAVE=1 \
    RUN_ID=e2-wiki-ffn-attn-lora-single-router-moe-mha-a100-bf16-mb96-1800 \
    TRAIN_WEIGHTS="$PROJECT_ROOT/.local/weights/a100/mha/wiki-shared-router-hybrid-pretrain-local/e2-wiki-ffn-attn-lora-single-router-moe-mha-a100-bf16-mb96-1800" \
    WANDB_PROJECT=flame-continual-top2-qv-lora \
    WANDB_EXP_NAME="E2 - Wiki FFN+Attn LoRA Single Router MoE Pretrain MHA mb96" \
    MICRO_BATCH_SIZE=96 \
    GLOBAL_BATCH_SIZE=2304 \
    TRAIN_ITERS=1800 \
    MOE_ROUTER_TOPK=2 \
    ATTN_LORA_RANK=16 \
    ATTN_LORA_ALPHA=16 \
    MASTER_PORT=29574 \
    "$SCRIPT_DIR/run_guarded_training.sh" \
    bash "$SCRIPT_DIR/wiki_e2_mha_a100_bf16.sh"

run_and_pause "5_E2_code" env \
    WANDB_MODE=offline \
    DIRECT_LOCAL_SAVE=1 \
    RUN_ID=e2-wiki-to-code-ffn-attn-lora-single-router-moe-mha-a100-bf16-mb96-1800 \
    TRAIN_WEIGHTS="$PROJECT_ROOT/.local/weights/a100/mha/code-from-wiki-shared-router-hybrid-expand-local/e2-wiki-to-code-ffn-attn-lora-single-router-moe-mha-a100-bf16-mb96-1800" \
    SOURCE_WEIGHTS_DIR="$E2_SOURCE_WEIGHTS_DIR" \
    WANDB_PROJECT=flame-continual-top2-qv-lora \
    WANDB_EXP_NAME="E2 - Wiki to Code FFN+Attn LoRA Single Router MoE MHA mb96" \
    MICRO_BATCH_SIZE=96 \
    GLOBAL_BATCH_SIZE=2304 \
    TRAIN_ITERS=1800 \
    SOURCE_NUM_EXPERTS=4 \
    NUM_EXPERTS=7 \
    MOE_ROUTER_TOPK=2 \
    ATTN_LORA_RANK=16 \
    ATTN_LORA_ALPHA=16 \
    MASTER_PORT=29575 \
    "$SCRIPT_DIR/run_guarded_training.sh" \
    bash "$SCRIPT_DIR/code_from_wiki_e2_mha_a100_bf16.sh"

run_and_pause "6_F_full_rank_lora" env \
    WANDB_MODE=offline \
    DIRECT_LOCAL_SAVE=1 \
    RUN_ID=f1-wiki-to-code-ffn-moe-unfreeze-attn-full-rank-lora-mha-a100-bf16-mb32-1800 \
    TRAIN_WEIGHTS="$PROJECT_ROOT/.local/weights/a100/mha/f-attn-full-rank-lora-bf16/f1-wiki-to-code-ffn-moe-unfreeze-attn-full-rank-lora-mha-a100-bf16-mb32-1800" \
    SOURCE_WEIGHTS_DIR="$A2_SOURCE_WEIGHTS_DIR" \
    WANDB_PROJECT=flame-continual-top2-qv-lora \
    WANDB_EXP_NAME="F - Wiki to Code FFN MoE Unfreeze + Attn Full-Rank LoRA MHA mb32" \
    MICRO_BATCH_SIZE=32 \
    GLOBAL_BATCH_SIZE=2304 \
    TRAIN_ITERS=1800 \
    SOURCE_NUM_EXPERTS=4 \
    NUM_EXPERTS=7 \
    MOE_ROUTER_TOPK=2 \
    OLD_MODEL_KL_COEFF=1.0 \
    OLD_MODEL_KL_TEMPERATURE=1.0 \
    ATTN_FULL_RANK_LORA_RANK=1024 \
    ATTN_FULL_RANK_LORA_ALPHA=1024 \
    MASTER_PORT=29613 \
    "$SCRIPT_DIR/run_guarded_training.sh" \
    bash "$SCRIPT_DIR/code_from_wiki_f1_attn_full_rank_lora_mha_a100_bf16.sh"

echo "[ALL DONE] $(date)"
