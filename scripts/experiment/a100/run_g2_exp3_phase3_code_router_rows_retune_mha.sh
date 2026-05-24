#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_LOG_CHECKPOINTS="${WANDB_LOG_CHECKPOINTS:-0}"
export RETUNE_ITERS="${RETUNE_ITERS:-1800}"
export SOURCE_REQUIRED_ITERS="${SOURCE_REQUIRED_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-600}"
export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-1}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export TRAIN_LOG_STEP_TIME_ONLY="${TRAIN_LOG_STEP_TIME_ONLY:-0}"
export TRAIN_ROUTER_USAGE_LOG_INTERVAL="${TRAIN_ROUTER_USAGE_LOG_INTERVAL:-1}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-48}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export G2_ROOT="${G2_ROOT:-$PROJECT_ROOT/.local/weights/a100/mha/g2-checkpoints}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-/home/work/.cache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6}"

export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-8}"
export NUM_EXPERTS="${NUM_EXPERTS:-16}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.0}"
export MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.0}"
export ATTN_LORA_RANK="${ATTN_LORA_RANK:-256}"
export ATTN_LORA_ALPHA="${ATTN_LORA_ALPHA:-$ATTN_LORA_RANK}"
export ATTN_FULL_RANK_LORA_RANK="${ATTN_FULL_RANK_LORA_RANK:-256}"
export ATTN_FULL_RANK_LORA_ALPHA="${ATTN_FULL_RANK_LORA_ALPHA:-$ATTN_FULL_RANK_LORA_RANK}"
export ATTN_FULL_RANK_LORA_TARGETS="${ATTN_FULL_RANK_LORA_TARGETS:-qkvo}"
export ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="${ATTN_FULL_RANK_LORA_ACTIVE_TARGETS:-}"
export ROUTER_MEMORY_KL_COEFF=0.0
export ROUTER_MEMORY_INTERVAL=0
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"
export ATTN_LORA_GROUPED_GEMM="${ATTN_LORA_GROUPED_GEMM:-1}"
export MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}"

export SRC_RUN_ID="${SRC_RUN_ID:-g2-exp3-top4-e8to16-ffn352-r256-wiki-to-code-routerkd-kl10p0-top4plus-allcode-newexperts-allrouter-mha-a100-bf16-mb48-1800}"
export SRC_WEIGHTS="${SRC_WEIGHTS:-$G2_ROOT/code/exp3/$SRC_RUN_ID}"
export RUN_ID="${RUN_ID:-g2-exp3-phase3-code-router-rows-only-retune-from-routerkd-kl10p0-top4plus-allcode-normal-top4-no-reinit-mb${MICRO_BATCH_SIZE}-${RETUNE_ITERS}}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$G2_ROOT/code/phase3/$RUN_ID}"
export MASTER_PORT="${MASTER_PORT:-29784}"

read_latest_step() {
    local run_dir="$1"
    local latest_file="${run_dir}/latest_checkpointed_iteration.txt"
    if [ ! -f "$latest_file" ]; then
        echo ""
        return
    fi
    tr -d '\n\r[:space:]' < "$latest_file"
}

src_step="$(read_latest_step "$SRC_WEIGHTS")"
if [ -z "$src_step" ]; then
    echo "[ERROR] missing Exp3 source checkpoint tracker: $SRC_WEIGHTS" >&2
    exit 1
fi
if [ "$src_step" -lt "$SOURCE_REQUIRED_ITERS" ]; then
    echo "[ERROR] Exp3 source is not complete enough: latest=$src_step required=$SOURCE_REQUIRED_ITERS" >&2
    echo "source: $SRC_WEIGHTS" >&2
    exit 1
fi

target_step=$((src_step + RETUNE_ITERS))
dst_step="$(read_latest_step "$TRAIN_WEIGHTS")"
if [ -n "$dst_step" ] && [ "$dst_step" -ge "$target_step" ]; then
    echo "[SKIP] already completed: $TRAIN_WEIGHTS latest=$dst_step target=$target_step"
    exit 0
fi

mkdir -p "$(dirname "$TRAIN_WEIGHTS")"
if [ ! -d "$TRAIN_WEIGHTS" ]; then
    echo "[COPY] source -> phase3 copy"
    echo "[COPY] source: $SRC_WEIGHTS"
    echo "[COPY] target: $TRAIN_WEIGHTS"
    mkdir -p "$TRAIN_WEIGHTS"
    ionice -c2 -n7 nice -n 10 rsync -aH --info=progress2 \
        --exclude 'wandb/' \
        "$SRC_WEIGHTS/" "$TRAIN_WEIGHTS/"
    {
        echo "phase=exp3_phase3"
        echo "purpose=code-router-row-only retuning copy"
        echo "source_label=G2 experiment 3 KD+Code, top4+all-code forward"
        echo "source=$SRC_WEIGHTS"
        echo "source_step=$src_step"
        echo "router_init=no-reinit"
        echo "forward=ordinary top${MOE_ROUTER_TOPK} over all ${NUM_EXPERTS} experts"
        echo "trainable=shared-router code rows only, expert ids ${SOURCE_NUM_EXPERTS}..$((NUM_EXPERTS - 1))"
        echo "frozen=wiki router rows, all experts, dense trunk, embeddings, output"
        echo "loss=code LM loss"
        echo "copied_at=$(date -Iseconds)"
    } > "$TRAIN_WEIGHTS/PHASE3_SOURCE.txt"
else
    echo "[COPY] using existing phase3 copy: $TRAIN_WEIGHTS"
fi

echo "[CONFIG] G2 Exp3 Phase 3 code-router-row-only retune"
echo "[CONFIG] source_step=${src_step}, target_step=${target_step}, retune_iters=${RETUNE_ITERS}"
echo "[CONFIG] data=full code train only"
echo "[CONFIG] forward=normal top${MOE_ROUTER_TOPK} over wiki+code experts; no all-code union"
echo "[CONFIG] trainable=shared-router code rows only (${SOURCE_NUM_EXPERTS}..$((NUM_EXPERTS - 1)))"
echo "[CONFIG] frozen=wiki router rows + all experts + dense/shared params"
echo "[CONFIG] wandb_mode=${WANDB_MODE}, mb=${MICRO_BATCH_SIZE}, log_interval=${LOG_INTERVAL}"
echo "[CONFIG] train_router_usage_log_interval=${TRAIN_ROUTER_USAGE_LOG_INTERVAL}"
echo "[CONFIG] weights=${TRAIN_WEIGHTS}"

env \
    RUN_ID="$RUN_ID" \
    TRAIN_WEIGHTS="$TRAIN_WEIGHTS" \
    SOURCE_STEP="$src_step" \
    TRAIN_ITERS="$target_step" \
    WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2 - exp3 Phase 3 code-router-row-only retune}" \
    WANDB_RUN_ID="$RUN_ID" \
    MASTER_PORT="$MASTER_PORT" \
    bash "$SCRIPT_DIR/run_guarded_training.sh" \
        bash "$SCRIPT_DIR/exp3_phase3_code_router_rows_only_retune_local_bf16.sh"
