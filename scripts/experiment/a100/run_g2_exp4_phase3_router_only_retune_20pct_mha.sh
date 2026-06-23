#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
export G2_ROOT="${G2_ROOT:-$LOCAL_WEIGHTS/a100/mha/g2-checkpoints}"

export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-72}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export SOURCE_REQUIRED_ITERS="${SOURCE_REQUIRED_ITERS:-1800}"
export RETUNE_ITERS="${RETUNE_ITERS:-720}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-180}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-36}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-36}"
export PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-25}"
export SECONDARY_PROBE_EVAL_ITERS="${SECONDARY_PROBE_EVAL_ITERS:-25}"
export TRAIN_ROUTER_USAGE_LOG_INTERVAL="${TRAIN_ROUTER_USAGE_LOG_INTERVAL:-0}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export MASTER_PORT="${MASTER_PORT:-29792}"

export FIXED_DATA_SEED="${FIXED_DATA_SEED:-1234}"
export ROUTER_FINETUNE_DATASET_ROOT="${ROUTER_FINETUNE_DATASET_ROOT:-$LOCAL_BASE/datasets/router_finetune_20pct/seed${FIXED_DATA_SEED}}"

export SOURCE_RUN_ID="${SOURCE_RUN_ID:-g2-exp4-top4oldfreeze-e8to16-ffn352-r256-wiki-to-code-layerwise-top4-old-freeze-qkvo-mha-a100-bf16-mb72-1800}"
export SOURCE_WEIGHTS="${SOURCE_WEIGHTS:-$G2_ROOT/code/phase1/$SOURCE_RUN_ID}"
export RUN_ID="${RUN_ID:-g2-exp4-phase3-router-only-retune-20pct-wikicode-from-layerwise-top4-old-freeze-no-reinit-mb72-720}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$G2_ROOT/code/phase3/$RUN_ID}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2 - exp4 top4 old-freeze - phase3 router finetune 20pct}"

if ps -ef | grep -E 'pretrain_gpt.py|torchrun' | grep -v grep >/dev/null; then
    echo "[ERROR] another training process is running"
    ps -ef | grep -E 'pretrain_gpt.py|torchrun' | grep -v grep || true
    exit 1
fi

if [ ! -f "$SOURCE_WEIGHTS/latest_checkpointed_iteration.txt" ]; then
    echo "[ERROR] missing source checkpoint tracker: $SOURCE_WEIGHTS/latest_checkpointed_iteration.txt" >&2
    exit 1
fi

source_step="$(tr -d '\n\r[:space:]' < "$SOURCE_WEIGHTS/latest_checkpointed_iteration.txt")"
if [ "$source_step" -lt "$SOURCE_REQUIRED_ITERS" ]; then
    echo "[ERROR] source checkpoint is not complete enough: latest=$source_step required=$SOURCE_REQUIRED_ITERS" >&2
    exit 1
fi
target_step="$((source_step + RETUNE_ITERS))"

if [ ! -f "$ROUTER_FINETUNE_DATASET_ROOT/wiki/train/train_text_document.bin" ] || \
   [ ! -f "$ROUTER_FINETUNE_DATASET_ROOT/code/train/train_text_document.bin" ]; then
    echo "[ERROR] fixed 20pct finetune dataset is missing under: $ROUTER_FINETUNE_DATASET_ROOT" >&2
    echo "[HINT] create it once with:" >&2
    echo "  bash scripts/experiment/a100/prepare_g2_router_finetune_20pct_data_mha.sh" >&2
    exit 1
fi

copy_source_if_needed() {
    if [ -f "$TRAIN_WEIGHTS/latest_checkpointed_iteration.txt" ]; then
        return 0
    fi
    if [ -e "$TRAIN_WEIGHTS" ]; then
        echo "[ERROR] destination exists but has no tracker: $TRAIN_WEIGHTS" >&2
        exit 1
    fi

    echo "[COPY] source:      $SOURCE_WEIGHTS"
    echo "[COPY] destination: $TRAIN_WEIGHTS"
    mkdir -p "$(dirname "$TRAIN_WEIGHTS")"
    rsync -aH --info=progress2 \
        --exclude 'wandb/' \
        --exclude 'events.out.tfevents*' \
        "$SOURCE_WEIGHTS/" "$TRAIN_WEIGHTS/"
    {
        echo "source_weights=$SOURCE_WEIGHTS"
        echo "source_step=$source_step"
        echo "copied_at=$(date -Iseconds)"
    } > "$TRAIN_WEIGHTS/PHASE3_SOURCE.txt"
}

echo "[CONFIG] G2 exp4 phase3 router-only finetune with fixed 20pct wiki+code data"
echo "[CONFIG] source=$SOURCE_WEIGHTS"
echo "[CONFIG] source_step=$source_step"
echo "[CONFIG] target_step=$target_step"
echo "[CONFIG] train_weights=$TRAIN_WEIGHTS"
echo "[CONFIG] fixed_data=$ROUTER_FINETUNE_DATASET_ROOT"
echo "[CONFIG] retune_iters=$RETUNE_ITERS save_interval=$SAVE_INTERVAL probe_interval=$PROBE_EVAL_INTERVAL"
echo "[CONFIG] wandb_mode=$WANDB_MODE"

if [ -f "$TRAIN_WEIGHTS/latest_checkpointed_iteration.txt" ]; then
    current="$(tr -d '\n\r[:space:]' < "$TRAIN_WEIGHTS/latest_checkpointed_iteration.txt")"
    if [ "$current" -ge "$target_step" ]; then
        echo "[SKIP] already at or past target: latest=$current target=$target_step"
        exit 0
    fi
    echo "[RESUME] current checkpoint tracker: $current"
fi

copy_source_if_needed

export TRAIN_DATASET_WIKI="$ROUTER_FINETUNE_DATASET_ROOT/wiki/train"
export TRAIN_DATASET_CODE="$ROUTER_FINETUNE_DATASET_ROOT/code/train"
export DATASET_NAME="${DATASET_NAME:-wiki_code_fixed_20pct_seed${FIXED_DATA_SEED}}"
export DATASET_SOURCE="${DATASET_SOURCE:-Fixed 20pct deterministic random GPT sample stream from wiki train + code train; seed=${FIXED_DATA_SEED}}"

exec env \
    WANDB_MODE="$WANDB_MODE" \
    RUN_ID="$RUN_ID" \
    TRAIN_WEIGHTS="$TRAIN_WEIGHTS" \
    SOURCE_STEP="$source_step" \
    TRAIN_ITERS="$target_step" \
    RETUNE_ITERS="$RETUNE_ITERS" \
    SAVE_INTERVAL="$SAVE_INTERVAL" \
    PROBE_EVAL_INTERVAL="$PROBE_EVAL_INTERVAL" \
    SECONDARY_PROBE_EVAL_INTERVAL="$SECONDARY_PROBE_EVAL_INTERVAL" \
    PROBE_EVAL_ITERS="$PROBE_EVAL_ITERS" \
    SECONDARY_PROBE_EVAL_ITERS="$SECONDARY_PROBE_EVAL_ITERS" \
    TRAIN_ROUTER_USAGE_LOG_INTERVAL="$TRAIN_ROUTER_USAGE_LOG_INTERVAL" \
    MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" \
    GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
    ROUTER_FINETUNE_DATASET_ROOT="$ROUTER_FINETUNE_DATASET_ROOT" \
    TRAIN_DATASET_WIKI="$TRAIN_DATASET_WIKI" \
    TRAIN_DATASET_CODE="$TRAIN_DATASET_CODE" \
    DATASET_NAME="$DATASET_NAME" \
    DATASET_SOURCE="$DATASET_SOURCE" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_EXP_NAME="$WANDB_EXP_NAME" \
    WANDB_RUN_ID="$RUN_ID" \
    MASTER_PORT="$MASTER_PORT" \
    bash "$SCRIPT_DIR/run_guarded_training.sh" \
        bash "$SCRIPT_DIR/phase3_router_only_retune_shared_router_hybrid_mixed_local_bf16.sh"
