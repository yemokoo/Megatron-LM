#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
export G2_ROOT="${G2_ROOT:-$LOCAL_WEIGHTS/a100/mha/g2-checkpoints}"

export SOURCE_WEIGHTS="${SOURCE_WEIGHTS:-$G2_ROOT/code/phase1/g2-top4-e8to16-ffn352-r256-wiki-to-code-shared-router-qkvo-all-experts-router-mha-a100-bf16-mb72-1800}"
export SWEEP_ROOT="${SWEEP_ROOT:-$G2_ROOT/code/phase3_data_budget}"
export RUN_ID="${RUN_ID:-g2-exp1-phase3-router-only-retune-wikicode-data-budget-100pct-checkpoints-from-all-experts-router-no-reinit-mb72-3600}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$SWEEP_ROOT/$RUN_ID}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-72}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export CURRENT_50PCT_RETUNE_ITERS="${CURRENT_50PCT_RETUNE_ITERS:-1800}"
export FULL_100PCT_RETUNE_ITERS="${FULL_100PCT_RETUNE_ITERS:-$((CURRENT_50PCT_RETUNE_ITERS * 2))}"
export SAVE_EVERY_PERCENT_ITERS="${SAVE_EVERY_PERCENT_ITERS:-36}"
export MASTER_PORT="${MASTER_PORT:-29810}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2 exp1 phase3 router data budget 1-100% wiki+code}"
export PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-25}"
export SECONDARY_PROBE_EVAL_ITERS="${SECONDARY_PROBE_EVAL_ITERS:-25}"
export TRAIN_ROUTER_USAGE_LOG_INTERVAL="${TRAIN_ROUTER_USAGE_LOG_INTERVAL:-0}"
export DRY_RUN="${DRY_RUN:-0}"

if ps -ef | grep -E 'pretrain_gpt.py|torchrun' | grep -v grep >/dev/null; then
    echo "[ERROR] another training process is running"
    ps -ef | grep -E 'pretrain_gpt.py|torchrun' | grep -v grep || true
    exit 1
fi

if [ ! -f "$SOURCE_WEIGHTS/latest_checkpointed_iteration.txt" ]; then
    echo "[ERROR] source checkpoint tracker is missing: $SOURCE_WEIGHTS/latest_checkpointed_iteration.txt" >&2
    exit 1
fi

source_step="$(tr -d '\n\r[:space:]' < "$SOURCE_WEIGHTS/latest_checkpointed_iteration.txt")"
target_step="$((source_step + FULL_100PCT_RETUNE_ITERS))"
outlog="g2_exp1_phase3_router_data_budget_100pct_checkpoints_$(date +%Y%m%d_%H%M%S).log"

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

echo "[CONFIG] G2 exp1 phase3 router-only data-budget run"
echo "[CONFIG] source=$SOURCE_WEIGHTS"
echo "[CONFIG] source_step=$source_step"
echo "[CONFIG] train_weights=$TRAIN_WEIGHTS"
echo "[CONFIG] current_50pct_retune_iters=$CURRENT_50PCT_RETUNE_ITERS"
echo "[CONFIG] full_100pct_retune_iters=$FULL_100PCT_RETUNE_ITERS"
echo "[CONFIG] save/probe every ${SAVE_EVERY_PERCENT_ITERS} retune steps = 1% increments"
echo "[CONFIG] key milestones:"
echo "[CONFIG]   1%  -> $((source_step + 36))"
echo "[CONFIG]   5%  -> $((source_step + 180))"
echo "[CONFIG]   10% -> $((source_step + 360))"
echo "[CONFIG]   25% -> $((source_step + 900))"
echo "[CONFIG]   50% -> $((source_step + CURRENT_50PCT_RETUNE_ITERS))"
echo "[CONFIG]   100% -> $target_step"
echo "[CONFIG] wandb_mode=$WANDB_MODE"

if [ -f "$TRAIN_WEIGHTS/latest_checkpointed_iteration.txt" ]; then
    current="$(tr -d '\n\r[:space:]' < "$TRAIN_WEIGHTS/latest_checkpointed_iteration.txt")"
    if [ "$current" = "$target_step" ]; then
        echo "[SKIP] already completed target checkpoint $target_step"
        exit 0
    fi
    echo "[RESUME] current checkpoint tracker: $current"
fi

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY_RUN] would run one 100% retune with checkpoint/probe every $SAVE_EVERY_PERCENT_ITERS steps"
    exit 0
fi

copy_source_if_needed

WANDB_MODE="$WANDB_MODE" \
RUN_ID="$RUN_ID" \
TRAIN_WEIGHTS="$TRAIN_WEIGHTS" \
RETUNE_ITERS="$FULL_100PCT_RETUNE_ITERS" \
MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" \
GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
SAVE_INTERVAL="$SAVE_EVERY_PERCENT_ITERS" \
PROBE_EVAL_INTERVAL="$SAVE_EVERY_PERCENT_ITERS" \
SECONDARY_PROBE_EVAL_INTERVAL="$SAVE_EVERY_PERCENT_ITERS" \
PROBE_EVAL_ITERS="$PROBE_EVAL_ITERS" \
SECONDARY_PROBE_EVAL_ITERS="$SECONDARY_PROBE_EVAL_ITERS" \
TRAIN_ROUTER_USAGE_LOG_INTERVAL="$TRAIN_ROUTER_USAGE_LOG_INTERVAL" \
DATASET_NAME="wiki_code_mixed_data_budget_100pct_with_1pct_checkpoints" \
DATASET_SOURCE="Balanced wiki+code stream; current 50pct=${CURRENT_50PCT_RETUNE_ITERS}; full 100pct=${FULL_100PCT_RETUNE_ITERS}; checkpoint/probe every ${SAVE_EVERY_PERCENT_ITERS} retune steps" \
WANDB_PROJECT="$WANDB_PROJECT" \
WANDB_EXP_NAME="$WANDB_EXP_NAME" \
MASTER_PORT="$MASTER_PORT" \
bash "$SCRIPT_DIR/run_guarded_training.sh" \
    bash "$SCRIPT_DIR/phase3_router_only_retune_shared_router_hybrid_mixed_local_bf16.sh" \
    > "$outlog" 2>&1

echo "[DONE] data-budget 100% run finished | outlog=$outlog"
