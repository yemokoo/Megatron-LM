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
export RUN_ID_PREFIX="${RUN_ID_PREFIX:-g2-exp1-phase3-router-only-retune-wikicode-data-budget}"
export ROUTER_INIT="${ROUTER_INIT:-no-reinit}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-72}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export CURRENT_50PCT_RETUNE_ITERS="${CURRENT_50PCT_RETUNE_ITERS:-1800}"
export FULL_100PCT_RETUNE_ITERS="${FULL_100PCT_RETUNE_ITERS:-$((CURRENT_50PCT_RETUNE_ITERS * 2))}"
export DATA_BUDGET_PCTS="${DATA_BUDGET_PCTS:-1 5 10 25 50 100}"
export MASTER_PORT_BASE="${MASTER_PORT_BASE:-29810}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
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
mkdir -p "$SWEEP_ROOT"

iters_for_pct() {
    local pct="$1"
    case "$pct" in
        1) echo 36 ;;
        5) echo 180 ;;
        10) echo 360 ;;
        25) echo 900 ;;
        50) echo "$CURRENT_50PCT_RETUNE_ITERS" ;;
        100) echo "$FULL_100PCT_RETUNE_ITERS" ;;
        *)
            echo "[ERROR] unsupported DATA_BUDGET_PCTS entry: $pct" >&2
            echo "[ERROR] allowed values: 1 5 10 25 50 100" >&2
            return 1
            ;;
    esac
}

copy_source_if_needed() {
    local dst="$1"
    if [ -f "$dst/latest_checkpointed_iteration.txt" ]; then
        return 0
    fi

    if [ -e "$dst" ]; then
        echo "[ERROR] destination exists but has no tracker: $dst" >&2
        exit 1
    fi

    echo "[COPY] source:      $SOURCE_WEIGHTS"
    echo "[COPY] destination: $dst"
    mkdir -p "$(dirname "$dst")"
    rsync -aH --info=progress2 \
        --exclude 'wandb/' \
        --exclude 'events.out.tfevents*' \
        "$SOURCE_WEIGHTS/" "$dst/"
    {
        echo "source_weights=$SOURCE_WEIGHTS"
        echo "source_step=$source_step"
        echo "copied_at=$(date -Iseconds)"
    } > "$dst/PHASE3_SOURCE.txt"
}

run_one_budget() {
    local pct="$1"
    local idx="$2"
    local retune_iters target_step run_id train_weights outlog

    retune_iters="$(iters_for_pct "$pct")"
    target_step="$((source_step + retune_iters))"
    run_id="${RUN_ID_PREFIX}-${pct}pct-from-all-experts-router-${ROUTER_INIT}-mb${MICRO_BATCH_SIZE}-${retune_iters}"
    train_weights="$SWEEP_ROOT/$run_id"
    outlog="g2_exp1_phase3_router_data_budget_${pct}pct_$(date +%Y%m%d_%H%M%S).log"

    echo
    echo "=== budget ${pct}% ==="
    echo "source_step=$source_step retune_iters=$retune_iters target_step=$target_step"
    echo "TRAIN_WEIGHTS=$train_weights"

    if [ -f "$train_weights/latest_checkpointed_iteration.txt" ]; then
        current="$(tr -d '\n\r[:space:]' < "$train_weights/latest_checkpointed_iteration.txt")"
        if [ "$current" = "$target_step" ]; then
            echo "[SKIP] already completed target checkpoint $target_step"
            return 0
        fi
        echo "[RESUME] current checkpoint tracker: $current"
    fi

    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY_RUN] would copy and run $run_id"
        return 0
    fi

    copy_source_if_needed "$train_weights"

    WANDB_MODE="$WANDB_MODE" \
    RUN_ID="$run_id" \
    TRAIN_WEIGHTS="$train_weights" \
    RETUNE_ITERS="$retune_iters" \
    MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" \
    GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
    SAVE_INTERVAL="$retune_iters" \
    PROBE_EVAL_INTERVAL="$retune_iters" \
    SECONDARY_PROBE_EVAL_INTERVAL="$retune_iters" \
    PROBE_EVAL_ITERS="$PROBE_EVAL_ITERS" \
    SECONDARY_PROBE_EVAL_ITERS="$SECONDARY_PROBE_EVAL_ITERS" \
    TRAIN_ROUTER_USAGE_LOG_INTERVAL="$TRAIN_ROUTER_USAGE_LOG_INTERVAL" \
    DATASET_NAME="wiki_code_mixed_data_budget_${pct}pct" \
    DATASET_SOURCE="Balanced wiki+code stream; retune_iters=${retune_iters}; current 50pct=${CURRENT_50PCT_RETUNE_ITERS}; full 100pct=${FULL_100PCT_RETUNE_ITERS}" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_EXP_NAME="G2 exp1 phase3 router data budget ${pct}% wiki+code" \
    MASTER_PORT="$((MASTER_PORT_BASE + idx))" \
    bash "$SCRIPT_DIR/run_guarded_training.sh" \
        bash "$SCRIPT_DIR/phase3_router_only_retune_shared_router_hybrid_mixed_local_bf16.sh" \
        > "$outlog" 2>&1

    echo "[DONE] budget ${pct}% | outlog=$outlog"
}

echo "[CONFIG] G2 exp1 phase3 router-only data-budget sweep"
echo "[CONFIG] source=$SOURCE_WEIGHTS"
echo "[CONFIG] source_step=$source_step"
echo "[CONFIG] budgets=$DATA_BUDGET_PCTS"
echo "[CONFIG] current_50pct_retune_iters=$CURRENT_50PCT_RETUNE_ITERS"
echo "[CONFIG] full_100pct_retune_iters=$FULL_100PCT_RETUNE_ITERS"
echo "[CONFIG] sweep_root=$SWEEP_ROOT"
echo "[CONFIG] wandb_mode=$WANDB_MODE"

idx=0
for pct in $DATA_BUDGET_PCTS; do
    run_one_budget "$pct" "$idx"
    idx=$((idx + 1))
done

echo
echo "[DONE] all requested data-budget runs finished"
