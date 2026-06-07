#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export FIXED_DATA_SEED="${FIXED_DATA_SEED:-20260607}"
export ROUTER_FINETUNE_DATASET_ROOT="${ROUTER_FINETUNE_DATASET_ROOT:-$LOCAL_BASE/datasets/router_finetune_20pct/seed${FIXED_DATA_SEED}}"
export RETUNE_ITERS="${RETUNE_ITERS:-720}"
export DATASET_NAME="${DATASET_NAME:-wiki_code_fixed_20pct_seed${FIXED_DATA_SEED}}"
export DATASET_SOURCE="${DATASET_SOURCE:-Fixed 20pct deterministic random wiki/code train subsets; seed=${FIXED_DATA_SEED}}"

if [ -f "$ROUTER_FINETUNE_DATASET_ROOT/env.sh" ]; then
    # shellcheck disable=SC1090
    source "$ROUTER_FINETUNE_DATASET_ROOT/env.sh"
fi

if [ ! -f "$ROUTER_FINETUNE_DATASET_ROOT/wiki/train/train_text_document.bin" ] || \
   [ ! -f "$ROUTER_FINETUNE_DATASET_ROOT/code/train/train_text_document.bin" ]; then
    echo "[ERROR] fixed 20pct finetune dataset is missing under: $ROUTER_FINETUNE_DATASET_ROOT" >&2
    echo "[HINT] create it once with:" >&2
    echo "  bash scripts/experiment/a100/prepare_g2_router_finetune_20pct_data_mha.sh" >&2
    exit 1
fi

echo "[CONFIG] using fixed 20pct router-finetune data"
echo "[CONFIG] root=$ROUTER_FINETUNE_DATASET_ROOT"
echo "[CONFIG] wiki=$ROUTER_FINETUNE_DATASET_ROOT/wiki/train"
echo "[CONFIG] code=$ROUTER_FINETUNE_DATASET_ROOT/code/train"
echo "[CONFIG] retune_iters=$RETUNE_ITERS"

exec bash "$SCRIPT_DIR/run_g2_phase3_router_only_retune_mha.sh" "$@"
