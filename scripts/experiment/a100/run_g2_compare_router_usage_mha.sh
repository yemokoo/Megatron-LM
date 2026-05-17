#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

BASE_WEIGHTS_DIR="$PROJECT_ROOT/.local/weights/a100/mha/shared-router-granularity-qkvo"
BASELINE_CHECKPOINT_RUN_ID="${BASELINE_CHECKPOINT_RUN_ID:-g2-top4-e8to16-ffn352-r256-wiki-to-code-shared-router-qkvo-mha-a100-bf16-mb72-code-train-mask-wiki-experts-1800}"
KD_CHECKPOINT_RUN_ID="${KD_CHECKPOINT_RUN_ID:-g2-ts-routerkd-allrouter-kl10p0-log20-save60-1800}"
BASELINE_CHECKPOINT_DIR="${BASELINE_CHECKPOINT_DIR:-$BASE_WEIGHTS_DIR/code/$BASELINE_CHECKPOINT_RUN_ID}"
KD_CHECKPOINT_DIR="${KD_CHECKPOINT_DIR:-$BASE_WEIGHTS_DIR/code/$KD_CHECKPOINT_RUN_ID}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
DIAG_ROOT="${DIAG_ROOT:-$BASE_WEIGHTS_DIR/diagnostics/router-usage-compare-${TIMESTAMP}}"
BASELINE_OUTPUT_DIR="${BASELINE_OUTPUT_DIR:-$DIAG_ROOT/g2-no-kd}"
KD_OUTPUT_DIR="${KD_OUTPUT_DIR:-$DIAG_ROOT/g2-routerkd-kl10}"
PLOT_OUTPUT_DIR="${PLOT_OUTPUT_DIR:-$DIAG_ROOT/plots}"

PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-5}"
WANDB_MODE="${WANDB_MODE:-offline}"
BASELINE_LABEL="${BASELINE_LABEL:-G2 no KD}"
KD_LABEL="${KD_LABEL:-G2 RouterKD lambda=10}"
RUN_PROBES="${RUN_PROBES:-1}"

require_checkpoint() {
    local label="$1"
    local checkpoint_dir="$2"
    if [ ! -d "$checkpoint_dir" ]; then
        echo "[ERROR] missing ${label} checkpoint: $checkpoint_dir" >&2
        exit 1
    fi
}

run_probe() {
    local label="$1"
    local checkpoint_dir="$2"
    local output_dir="$3"
    local run_id="$4"
    local run_name="$5"
    local master_port="$6"

    echo "[RUN] ${label} router usage probe"
    CHECKPOINT_DIR="$checkpoint_dir" \
    OUTPUT_DIR="$output_dir" \
    RUN_ID="$run_id" \
    RUN_NAME="$run_name" \
    PROBE_EVAL_ITERS="$PROBE_EVAL_ITERS" \
    WANDB_MODE="$WANDB_MODE" \
    MASTER_PORT="$master_port" \
    bash "$SCRIPT_DIR/run_g2_probe_router_usage_mha.sh"
}

require_checkpoint "$BASELINE_LABEL" "$BASELINE_CHECKPOINT_DIR"
require_checkpoint "$KD_LABEL" "$KD_CHECKPOINT_DIR"

echo "[CONFIG] compare code router usage"
echo "[CONFIG] baseline=$BASELINE_CHECKPOINT_DIR"
echo "[CONFIG] kd=$KD_CHECKPOINT_DIR"
echo "[CONFIG] probe_eval_iters=$PROBE_EVAL_ITERS"
echo "[CONFIG] output=$DIAG_ROOT"

if [ "$RUN_PROBES" = "1" ]; then
    run_probe \
        "$BASELINE_LABEL" \
        "$BASELINE_CHECKPOINT_DIR" \
        "$BASELINE_OUTPUT_DIR" \
        "router-usage-g2-nokd-${TIMESTAMP}" \
        "Router usage code probe - G2 no KD" \
        "${BASELINE_MASTER_PORT:-29742}"

    run_probe \
        "$KD_LABEL" \
        "$KD_CHECKPOINT_DIR" \
        "$KD_OUTPUT_DIR" \
        "router-usage-g2-routerkd-kl10-${TIMESTAMP}" \
        "Router usage code probe - G2 RouterKD lambda=10" \
        "${KD_MASTER_PORT:-29744}"
else
    echo "[SKIP] RUN_PROBES=0, plotting existing diagnostics only"
fi

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/analysis/plot_router_usage_comparison.py" \
    --baseline-dir "$BASELINE_OUTPUT_DIR" \
    --kd-dir "$KD_OUTPUT_DIR" \
    --output-dir "$PLOT_OUTPUT_DIR" \
    --baseline-label "$BASELINE_LABEL" \
    --kd-label "$KD_LABEL" \
    --num-experts "${NUM_EXPERTS:-16}" \
    --num-existing-experts "${SOURCE_NUM_EXPERTS:-8}" \
    --probe-name code_probe

echo "[DONE] comparison plots: $PLOT_OUTPUT_DIR"
