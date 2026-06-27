#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

run_variant() {
    local tag="$1"
    local label="$2"
    local ffn_hidden="$3"
    local port_base="$4"

    echo "================================================================================"
    echo "[VARIANT START] $label | ffn_hidden=$ffn_hidden | $(date)"
    echo "================================================================================"

    env \
        DENSE_VARIANT_TAG="$tag" \
        DENSE_VARIANT_LABEL="$label" \
        FFN_HIDDEN_SIZE="$ffn_hidden" \
        WIKI_MASTER_PORT="$((port_base + 1))" \
        CODE_MASTER_PORT="$((port_base + 2))" \
        CONV_MASTER_PORT="$((port_base + 3))" \
        bash "$SCRIPT_DIR/run_g2_dense24_wiki_code_conversation_mha.sh"

    echo "================================================================================"
    echo "[VARIANT END] $label | $(date)"
    echo "================================================================================"
    sleep "${BETWEEN_VARIANT_PAUSE_SECONDS:-300}"
}

export WANDB_MODE="${WANDB_MODE:-offline}"
export SEED="${SEED:-1234}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-36}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1800}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}"
export TERTIARY_PROBE_EVAL_INTERVAL="${TERTIARY_PROBE_EVAL_INTERVAL:-50}"
export OLD_MODEL_KL_COEFF="${OLD_MODEL_KL_COEFF:-1.0}"
export OLD_MODEL_KL_TEMPERATURE="${OLD_MODEL_KL_TEMPERATURE:-1.0}"
export PAUSE_SECONDS="${PAUSE_SECONDS:-180}"

echo "[CONFIG] G2 dense capacity/active continual baselines"
echo "[CONFIG] stages per variant: wiki -> code -> conversation"
echo "[CONFIG] variants:"
echo "[CONFIG]   capacity-match: ffn_hidden=8448 = 24 experts * 352 total capacity"
echo "[CONFIG]   active-match:   ffn_hidden=1408 = top4 active experts * 352"
echo "[CONFIG] mb=$MICRO_BATCH_SIZE gbs=$GLOBAL_BATCH_SIZE seed=$SEED save_interval=$SAVE_INTERVAL"
echo "[CONFIG] wandb_mode=$WANDB_MODE; code/conversation KD coeff=$OLD_MODEL_KL_COEFF"

run_variant "dense24_capacity" "dense24-capacity ffn8448" 8448 "${CAPACITY_MASTER_PORT_BASE:-29940}"
run_variant "dense4_active" "dense4-active ffn1408" 1408 "${ACTIVE_MASTER_PORT_BASE:-29950}"

echo "[ALL DONE] dense capacity/active wiki -> code -> conversation chain $(date)"
