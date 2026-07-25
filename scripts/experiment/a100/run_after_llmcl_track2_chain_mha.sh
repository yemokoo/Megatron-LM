#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"

LLMCL_ROOT="${LLMCL_ROOT:-$WORKSPACE_ROOT/30_flame_agent/llmcl_benchmark}"
LLMCL_OUTPUT="${LLMCL_OUTPUT:-$LLMCL_ROOT/output/track2_OLMoE_ept1_force_upper_5k_seed1234}"
LLMCL_CHECKPOINT="${LLMCL_CHECKPOINT:-$LLMCL_OUTPUT/7}"
LLMCL_MODEL_FILE="$LLMCL_CHECKPOINT/pytorch_model.bin"
LLMCL_META_FILE="$LLMCL_CHECKPOINT/moe_ffn_meta.json"
POLL_SECONDS="${POLL_SECONDS:-30}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-0}"

CHAIN_SCRIPT="$SCRIPT_DIR/run_g2_ffn_only_code1_kd_conv1_chain_mha.sh"
CHAIN_CUDA_VISIBLE_DEVICES="${CHAIN_CUDA_VISIBLE_DEVICES:-0,1}"
CHAIN_NPROC_PER_NODE="${CHAIN_NPROC_PER_NODE:-2}"
WRAPPER_LOG_DIR="${WRAPPER_LOG_DIR:-$PROJECT_ROOT/.local/logs/g2_code1_kd_conv1_chain}"
WRAPPER_LOG="$WRAPPER_LOG_DIR/after_llmcl_track2_chain.log"

mkdir -p "$WRAPPER_LOG_DIR"

log() {
    echo "[$(date -Is)] $*" | tee -a "$WRAPPER_LOG"
}

checkpoint_complete() {
    [ -s "$LLMCL_MODEL_FILE" ] || return 1
    [ -s "$LLMCL_META_FILE" ] || return 1
    python3 - "$LLMCL_META_FILE" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    meta = json.load(handle)
if meta.get("num_new_experts") != 8 or meta.get("experts_per_task") != 1:
    raise SystemExit(1)
PY
}

if [ ! -x "$CHAIN_SCRIPT" ]; then
    log "ERROR: chain script is missing or not executable: $CHAIN_SCRIPT"
    exit 1
fi
if ! [[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    log "ERROR: POLL_SECONDS must be a positive integer: $POLL_SECONDS"
    exit 1
fi
if ! [[ "$WAIT_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]]; then
    log "ERROR: WAIT_TIMEOUT_SECONDS must be a non-negative integer: $WAIT_TIMEOUT_SECONDS"
    exit 1
fi

start_time=$(date +%s)
log "Waiting for completed LLMCL Track-2 checkpoint: $LLMCL_CHECKPOINT"
log "Expected metadata: num_new_experts=8, experts_per_task=1"

until checkpoint_complete; do
    if [ "$WAIT_TIMEOUT_SECONDS" -gt 0 ]; then
        now=$(date +%s)
        if [ $((now - start_time)) -ge "$WAIT_TIMEOUT_SECONDS" ]; then
            log "ERROR: timed out waiting after ${WAIT_TIMEOUT_SECONDS}s"
            exit 1
        fi
    fi
    sleep "$POLL_SECONDS"
done

log "Checkpoint complete. Waiting for LLMCL workers to exit."
while pgrep -f "[m]ain_Ours_MoE_FFN.py" >/dev/null; do
    sleep "$POLL_SECONDS"
done

sync
log "Starting G2 chain with GPUs=$CHAIN_CUDA_VISIBLE_DEVICES, nproc=$CHAIN_NPROC_PER_NODE"

set +e
CUDA_VISIBLE_DEVICES="$CHAIN_CUDA_VISIBLE_DEVICES" \
NPROC_PER_NODE="$CHAIN_NPROC_PER_NODE" \
bash "$CHAIN_SCRIPT" 2>&1 | tee -a "$WRAPPER_LOG"
chain_status=${PIPESTATUS[0]}
set -e

if [ "$chain_status" -ne 0 ]; then
    log "ERROR: G2 chain failed with exit=$chain_status"
    exit "$chain_status"
fi

log "G2 chain completed successfully."
