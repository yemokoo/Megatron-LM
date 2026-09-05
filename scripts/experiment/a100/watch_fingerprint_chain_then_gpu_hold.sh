#!/usr/bin/env bash
set -euo pipefail

CHAIN_PID="${1:?usage: $0 CHAIN_PID}"
FLAME_ENV="${FLAME_ENV:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100}"
STATUS_FILE="${STATUS_FILE:-/data2/seonghyeonnoh/LLM-continual-learning-runs/fingerprint_router_geometry_20260809/logs/overnight_full_chain/status.tsv}"
GPU_HOLD_LOG="${GPU_HOLD_LOG:-/data2/seonghyeonnoh/LLM-continual-learning-runs/fingerprint_router_geometry_20260809/logs/overnight_full_chain/gpu_hold.log}"

while kill -0 "$CHAIN_PID" 2>/dev/null; do
    sleep 30
done

printf '%s\tGPU_HOLD\tchain pid %s ended; starting gpu_hold.py regardless of exit status\n' \
    "$(date -Is)" "$CHAIN_PID" >> "$STATUS_FILE"

exec "$FLAME_ENV/bin/python" /home/seonghyeonnoh/yemokoo/gpu_hold.py \
    --gpus 0,1,2,3,4,5,6,7 >> "$GPU_HOLD_LOG" 2>&1
