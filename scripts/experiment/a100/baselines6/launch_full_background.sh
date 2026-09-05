#!/bin/bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${BASELINES6_OUTPUT_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/baselines6_wiki_code_conversation_20260816}"
PID_FILE="$OUTPUT_ROOT/chain.pid"
LOG_FILE="$OUTPUT_ROOT/chain.log"

mkdir -p "$OUTPUT_ROOT"
if [ -s "$PID_FILE" ]; then
    existing_pid="$(tr -d '[:space:]' < "$PID_FILE")"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
        echo "ERROR: baseline chain is already running with PID $existing_pid" >&2
        exit 1
    fi
fi

nohup setsid env \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}" \
    BASELINES6_OUTPUT_ROOT="$OUTPUT_ROOT" \
    bash "$DIR/run_all.sh" >> "$LOG_FILE" 2>&1 < /dev/null &
chain_pid=$!
printf '%s\n' "$chain_pid" > "$PID_FILE"

sleep 2
if ! kill -0 "$chain_pid" 2>/dev/null; then
    echo "ERROR: baseline chain exited during launch; inspect $LOG_FILE" >&2
    exit 1
fi
echo "[LAUNCHED] pid=$chain_pid log=$LOG_FILE output=$OUTPUT_ROOT"
