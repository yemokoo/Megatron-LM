#!/usr/bin/env bash
# Shared six-cell queue. 0,1 and 2,3 start now; 4,5 joins once both are free.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SWEEP_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/moelpr_gamma_2gpu_nongrouped_20260911}"
QUEUE="$ROOT/queue"
LOCK="$ROOT/queue.lock"
LOGS="$ROOT/scheduler_logs"
mkdir -p "$ROOT/state" "$LOGS"

pop_job() {
    exec 9>"$LOCK"; flock 9
    local line
    line=$(head -1 "$QUEUE" 2>/dev/null || true)
    [ -z "$line" ] || sed -i '1d' "$QUEUE"
    flock -u 9; exec 9>&-
    printf '%s' "$line"
}

worker() {
    local pair=$1 cell gamma out rc port_base
    case "$pair" in
        0,1) port_base=30400 ;;
        2,3) port_base=30500 ;;
        4,5) port_base=30600 ;;
        *) echo "unsupported GPU pair: $pair" >&2; return 2 ;;
    esac
    while cell=$(pop_job) && [ -n "$cell" ]; do
        IFS='|' read -r cell gamma <<< "$cell"
        out="$ROOT/$cell"
        if [ "$(tr -d '[:space:]' < "$out/conversation_router_lpr/latest_checkpointed_iteration.txt" 2>/dev/null || true)" = 2160 ]; then
            echo "DONE(skip)" > "$ROOT/state/$cell"; continue
        fi
        echo "RUNNING gpus=$pair $(date '+%F %T')" > "$ROOT/state/$cell"
        set +e
        GAMMA="$gamma" NAME="$cell" GPUS="$pair" PORT_BASE="$port_base" SWEEP_ROOT="$ROOT" \
            bash "$HERE/run_moelpr_gamma_2gpu.sh" >> "$LOGS/$cell.log" 2>&1
        rc=$?
        set -e
        if [ "$(tr -d '[:space:]' < "$out/conversation_router_lpr/latest_checkpointed_iteration.txt" 2>/dev/null || true)" = 2160 ]; then
            echo "DONE rc=$rc $(date '+%F %T')" > "$ROOT/state/$cell"
        else
            echo "FAIL rc=$rc $(date '+%F %T')" > "$ROOT/state/$cell"
        fi
    done
}

wait_and_worker_45() {
    while true; do
        local m4 m5
        m4=$(nvidia-smi -i 4 --query-gpu=memory.used --format=csv,noheader,nounits)
        m5=$(nvidia-smi -i 5 --query-gpu=memory.used --format=csv,noheader,nounits)
        if [ "$m4" -lt 2000 ] && [ "$m5" -lt 2000 ]; then worker 4,5; return; fi
        sleep 30
    done
}

case "${1:-status}" in
    start)
        printf '%s\n' 'g0.01|0.01' 'g0.05|0.05' 'g0.1|0.1' 'g0.5|0.5' 'g1|1' 'g5|5' > "$QUEUE"
        setsid nohup bash "$0" worker 0,1 > "$LOGS/worker.0_1.log" 2>&1 < /dev/null &
        setsid nohup bash "$0" worker 2,3 > "$LOGS/worker.2_3.log" 2>&1 < /dev/null &
        setsid nohup bash "$0" wait45 > "$LOGS/worker.4_5.log" 2>&1 < /dev/null &
        echo "started: 0,1 + 2,3; 4,5 waiting"
        ;;
    worker) worker "${2:?pair required}" ;;
    wait45) wait_and_worker_45 ;;
    status)
        for f in "$ROOT/state"/*; do [ -e "$f" ] && echo "$(basename "$f"): $(cat "$f")"; done
        echo "queued: $(wc -l < "$QUEUE" 2>/dev/null || echo 0)"
        ;;
    *) echo "usage: $0 start|status" >&2; exit 2 ;;
esac
