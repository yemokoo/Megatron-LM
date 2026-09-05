#!/bin/bash
# Lane queue for the replay-matched baselines.  Same lane machinery as
# run_lanes.sh; the queue is the six replay_*.sh method groups.  Longest
# (three-stage-equivalent, slower mb) groups first.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BASELINES6_OUTPUT_ROOT="${BASELINES6_OUTPUT_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/baselines6_replay0p1_20260818}"
source "$DIR/common.sh"
LANES="${BASELINES6_LANES:-6,7}"
ATTACH="${BASELINES6_LANE_ATTACH:-0}"
STAGGER="${BASELINES6_LANE_STAGGER:-20}"
RUN_DIR="$BASELINES6_OUTPUT_ROOT/lanes"; QUEUE="$RUN_DIR/queue.txt"; LOCK="$RUN_DIR/queue.lock"; FAIL="$RUN_DIR/failures.txt"
mkdir -p "$RUN_DIR"
if [ "$ATTACH" != "1" ]; then
    : > "$FAIL"
    {
        echo "slora_pre|bash '$DIR/replay_slora_pre.sh'"          # mb 24, slowest
        echo "olora|bash '$DIR/replay_olora.sh'"                  # mb 32
        echo "fixed_moe|bash '$DIR/replay_fixed_moe.sh'"          # mb 48
        echo "ewc|bash '$DIR/replay_ewc.sh'"
        echo "trace_gem|bash '$DIR/replay_trace_gem.sh'"
        echo "sequential_dense|bash '$DIR/replay_sequential_dense.sh'"
    } > "$QUEUE"
    echo "[LANES] queue:"; sed 's/^/  /' "$QUEUE"
else
    [ -f "$QUEUE" ] || { echo "ERROR: attach needs existing queue $QUEUE" >&2; exit 1; }
    echo "[LANES] attaching '$LANES' ($(wc -l < "$QUEUE") units left)"
fi
pop() { local u=""; { flock 9; u="$(head -n1 "$QUEUE" 2>/dev/null||true)"; [ -n "$u" ] && sed -i '1d' "$QUEUE"; } 9>"$LOCK"; printf '%s' "$u"; }
worker() {
    local gpus="$1" id="$2" log="$RUN_DIR/lane${id}.log" u name cmd st
    while :; do
        u="$(pop)"; [ -z "$u" ] && break
        name="${u%%|*}"; cmd="${u#*|}"
        printf '[LANE %s gpus=%s] START %s %s\n' "$id" "$gpus" "$name" "$(date '+%F %T')" | tee -a "$log"
        CUDA_VISIBLE_DEVICES="$gpus" BASELINES6_OUTPUT_ROOT="$BASELINES6_OUTPUT_ROOT" bash -c "$cmd" >> "$log" 2>&1; st=$?
        printf '[LANE %s] %s exit=%s %s\n' "$id" "$name" "$st" "$(date '+%F %T')" | tee -a "$log"
        [ "$st" -ne 0 ] && printf '%s exit=%s\n' "$name" "$st" >> "$FAIL"
    done
    printf '[LANE %s gpus=%s] queue empty, lane finished %s\n' "$id" "$gpus" "$(date '+%F %T')" | tee -a "$log"
}
id="${BASELINES6_LANE_ID_BASE:-0}"
for g in $LANES; do worker "$g" "$id" & id=$((id+1)); sleep "$STAGGER"; done
wait
echo "[LANES] all lanes finished $(date '+%F %T')"
[ -s "$FAIL" ] && { echo "[LANES] FAILURES:"; sed 's/^/  /' "$FAIL"; exit 1; }
echo "[LANES] every queued group exited 0"
