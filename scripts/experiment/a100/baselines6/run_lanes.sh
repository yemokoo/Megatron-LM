#!/bin/bash
# Run the remaining baselines6 work as parallel 2-GPU lanes fed by one queue.
#
# A queue unit is a whole method group, never a single stage: inside a group,
# Code -> Conversation is chained through checkpoints and sidecars, so a group
# must stay on one lane and run in order.  Groups are independent of each other
# once the shared common_dense/wiki checkpoint exists, which is what makes them
# safe to run concurrently.
#
# Lanes pop the next unit the moment they go idle, so a lane that finishes a
# two-stage group picks up more work instead of idling behind a three-stage one.
# A unit that fails is recorded and the lane moves on; one broken method must
# not stall the others.
#
#   BASELINES6_LANES              GPU pairs, default "0,1 2,3 4,5 6,7"
#   BASELINES6_INCLUDE_ABLATION   1 to append the SLoRA rank sweep, default 0
#   BASELINES6_ABLATION_RANKS     default "16 32 128 256"

set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/common.sh"

LANES="${BASELINES6_LANES:-0,1 2,3 4,5 6,7}"
INCLUDE_ABLATION="${BASELINES6_INCLUDE_ABLATION:-0}"
ABLATION_RANKS="${BASELINES6_ABLATION_RANKS:-16 32 128 256}"
STAGGER_SECONDS="${BASELINES6_LANE_STAGGER:-20}"
DRYRUN="${BASELINES6_LANE_DRYRUN:-0}"
ATTACH="${BASELINES6_LANE_ATTACH:-0}"

RUN_DIR="$BASELINES6_OUTPUT_ROOT/lanes"
QUEUE="$RUN_DIR/queue.txt"
LOCK="$RUN_DIR/queue.lock"
FAILURES="$RUN_DIR/failures.txt"

# Every group script starts by calling train_dense_wiki.sh.  That is only a
# fast [SKIP] when the shared Wiki is already complete; if it were not, all
# four lanes would race to train the same checkpoint directory at once.
if ! stage_is_complete "$BASELINES6_OUTPUT_ROOT/common_dense/wiki"; then
    echo "ERROR: common_dense/wiki is incomplete; lanes would race to train it" >&2
    echo "       run the single-chain launcher until Wiki is done first" >&2
    exit 1
fi

mkdir -p "$RUN_DIR"

# Attach mode adds lanes to a queue that is already being drained, which is how
# GPUs that were borrowed mid-run get put back to work.  The queue is a file
# guarded by flock, so extra workers on it are safe; rebuilding it would instead
# re-run everything.
if [ "$ATTACH" = "1" ]; then
    if [ ! -f "$QUEUE" ]; then
        echo "ERROR: attach mode needs an existing queue at $QUEUE" >&2
        exit 1
    fi
    echo "[LANES] attaching lanes '$LANES' to the running queue ($(wc -l < "$QUEUE") units left)"
else
    : > "$FAILURES"
fi

# Longest groups first so the three-stage ones start before the short ones and
# do not become the tail of the schedule.  Paths are expanded here, not left as
# $DIR: each unit is later run through `bash -c` in a fresh shell that would
# not inherit this script's variables.  Attach mode must never reach this --
# rewriting a queue that lanes are already draining would re-run finished
# groups and lose whatever is left.
if [ "$ATTACH" != "1" ]; then
    {
        echo "olora|bash '$DIR/olora.sh'"
        echo "fixed_moe|bash '$DIR/fixed_moe.sh'"
        echo "ewc|bash '$DIR/ewc.sh'"
        echo "trace_gem|bash '$DIR/trace_gem.sh'"
        echo "slora_r64|SLORA_RANKS=64 bash '$DIR/slora_pre.sh'"
        echo "sequential_dense|bash '$DIR/sequential_dense.sh'"
        if [ "$INCLUDE_ABLATION" = "1" ]; then
            for RANK in $ABLATION_RANKS; do
                echo "slora_r${RANK}|SLORA_RANKS=${RANK} bash '$DIR/slora_pre.sh'"
            done
        fi
    } > "$QUEUE"
fi

echo "[LANES] queue:"
sed 's/^/  /' "$QUEUE"
echo "[LANES] lanes: $LANES"

pop_unit() {
    local unit=""
    {
        flock 9
        unit="$(head -n 1 "$QUEUE" 2>/dev/null || true)"
        if [ -n "$unit" ]; then
            sed -i '1d' "$QUEUE"
        fi
    } 9>"$LOCK"
    printf '%s' "$unit"
}

lane_worker() {
    local gpus="$1" lane_id="$2"
    local log="$RUN_DIR/lane${lane_id}.log"
    local unit name cmd status
    while :; do
        unit="$(pop_unit)"
        [ -z "$unit" ] && break
        name="${unit%%|*}"
        cmd="${unit#*|}"
        printf '[LANE %s gpus=%s] START %s %s\n' \
            "$lane_id" "$gpus" "$name" "$(date '+%F %T')" | tee -a "$log"
        if [ "$DRYRUN" = "1" ]; then
            # Exercise the queue, locking and lane hand-off without any GPU work.
            bash -c "sleep $(( (RANDOM % 4) + 1 )); echo \"[dry] $cmd\"" >> "$log" 2>&1
        else
            CUDA_VISIBLE_DEVICES="$gpus" \
            BASELINES6_OUTPUT_ROOT="$BASELINES6_OUTPUT_ROOT" \
                bash -c "$cmd" >> "$log" 2>&1
        fi
        status=$?
        printf '[LANE %s] %s exit=%s %s\n' \
            "$lane_id" "$name" "$status" "$(date '+%F %T')" | tee -a "$log"
        [ "$status" -ne 0 ] && printf '%s exit=%s\n' "$name" "$status" >> "$FAILURES"
    done
    printf '[LANE %s gpus=%s] queue empty, lane finished %s\n' \
        "$lane_id" "$gpus" "$(date '+%F %T')" | tee -a "$log"
}

lane_id="${BASELINES6_LANE_ID_BASE:-0}"
for gpus in $LANES; do
    lane_worker "$gpus" "$lane_id" &
    lane_id=$(( lane_id + 1 ))
    # Stagger so four dataset-index builds and NCCL rendezvous do not collide.
    sleep "$STAGGER_SECONDS"
done
wait

echo "[LANES] all lanes finished $(date '+%F %T')"
if [ -s "$FAILURES" ]; then
    echo "[LANES] FAILURES:"
    sed 's/^/  /' "$FAILURES"
    exit 1
fi
echo "[LANES] every queued group exited 0"
