#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: run_guarded_training.sh <command...>" >&2
    exit 1
fi

RUN_ID="${RUN_ID:?RUN_ID must be set}"
TRAIN_ITERS="${TRAIN_ITERS:?TRAIN_ITERS must be set}"
TRAIN_WEIGHTS="${TRAIN_WEIGHTS:?TRAIN_WEIGHTS must be set}"
LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
GUARD_POLL_SECONDS="${GUARD_POLL_SECONDS:-30}"
GUARD_GRACE_SECONDS="${GUARD_GRACE_SECONDS:-180}"

tmp_root="${LOCAL_SSD_ROOT}/${RUN_ID}"
candidate_dirs=(
    "${tmp_root}/weights"
    "${tmp_root}/target_weights"
    "${TRAIN_WEIGHTS}"
)

find_tracker() {
    local dir
    for dir in "${candidate_dirs[@]}"; do
        if [ -f "${dir}/latest_checkpointed_iteration.txt" ]; then
            echo "${dir}/latest_checkpointed_iteration.txt"
            return 0
        fi
    done
    return 1
}

find_sync_source() {
    local dir
    for dir in "${candidate_dirs[@]}"; do
        if [ -d "$dir" ]; then
            echo "$dir"
            return 0
        fi
    done
    return 1
}

echo "[guard] starting ${RUN_ID} at $(date)"
setsid "$@" &
cmd_pid=$!

completion_seen_at=""
while kill -0 "$cmd_pid" 2>/dev/null; do
    tracker="$(find_tracker || true)"
    if [ -n "${tracker:-}" ]; then
        tracker_value="$(tr -d '\n\r[:space:]' < "$tracker" 2>/dev/null || true)"
        if [ "$tracker_value" = "$TRAIN_ITERS" ]; then
            if [ -z "$completion_seen_at" ]; then
                completion_seen_at="$(date +%s)"
                echo "[guard] ${RUN_ID} reached target iteration ${TRAIN_ITERS} at $(date)"
            fi

            now="$(date +%s)"
            if [ $((now - completion_seen_at)) -ge "$GUARD_GRACE_SECONDS" ]; then
                sync_source="$(find_sync_source || true)"
                if [ -n "${sync_source:-}" ] && [ "$sync_source" != "$TRAIN_WEIGHTS" ]; then
                    mkdir -p "$TRAIN_WEIGHTS"
                    echo "[guard] syncing ${sync_source} -> ${TRAIN_WEIGHTS}"
                    rsync -rlptD "${sync_source}/" "${TRAIN_WEIGHTS}/"
                fi
                echo "[guard] terminating hung run ${RUN_ID} after grace period"
                kill -TERM -"${cmd_pid}" 2>/dev/null || true
                sleep 5
                kill -KILL -"${cmd_pid}" 2>/dev/null || true
                break
            fi
        fi
    fi
    sleep "$GUARD_POLL_SECONDS"
done

wait "$cmd_pid" 2>/dev/null || true

tracker="$(find_tracker || true)"
if [ -n "${tracker:-}" ]; then
    tracker_value="$(tr -d '\n\r[:space:]' < "$tracker" 2>/dev/null || true)"
    if [ "$tracker_value" = "$TRAIN_ITERS" ]; then
        sync_source="$(find_sync_source || true)"
        if [ -n "${sync_source:-}" ] && [ "$sync_source" != "$TRAIN_WEIGHTS" ]; then
            mkdir -p "$TRAIN_WEIGHTS"
            echo "[guard] final sync ${sync_source} -> ${TRAIN_WEIGHTS}"
            rsync -rlptD "${sync_source}/" "${TRAIN_WEIGHTS}/"
        fi
        echo "[guard] completed ${RUN_ID} with checkpoint ${TRAIN_ITERS}"
        exit 0
    fi
fi

echo "[guard] ${RUN_ID} ended without confirmed checkpoint ${TRAIN_ITERS}" >&2
exit 1
