#!/usr/bin/env bash
set -euo pipefail

# Launch eight independent world-size-1 paired-forward workers.  This script
# only validates GPU availability; it never terminates existing processes.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_code_token_hidden_pair_mha.sh"

OUT_ROOT="${OUT_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/code_token_representation_gt_20260811}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-4147200}"
WORKER_COUNT=8
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
STAGGER_SECONDS="${STAGGER_SECONDS:-2}"
WAIT_FOR_WORKERS="${WAIT_FOR_WORKERS:-0}"

[[ -x "$RUNNER" ]] || {
    echo "[ERROR] runner is missing or not executable: $RUNNER" >&2
    exit 1
}
[[ "$TOTAL_SAMPLES" =~ ^[1-9][0-9]*$ ]] || {
    echo "[ERROR] TOTAL_SAMPLES must be positive" >&2
    exit 2
}
(( TOTAL_SAMPLES >= WORKER_COUNT )) || {
    echo "[ERROR] TOTAL_SAMPLES must be at least $WORKER_COUNT" >&2
    exit 2
}
[[ "$STAGGER_SECONDS" =~ ^[0-9]+$ ]] || {
    echo "[ERROR] STAGGER_SECONDS must be a non-negative integer" >&2
    exit 2
}

IFS=',' read -r -a gpus <<< "$GPU_LIST"
(( ${#gpus[@]} == WORKER_COUNT )) || {
    echo "[ERROR] GPU_LIST must contain exactly eight comma-separated GPUs" >&2
    exit 2
}
declare -A seen_gpu=()
for gpu in "${gpus[@]}"; do
    [[ "$gpu" =~ ^[0-7]$ ]] || {
        echo "[ERROR] invalid physical GPU in GPU_LIST: $gpu" >&2
        exit 2
    }
    [[ -z "${seen_gpu[$gpu]:-}" ]] || {
        echo "[ERROR] duplicate physical GPU in GPU_LIST: $gpu" >&2
        exit 2
    }
    seen_gpu[$gpu]=1
done

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/pids"

# Record an immutable partition plan.  A changed plan on resume is rejected.
manifest="$OUT_ROOT/partitions.tsv"
manifest_tmp="$(mktemp "$OUT_ROOT/.partitions.tsv.XXXXXX")"
cleanup_manifest_tmp() {
    [[ ! -e "$manifest_tmp" ]] || rm -f -- "$manifest_tmp"
}
trap cleanup_manifest_tmp EXIT
printf 'worker\tgpu\tstart_sample\tend_sample_exclusive\tsamples\n' > "$manifest_tmp"
base=$(( TOTAL_SAMPLES / WORKER_COUNT ))
rem=$(( TOTAL_SAMPLES % WORKER_COUNT ))
for (( worker=0; worker<WORKER_COUNT; worker++ )); do
    if (( worker < rem )); then
        count=$(( base + 1 ))
        start=$(( worker * base + worker ))
    else
        count=$base
        start=$(( worker * base + rem ))
    fi
    end=$(( start + count ))
    printf '%d\t%s\t%d\t%d\t%d\n' "$worker" "${gpus[$worker]}" "$start" "$end" "$count" >> "$manifest_tmp"
done
if [[ -f "$manifest" ]]; then
    cmp -s "$manifest" "$manifest_tmp" || {
        echo "[ERROR] existing partition manifest differs: $manifest" >&2
        exit 1
    }
else
    mv "$manifest_tmp" "$manifest"
fi

echo "[PLAN] total_samples=$TOTAL_SAMPLES workers=$WORKER_COUNT output=$OUT_ROOT"
column -t -s $'\t' "$manifest" 2>/dev/null || sed -n '1,20p' "$manifest"
if [[ "${PLAN_ONLY:-0}" == 1 ]]; then
    exit 0
fi

# Run every worker's read-only config/checkpoint/output preflight before any GPU
# process starts.  PLAN_ONLY deliberately stops before the per-GPU occupancy
# check and before torchrun.
for (( worker=0; worker<WORKER_COUNT; worker++ )); do
    GPU="${gpus[$worker]}" WORKER_INDEX="$worker" WORKER_COUNT="$WORKER_COUNT" \
        TOTAL_SAMPLES="$TOTAL_SAMPLES" OUT_ROOT="$OUT_ROOT" PLAN_ONLY=1 \
        bash "$RUNNER"
done

# Reject live PID-file targets before starting any worker.  Stale PID files are
# harmless and will be replaced only after the corresponding new launch.
for (( worker=0; worker<WORKER_COUNT; worker++ )); do
    rank_name="$(printf 'rank_%03d' "$worker")"
    pid_path="$OUT_ROOT/pids/${rank_name}.pid"
    [[ -f "$pid_path" ]] || continue
    old_pid="$(tr -d '[:space:]' < "$pid_path")"
    if [[ "$old_pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$old_pid" 2>/dev/null; then
        old_cmd="$(tr '\0' ' ' < "/proc/$old_pid/cmdline" 2>/dev/null || true)"
        if [[ "$old_cmd" == *run_code_token_hidden_pair_mha.sh* ]]; then
            echo "[ACTIVE] $rank_name already has runner PID $old_pid" >&2
            echo "[ABORT] no additional paired-forward worker was launched" >&2
            exit 75
        fi
        echo "[ERROR] PID file points to a live unrelated process: $pid_path -> $old_pid ($old_cmd)" >&2
        exit 1
    fi
done

# Preflight every GPU before starting any worker, preventing a half-launched run.
command -v nvidia-smi >/dev/null 2>&1 || {
    echo "[ERROR] nvidia-smi is unavailable" >&2
    exit 1
}
gpu_inventory="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)"
compute_inventory="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits)"
for gpu in "${gpus[@]}"; do
    uuid="$(awk -F', ' -v requested="$gpu" '$1 == requested {print $2}' <<< "$gpu_inventory")"
    [[ -n "$uuid" ]] || {
        echo "[ERROR] physical GPU $gpu was not found" >&2
        exit 1
    }
    pids="$(awk -F', ' -v requested="$uuid" '$1 == requested {print $2}' <<< "$compute_inventory")"
    [[ -z "$pids" ]] || {
        echo "[COLLISION] GPU $gpu is occupied by PIDs: $pids" >&2
        echo "[ABORT] no paired-forward worker was launched" >&2
        exit 75
    }
done

declare -a launched_pids=()
declare -a launched_workers=()
for (( worker=0; worker<WORKER_COUNT; worker++ )); do
    gpu="${gpus[$worker]}"
    rank_name="$(printf 'rank_%03d' "$worker")"
    pid_path="$OUT_ROOT/pids/${rank_name}.pid"
    launch_log="$OUT_ROOT/logs/${rank_name}.launcher.log"

    echo "[LAUNCH] $rank_name GPU=$gpu launcher_log=$launch_log"
    GPU="$gpu" WORKER_INDEX="$worker" WORKER_COUNT="$WORKER_COUNT" \
        TOTAL_SAMPLES="$TOTAL_SAMPLES" OUT_ROOT="$OUT_ROOT" \
        nohup bash "$RUNNER" >> "$launch_log" 2>&1 < /dev/null &
    pid=$!
    pid_tmp="$pid_path.tmp.$$"
    printf '%s\n' "$pid" > "$pid_tmp"
    mv "$pid_tmp" "$pid_path"
    launched_pids+=("$pid")
    launched_workers+=("$rank_name")
    if (( STAGGER_SECONDS > 0 && worker + 1 < WORKER_COUNT )); then
        sleep "$STAGGER_SECONDS"
    fi
done

echo "[STARTED] ${#launched_pids[@]} new worker(s)"
echo "[MONITOR] progress: $OUT_ROOT/rank_*/progress.json"
echo "[MONITOR] model logs: $OUT_ROOT/logs/rank_*.log"
echo "[MONITOR] launcher logs: $OUT_ROOT/logs/rank_*.launcher.log"

if [[ "$WAIT_FOR_WORKERS" == 1 ]]; then
    overall_rc=0
    for i in "${!launched_pids[@]}"; do
        pid="${launched_pids[$i]}"
        rank_name="${launched_workers[$i]}"
        if wait "$pid"; then
            echo "[DONE] $rank_name PID=$pid"
        else
            rc=$?
            echo "[FAILED] $rank_name PID=$pid rc=$rc" >&2
            overall_rc=1
        fi
    done
    exit "$overall_rc"
fi
