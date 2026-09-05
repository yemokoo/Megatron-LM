#!/usr/bin/env bash
set -euo pipefail

# Launch four independent world-size-1 paired-forward workers on GPUs 0--3.
# Each worker loads both checkpoints and compares the exact same Wiki batch.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_code_token_hidden_pair_mha.sh"

OUT_ROOT="${OUT_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/wiki_token_representation_positive_20260811}"
WIKI_PREFIX="${WIKI_PREFIX:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/wiki/train/train_text_document}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-21504}"
WORKER_COUNT=4
GPU_LIST="${GPU_LIST:-0,1,2,3}"
STAGGER_SECONDS="${STAGGER_SECONDS:-2}"
WAIT_FOR_WORKERS="${WAIT_FOR_WORKERS:-0}"
PAIR_LABEL="${PAIR_LABEL:-wiki_train_positive_post_wiki_kd_init_step600_vs_code_wiki_replay_one_phase_step1800}"

[[ -x "$RUNNER" ]] || { echo "[ERROR] missing runner: $RUNNER" >&2; exit 1; }
[[ -f "$WIKI_PREFIX.bin" && -f "$WIKI_PREFIX.idx" ]] || {
    echo "[ERROR] indexed Wiki corpus missing: $WIKI_PREFIX.{bin,idx}" >&2
    exit 1
}
[[ "$TOTAL_SAMPLES" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] invalid TOTAL_SAMPLES" >&2; exit 2; }
(( TOTAL_SAMPLES % WORKER_COUNT == 0 )) || {
    echo "[ERROR] TOTAL_SAMPLES must divide evenly over four workers" >&2
    exit 2
}
per_worker=$(( TOTAL_SAMPLES / WORKER_COUNT ))
(( per_worker % 48 == 0 )) || {
    echo "[ERROR] each partition must align to micro-batch 48: $per_worker" >&2
    exit 2
}
(( TOTAL_SAMPLES * 512 >= 10000000 )) || {
    echo "[ERROR] Wiki positive reference must cover at least 10M contextual tokens" >&2
    exit 2
}

IFS=',' read -r -a gpus <<< "$GPU_LIST"
(( ${#gpus[@]} == WORKER_COUNT )) || { echo "[ERROR] GPU_LIST needs four GPUs" >&2; exit 2; }
declare -A seen=()
for gpu in "${gpus[@]}"; do
    [[ "$gpu" =~ ^[0-3]$ ]] || { echo "[ERROR] only GPUs 0--3 are allowed: $gpu" >&2; exit 2; }
    [[ -z "${seen[$gpu]:-}" ]] || { echo "[ERROR] duplicate GPU: $gpu" >&2; exit 2; }
    seen[$gpu]=1
done

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/pids"
manifest="$OUT_ROOT/partitions.tsv"
manifest_tmp="$(mktemp "$OUT_ROOT/.partitions.tsv.XXXXXX")"
trap '[[ ! -e "$manifest_tmp" ]] || rm -f -- "$manifest_tmp"' EXIT
printf 'worker\tgpu\tstart_sample\tend_sample_exclusive\tsamples\ttokens\n' > "$manifest_tmp"
for ((worker=0; worker<WORKER_COUNT; worker++)); do
    start=$(( worker * per_worker ))
    end=$(( start + per_worker ))
    printf '%d\t%s\t%d\t%d\t%d\t%d\n' \
        "$worker" "${gpus[$worker]}" "$start" "$end" "$per_worker" "$((per_worker * 512))" \
        >> "$manifest_tmp"
done
if [[ -f "$manifest" ]]; then
    cmp -s "$manifest" "$manifest_tmp" || {
        echo "[ERROR] existing partition manifest differs: $manifest" >&2
        exit 1
    }
else
    mv "$manifest_tmp" "$manifest"
fi

echo "[PLAN] Wiki samples=$TOTAL_SAMPLES tokens=$((TOTAL_SAMPLES * 512)) workers=4"
sed -n '1,8p' "$manifest"
for ((worker=0; worker<WORKER_COUNT; worker++)); do
    GPU="${gpus[$worker]}" WORKER_INDEX="$worker" WORKER_COUNT="$WORKER_COUNT" \
        TOTAL_SAMPLES="$TOTAL_SAMPLES" OUT_ROOT="$OUT_ROOT" CODE_PREFIX="$WIKI_PREFIX" \
        PAIR_LABEL="$PAIR_LABEL" PLAN_ONLY=1 bash "$RUNNER"
done
[[ "${PLAN_ONLY:-0}" == 1 ]] && exit 0

# Check every requested device before starting any worker. Never terminate a process.
gpu_inventory="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)"
compute_inventory="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits)"
for gpu in "${gpus[@]}"; do
    uuid="$(awk -F', ' -v requested="$gpu" '$1 == requested {print $2}' <<< "$gpu_inventory")"
    [[ -n "$uuid" ]] || { echo "[ERROR] GPU not found: $gpu" >&2; exit 1; }
    pids="$(awk -F', ' -v requested="$uuid" '$1 == requested {print $2}' <<< "$compute_inventory")"
    [[ -z "$pids" ]] || {
        echo "[COLLISION] GPU $gpu is occupied by PIDs: $pids" >&2
        exit 75
    }
done

declare -a pids=()
declare -a ranks=()
for ((worker=0; worker<WORKER_COUNT; worker++)); do
    gpu="${gpus[$worker]}"
    rank="$(printf 'rank_%03d' "$worker")"
    launch_log="$OUT_ROOT/logs/${rank}.launcher.log"
    echo "[LAUNCH] $rank GPU=$gpu"
    GPU="$gpu" WORKER_INDEX="$worker" WORKER_COUNT="$WORKER_COUNT" \
        TOTAL_SAMPLES="$TOTAL_SAMPLES" OUT_ROOT="$OUT_ROOT" CODE_PREFIX="$WIKI_PREFIX" \
        PAIR_LABEL="$PAIR_LABEL" nohup bash "$RUNNER" >> "$launch_log" 2>&1 < /dev/null &
    pid=$!
    printf '%s\n' "$pid" > "$OUT_ROOT/pids/${rank}.pid"
    pids+=("$pid")
    ranks+=("$rank")
    (( STAGGER_SECONDS == 0 || worker + 1 == WORKER_COUNT )) || sleep "$STAGGER_SECONDS"
done

echo "[STARTED] ${#pids[@]} workers; monitor $OUT_ROOT/rank_*/progress.json"
if [[ "$WAIT_FOR_WORKERS" == 1 ]]; then
    overall=0
    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then
            echo "[DONE] ${ranks[$index]}"
        else
            echo "[FAILED] ${ranks[$index]}" >&2
            overall=1
        fi
    done
    exit "$overall"
fi
