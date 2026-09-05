#!/usr/bin/env bash
set -euo pipefail

# Restartable full Code-train CKA census on four independent H100 workers.
# The chain first benchmarks safe batch sizes on GPU 0, selects the fastest
# sustained configuration below the HBM ceiling, and only then launches the
# exhaustive four-way manifest partition.  It never kills an occupying job.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SELF="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
RUNNER="$SCRIPT_DIR/run_cka_gt_full_census_mha.sh"
CENSUS_TOOL="$SCRIPT_DIR/cka_gt_full_census.py"
PY_ENV="${PY_ENV:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100}"
PYTHON_BIN="${PYTHON_BIN:-$PY_ENV/bin/python}"

OUT_ROOT="${OUT_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_gt_full_census_20260816/analysis/cka_gt_full_census_v1}"
PILOT_ROOT="${PILOT_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_gt_pilot_20260815/analysis/cka_gt_pilot_v1}"
ANALYSIS_CONFIG="${ANALYSIS_CONFIG:-$PILOT_ROOT/analysis_config.json}"
PILOT_MANIFEST="${PILOT_MANIFEST:-$PILOT_ROOT/splits/code_manifest.json}"
CODE_PREFIX="${CODE_PREFIX:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/code/train/train_text_document}"
MANIFEST_DIR="$OUT_ROOT/manifest"
CENSUS_MANIFEST="$MANIFEST_DIR/manifest.json"
FULL_OUTPUT="$OUT_ROOT/full_census"
BENCHMARK_ROOT="$OUT_ROOT/batch_benchmark"
SELECTED_BATCH_JSON="$BENCHMARK_ROOT/selected_batch.json"
LOG_ROOT="$OUT_ROOT/logs"
STATE_ROOT="$OUT_ROOT/launch_state"
CHAIN_LOG="$LOG_ROOT/launch_cka_gt_full_census_4gpu.log"
LOCK_FILE="$OUT_ROOT/.launch_cka_gt_full_census_4gpu.lock"
GPU_USAGE_LOG="$LOG_ROOT/gpu_usage.csv"

PLAN_ONLY="${PLAN_ONLY:-0}"
START_STAGE="${START_STAGE:-manifest}"
STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-full}"
START_SYSTEMD="${START_SYSTEMD:-0}"
SYSTEMD_SERVICE="${SYSTEMD_SERVICE:-cka-gt-full-census-20260816}"
BENCHMARK_WINDOWS="${BENCHMARK_WINDOWS:-2048}"
BENCHMARK_BATCHES="${BENCHMARK_BATCHES:-64,96,128,192}"
BENCHMARK_MIN_GAIN="${BENCHMARK_MIN_GAIN:-0.03}"
MAX_PEAK_GIB="${MAX_PEAK_GIB:-70}"
CHECKPOINT_EVERY_BATCHES="${CHECKPOINT_EVERY_BATCHES:-100}"
HISTOGRAM_BINS="${HISTOGRAM_BINS:-4096}"
RESERVOIR_SIZE="${RESERVOIR_SIZE:-5000000}"
BENCHMARK_PORT="${BENCHMARK_PORT:-35760}"
FULL_PORT_BASE="${FULL_PORT_BASE:-35770}"

STAGES=(manifest benchmark full)
die() { echo "[ERROR] $*" >&2; exit 1; }

stage_number() {
    local requested="$1" index
    for index in "${!STAGES[@]}"; do
        if [[ "$requested" == "${STAGES[$index]}" ]]; then
            echo "$((index + 1))"
            return
        fi
    done
    die "unknown stage '$requested'; expected: ${STAGES[*]}"
}

START_INDEX="$(stage_number "$START_STAGE")"
STOP_INDEX="$(stage_number "$STOP_AFTER_STAGE")"
(( START_INDEX <= STOP_INDEX )) || die "START_STAGE comes after STOP_AFTER_STAGE"
[[ "$PLAN_ONLY" == 0 || "$PLAN_ONLY" == 1 ]] || die "PLAN_ONLY must be 0 or 1"
[[ "$START_SYSTEMD" == 0 || "$START_SYSTEMD" == 1 ]] || die "START_SYSTEMD must be 0 or 1"
[[ "$BENCHMARK_WINDOWS" =~ ^[1-9][0-9]*$ ]] || die "BENCHMARK_WINDOWS must be positive"
[[ "$CHECKPOINT_EVERY_BATCHES" =~ ^[1-9][0-9]*$ ]] \
    || die "CHECKPOINT_EVERY_BATCHES must be positive"
[[ "$HISTOGRAM_BINS" =~ ^[1-9][0-9]*$ ]] && (( HISTOGRAM_BINS >= 200 )) \
    || die "HISTOGRAM_BINS must be at least 200"
[[ "$RESERVOIR_SIZE" =~ ^[1-9][0-9]*$ ]] || die "RESERVOIR_SIZE must be positive"
[[ "$MAX_PEAK_GIB" =~ ^[1-9][0-9]*$ ]] || die "MAX_PEAK_GIB must be a positive integer"

[[ "$BENCHMARK_BATCHES" == "64,96,128" || "$BENCHMARK_BATCHES" == "64,96,128,192" ]] \
    || die "BENCHMARK_BATCHES must be 64,96,128 with optional trailing 192"

cat <<PLAN
[CKA FULL CENSUS 4-GPU CHAIN]
output: $OUT_ROOT
physical GPUs: 0,1,2,3 (four independent world-size-1 workers)
stages: $START_STAGE -> $STOP_AFTER_STAGE
train policy: complete document-bounded Code train manifest; no artificial pilot test holdout
expected exhaustive audit: 4,161,493 windows / 2,083,238,288 retained tokens / 2,058,949,248 eligible tokens
measurement: threshold-free B/T/M distributions, layers 2--9, scales 128/256
pilot 95/97/99 lines: histogram overlays only; no GT/selector is fixed in this pass
storage: compact histograms/sketches + deterministic token-score reservoir/window summaries; no raw hidden
global deterministic token-score reservoir: $RESERVOIR_SIZE occurrences
benchmark: one checkpoint load on GPU0, same $BENCHMARK_WINDOWS full-length windows, batches [$BENCHMARK_BATCHES]
batch 192 gate: only after >3% gain from 96 to 128 and safe HBM
HBM safety ceiling: ${MAX_PEAK_GIB} GiB
full ports: $FULL_PORT_BASE..$((FULL_PORT_BASE + 3)); benchmark port: $BENCHMARK_PORT
systemd service: $SYSTEMD_SERVICE (START_SYSTEMD=$START_SYSTEMD)
chain log: $CHAIN_LOG

1. Build and validate the exhaustive Code-train document-window manifest (CPU).
2. In one model process benchmark 64/96/128; try 192 only after >3% gain and HBM headroom; choose max sustained windows/s (GPU0).
3. Launch four disjoint, restartable workers on GPUs 0--3 and verify exact full-manifest coverage.
4. Inspect merged distributions, then lock a human-selected threshold before any later GT pass.
PLAN

if [[ "$PLAN_ONLY" == 1 ]]; then
    echo "[PLAN COMMAND] $PYTHON_BIN $CENSUS_TOOL build-manifest --dataset-prefix $CODE_PREFIX --pilot-manifest $PILOT_MANIFEST --output-dir $MANIFEST_DIR --enforce-known-code-counts"
    echo "[PLAN COMMAND] $PYTHON_BIN $CENSUS_TOOL validate-manifest --manifest $CENSUS_MANIFEST --dataset-prefix $CODE_PREFIX --enforce-known-code-counts"
    echo "[PLAN BENCHMARK] GPU=0 worker=0/1 one_process_batches=$BENCHMARK_BATCHES max_full_length_windows=$BENCHMARK_WINDOWS port=$BENCHMARK_PORT"
    for worker in 0 1 2 3; do
        echo "[PLAN FULL] GPU=$worker worker=$worker/4 port=$((FULL_PORT_BASE + worker)) selected_batch=<benchmark result>"
    done
    echo "[PLAN SYSTEMD] START_SYSTEMD=1 bash $SELF"
    exit 0
fi

if [[ "$START_SYSTEMD" == 1 && "${CKA_FULL_CENSUS_INSIDE_SYSTEMD:-0}" != 1 ]]; then
    command -v systemd-run >/dev/null 2>&1 || die "systemd-run is unavailable"
    exec systemd-run --user --unit "$SYSTEMD_SERVICE" --collect \
        --description "Full Code-train streaming CKA census on GPUs 0-3" \
        --setenv CKA_FULL_CENSUS_INSIDE_SYSTEMD=1 \
        --setenv START_SYSTEMD=0 \
        --setenv START_STAGE="$START_STAGE" \
        --setenv STOP_AFTER_STAGE="$STOP_AFTER_STAGE" \
        /usr/bin/bash "$SELF"
fi

for path in "$PYTHON_BIN" "$RUNNER" "$CENSUS_TOOL" "$ANALYSIS_CONFIG" \
        "$PILOT_MANIFEST" "$CODE_PREFIX.idx" "$CODE_PREFIX.bin"; do
    [[ -e "$path" ]] || die "missing required file: $path"
done
[[ -x "$RUNNER" ]] || die "runner is not executable: $RUNNER"
mkdir -p "$LOG_ROOT" "$STATE_ROOT" "$BENCHMARK_ROOT"
exec 9>"$LOCK_FILE"
flock -n 9 || die "another full CKA census launcher holds $LOCK_FILE"
exec > >(tee -a "$CHAIN_LOG") 2>&1

record() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$1] ${*:2}"; }

assert_gpu_idle() {
    local gpu="$1" applications
    [[ "$gpu" =~ ^[0-3]$ ]] || die "launcher GPU must be physical 0..3"
    applications="$(nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null)" \
        || die "cannot query GPU $gpu"
    applications="$(printf '%s\n' "$applications" | sed '/^[[:space:]]*$/d; /No running processes found/d')"
    if [[ -n "$applications" ]]; then
        echo "[OCCUPIED] GPU $gpu:" >&2
        printf '%s\n' "$applications" >&2
        die "GPU $gpu must be idle; no occupying process will be terminated"
    fi
}

assert_all_gpus_idle() {
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi unavailable"
    for gpu in 0 1 2 3; do assert_gpu_idle "$gpu"; done
}

port_is_free() {
    "$PYTHON_BIN" - "$1" <<'PY'
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("127.0.0.1", int(sys.argv[1])))
finally:
    s.close()
PY
}

mark_stage() {
    local stage="$1"
    "$PYTHON_BIN" - "$STATE_ROOT/${stage}.done.json" "$stage" <<'PY'
import json, os, socket, sys, tempfile, time
path, stage = sys.argv[1:]
payload = {"schema": "cka_gt_full_census_stage_v1", "stage": stage,
           "complete": True, "host": socket.gethostname(), "pid": os.getppid(),
           "completed_unix_time": time.time()}
fd, temporary = tempfile.mkstemp(prefix=os.path.basename(path)+".", dir=os.path.dirname(path))
with os.fdopen(fd, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(temporary, path)
PY
}

start_gpu_monitor() {
    local label="$1" output="$2"
    mkdir -p "$(dirname "$output")"
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
        --format=csv,noheader,nounits -i 0,1,2,3 -l 5 > "$output" 2>&1 &
    GPU_MONITOR_PID=$!
    record MONITOR "$label pid=$GPU_MONITOR_PID output=$output"
}

stop_gpu_monitor() {
    if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
        unset GPU_MONITOR_PID
    fi
}
trap stop_gpu_monitor EXIT INT TERM

run_worker() {
    local gpu="$1" worker="$2" count="$3" batch="$4" max_windows="$5" \
        port="$6" output="$7" log="$8"
    GPU="$gpu" WORKER_INDEX="$worker" WORKER_COUNT="$count" \
        WINDOW_BATCH_SIZE="$batch" MAX_WINDOWS="$max_windows" MASTER_PORT="$port" \
        OUT_ROOT="$OUT_ROOT" CENSUS_OUTPUT="$output" CENSUS_MANIFEST="$CENSUS_MANIFEST" \
        ANALYSIS_CONFIG="$ANALYSIS_CONFIG" CODE_PREFIX="$CODE_PREFIX" LOG_PATH="$log" \
        CHECKPOINT_EVERY_BATCHES="$CHECKPOINT_EVERY_BATCHES" HISTOGRAM_BINS="$HISTOGRAM_BINS" \
        RESERVOIR_SIZE="$RESERVOIR_SIZE" \
        BENCHMARK_BATCH_SIZES="${BENCHMARK_BATCH_SIZES_FOR_RUN:-}" \
        BENCHMARK_MIN_GAIN="$BENCHMARK_MIN_GAIN" BENCHMARK_MAX_PEAK_GIB="$MAX_PEAK_GIB" \
        bash "$RUNNER"
}

if (( START_INDEX <= 1 && STOP_INDEX >= 1 )); then
    record START "building/validating exhaustive train manifest"
    "$PYTHON_BIN" "$CENSUS_TOOL" build-manifest \
        --dataset-prefix "$CODE_PREFIX" --pilot-manifest "$PILOT_MANIFEST" \
        --output-dir "$MANIFEST_DIR" --enforce-known-code-counts
    "$PYTHON_BIN" "$CENSUS_TOOL" validate-manifest \
        --manifest "$CENSUS_MANIFEST" --dataset-prefix "$CODE_PREFIX" \
        --enforce-known-code-counts
    mark_stage manifest
    record DONE "manifest=$CENSUS_MANIFEST"
fi

[[ -f "$CENSUS_MANIFEST" ]] || die "full manifest missing: $CENSUS_MANIFEST"

if (( START_INDEX <= 2 && STOP_INDEX >= 2 )); then
    assert_all_gpus_idle
    port_is_free "$BENCHMARK_PORT" || die "benchmark port $BENCHMARK_PORT is occupied"
    start_gpu_monitor benchmark "$BENCHMARK_ROOT/gpu_usage.csv"
    record START "single-load in-process benchmark batches=$BENCHMARK_BATCHES windows=$BENCHMARK_WINDOWS"
    BENCHMARK_BATCH_SIZES_FOR_RUN="$BENCHMARK_BATCHES"
    run_worker 0 0 1 64 "$BENCHMARK_WINDOWS" "$BENCHMARK_PORT" "$BENCHMARK_ROOT" \
        "$BENCHMARK_ROOT/benchmark.log"
    unset BENCHMARK_BATCH_SIZES_FOR_RUN
    stop_gpu_monitor
    [[ -f "$SELECTED_BATCH_JSON" ]] \
        || die "in-process benchmark did not produce $SELECTED_BATCH_JSON"
    selected_now="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_batch_size"])' "$SELECTED_BATCH_JSON")"
    record DONE "single-load in-process benchmark selected batch=$selected_now"
    mark_stage benchmark
fi

[[ -f "$SELECTED_BATCH_JSON" ]] || die "selected batch record missing: $SELECTED_BATCH_JSON"
SELECTED_BATCH="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_batch_size"])' "$SELECTED_BATCH_JSON")"

if (( START_INDEX <= 3 && STOP_INDEX >= 3 )); then
    assert_all_gpus_idle
    for worker in 0 1 2 3; do
        port_is_free "$((FULL_PORT_BASE + worker))" \
            || die "full worker port $((FULL_PORT_BASE + worker)) is occupied"
    done
    start_gpu_monitor full "$GPU_USAGE_LOG"
    record START "full census batch=$SELECTED_BATCH on GPUs 0-3"
    pids=()
    for worker in 0 1 2 3; do
        run_worker "$worker" "$worker" 4 "$SELECTED_BATCH" 0 \
            "$((FULL_PORT_BASE + worker))" "$FULL_OUTPUT" \
            "$FULL_OUTPUT/logs/worker_$(printf '%03d' "$worker").log" &
        pids+=("$!")
        record PID "worker=$worker gpu=$worker pid=${pids[-1]} port=$((FULL_PORT_BASE + worker))"
    done
    failed=0
    for worker in 0 1 2 3; do
        set +e
        wait "${pids[$worker]}"
        rc=$?
        set -e
        if (( rc != 0 )); then
            record FAIL "worker=$worker rc=$rc log=$FULL_OUTPUT/logs/worker_$(printf '%03d' "$worker").log"
            failed=1
        else
            record DONE "worker=$worker"
        fi
    done
    stop_gpu_monitor
    (( failed == 0 )) || die "one or more full CKA census workers failed"

    "$PYTHON_BIN" "$CENSUS_TOOL" merge-workers \
        --output-dir "$FULL_OUTPUT" --worker-count 4

    "$PYTHON_BIN" - "$CENSUS_MANIFEST" "$FULL_OUTPUT" "$OUT_ROOT/full_verified.json" <<'PY'
import json, os, sys, tempfile, time
import numpy as np
manifest_path, root, output = sys.argv[1:]
with open(manifest_path, encoding="utf-8") as f: manifest=json.load(f)
windows_path=os.path.join(os.path.dirname(manifest_path), manifest.get("windows_file","windows.npy"))
rows=np.load(windows_path,mmap_mode="r",allow_pickle=False)
summaries=[]
for worker in range(4):
    path=os.path.join(root,f"worker_{worker:03d}","summary.json")
    with open(path,encoding="utf-8") as f: value=json.load(f)
    if not value.get("complete"): raise SystemExit(f"worker {worker} is incomplete")
    summaries.append(value)
processed=sum(int(v["processed_windows"]) for v in summaries)
if processed != int(rows.shape[0]):
    raise SystemExit(f"coverage mismatch: processed={processed} manifest={rows.shape[0]}")
payload={"schema":"cka_gt_full_census_full_verification_v1","complete":True,
         "manifest_windows":int(rows.shape[0]),"processed_windows":processed,
         "processed_retained_tokens":sum(int(v["processed_retained_tokens"]) for v in summaries),
         "processed_eligible_tokens":sum(int(v["processed_eligible_tokens"]) for v in summaries),
         "worker_summaries":[os.path.join(root,f"worker_{w:03d}","summary.json") for w in range(4)],
         "verified_unix_time":time.time()}
fd,tmp=tempfile.mkstemp(prefix=os.path.basename(output)+".",dir=os.path.dirname(output))
with os.fdopen(fd,"w",encoding="utf-8") as f:
    json.dump(payload,f,indent=2,sort_keys=True);f.write("\n");f.flush();os.fsync(f.fileno())
os.replace(tmp,output)
print(json.dumps(payload,sort_keys=True))
PY
    mark_stage full
    record DONE "full exact-coverage verification complete: $OUT_ROOT/full_verified.json"
fi

record COMPLETE "requested chain stages finished; output=$OUT_ROOT"
