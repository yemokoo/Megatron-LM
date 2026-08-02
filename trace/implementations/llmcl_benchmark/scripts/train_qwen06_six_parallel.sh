#!/bin/bash
# Train five paper baselines + Track1 with a two-GPU work queue, one A100 per model.
# Requires the completed output of scripts/probe_all_qwen06_batches.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-/home/work/Agent_HJ/00_models/Qwen3-0.6B}
PROBE_SUMMARY=${PROBE_SUMMARY:-eval_out/qwen06_batch_probe/summary.json}
ROOT=${ROOT:-output/qwen06_2epoch_fast6}
LOG_ROOT=${LOG_ROOT:-$ROOT/logs}
PY=${PYTHON_BIN:-/home/work/Agent_HJ/30_flame_agent/envs/train_env/bin/python}
EPOCHS=${EPOCHS:-2,2,2,2,2,2,2,2}
TRACK1_EXPERTS_PER_TASK=${TRACK1_EXPERTS_PER_TASK:-1}
METHODS=(seqlora loramoe ewc gem olora track1)

if [ ! -f "$PROBE_SUMMARY" ]; then
  echo "Missing $PROBE_SUMMARY; run: bash scripts/probe_all_qwen06_batches.sh" >&2
  exit 2
fi
BATCH=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["recommended_common_csv"])' "$PROBE_SUMMARY")
mkdir -p "$ROOT" "$LOG_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_method() {
  local gpu=$1 method=$2 out="$ROOT/$2" log="$LOG_ROOT/$2.log" rc
  if [ -f "$out/7/paper_baseline_meta.json" ] || \
     { [ "$method" = track1 ] && [ -f "$out/7/config.json" ]; }; then
    echo "[SKIP] $method already completed: $out"
    return
  fi
  echo "[START] $method GPU=$gpu batch=$BATCH log=$log"
  if [ "$method" = track1 ]; then
    GPUS=$gpu OUT=$out EPOCHS=$EPOCHS BATCH=$BATCH \
      EXPERTS_PER_TASK=$TRACK1_EXPERTS_PER_TASK \
      WEIGHT_DECAY=0.01 ADAM_EPSILON=1e-6 LR_SCHEDULER_TYPE=cosine \
      bash scripts/train_Ours_LoRA_MoE.sh "$TRACK1_EXPERTS_PER_TASK" "$MODEL" \
      >"$log" 2>&1 || { rc=$?; echo "[FAIL] $method rc=$rc" >&2; return "$rc"; }
  else
    local resume="" last=-1 index
    for index in 0 1 2 3 4 5 6 7; do
      if [ -f "$out/$index/paper_baseline_meta.json" ]; then
        last=$index
      else
        break
      fi
    done
    if [ "$last" -ge 0 ]; then resume="$out/$last"; fi
    GPUS=$gpu OUT=$out EPOCHS=$EPOCHS BATCH=$BATCH \
      LORAMOE_EXPERTS=4 EWC_LAMBDA="${EWC_LAMBDA:-400}" \
      RESUME_CHECKPOINT=$resume WEIGHT_DECAY=0.01 ADAM_EPSILON=1e-6 \
      LR_SCHEDULER_TYPE=cosine \
      bash scripts/train_paper_baseline.sh "$method" "$MODEL" \
      >"$log" 2>&1 || { rc=$?; echo "[FAIL] $method rc=$rc" >&2; return "$rc"; }
  fi
  echo "[DONE] $method"
}

declare -A PID_TO_GPU PID_TO_METHOD
next=0
running=0

launch_next() {
  local gpu=$1 method pid
  if [ "$next" -ge "${#METHODS[@]}" ]; then return 1; fi
  method=${METHODS[$next]}
  next=$((next + 1))
  echo "=== QUEUE: $method -> GPU$gpu ==="
  run_method "$gpu" "$method" & pid=$!
  PID_TO_GPU[$pid]=$gpu
  PID_TO_METHOD[$pid]=$method
  running=$((running + 1))
}

launch_next 0 || true
launch_next 1 || true
while [ "$running" -gt 0 ]; do
  finished_pid=""
  # Bash 5.1 supports -p: identify which queued job completed first.
  wait -n -p finished_pid || rc=$?
  rc=${rc:-0}
  gpu=${PID_TO_GPU[$finished_pid]}
  method=${PID_TO_METHOD[$finished_pid]}
  unset 'PID_TO_GPU[$finished_pid]' 'PID_TO_METHOD[$finished_pid]'
  running=$((running - 1))
  if [ "$rc" -ne 0 ]; then
    echo "Training failed: $method on GPU$gpu rc=$rc" >&2
    exit "$rc"
  fi
  echo "=== QUEUE SLOT FREE: GPU$gpu after $method ==="
  unset rc
  launch_next "$gpu" || true
done
echo "ALL SIX COMPLETE: $ROOT"
