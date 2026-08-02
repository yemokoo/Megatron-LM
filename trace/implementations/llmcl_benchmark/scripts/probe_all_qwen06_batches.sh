#!/bin/bash
# Probe six Qwen3-0.6B methods, two concurrently (one process/GPU), in 3 waves.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON_BIN:-/home/work/Agent_HJ/30_flame_agent/envs/train_env/bin/python}
MODEL=${MODEL:-/home/work/Agent_HJ/00_models/Qwen3-0.6B}
ROOT=${ROOT:-eval_out/qwen06_batch_probe}
MAX_BATCH=${MAX_BATCH:-512}
BUDGET_MIB=${BUDGET_MIB:-79000}
STEPS=${STEPS:-2}
TASKS=(C-STANCE FOMC MeetingBank Py150 ScienceQA NumGLUE-cm NumGLUE-ds 20Minuten)
WAVES=("seqlora loramoe" "ewc gem" "olora track1")

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$ROOT/parts" "$ROOT/logs"

probe_method() {
  local gpu=$1 method=$2 task
  for task in "${TASKS[@]}"; do
    echo "[$method] probing $task on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/probe_qwen06_batches.py \
      --method "$method" --model_name_or_path "$MODEL" \
      --tasks "$task" --max_batch "$MAX_BATCH" \
      --budget_mib "$BUDGET_MIB" --steps "$STEPS" \
      --data_output_path "/tmp/qwen06_probe_${method}_${task}" \
      --out "$ROOT/parts/${method}__${task}.json" \
      >"$ROOT/logs/${method}__${task}.log" 2>&1
  done
}

for wave in "${WAVES[@]}"; do
  read -r left right <<<"$wave"
  echo "=== $left (GPU0) + $right (GPU1) ==="
  probe_method 0 "$left" & left_pid=$!
  probe_method 1 "$right" & right_pid=$!
  wait "$left_pid"
  wait "$right_pid"
done

"$PY" scripts/summarize_qwen06_batch_probe.py --root "$ROOT"
echo "DONE: $ROOT/summary.json"
