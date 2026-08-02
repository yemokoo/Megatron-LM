#!/bin/bash
# TRACE OP+BWT evaluation for five baselines plus Track1 on a dynamic 2-GPU queue.
set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_ROOT=${TRAIN_ROOT:-output/qwen06_2epoch_fast6}
EVAL_ROOT=${EVAL_ROOT:-eval_out/qwen06_2epoch_fast6_six_models}
MODEL=${MODEL:-/home/work/Agent_HJ/00_models/Qwen3-0.6B}
PYTHON_BIN=${PYTHON_BIN:-/home/work/Agent_HJ/30_flame_agent/envs/train_env/bin/python}
EVAL_BATCH=${EVAL_BATCH:-64}
TEMPERATURE=${TEMPERATURE:-0.0}
# Slow custom/routed models first; the dynamic queue fills each freed GPU.
QUEUE=(track1 loramoe gem olora ewc seqlora)
TABLE_METHODS=seqlora,loramoe,ewc,gem,olora,track1
TASKS=(C-STANCE FOMC MeetingBank Py150 ScienceQA NumGLUE-cm NumGLUE-ds 20Minuten)
ALL_TASKS=$(IFS=,; echo "${TASKS[*]}")

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p "$EVAL_ROOT/logs"

for method in "${QUEUE[@]}"; do
  checkpoint="$TRAIN_ROOT/$method/7"
  if [ ! -s "$checkpoint/pytorch_model.bin" ]; then
    echo "Missing completed checkpoint: $checkpoint" >&2
    exit 2
  fi
  if [ "$method" = track1 ]; then
    test -s "$checkpoint/lora_moe_meta.json" || {
      echo "Missing Track1 metadata: $checkpoint/lora_moe_meta.json" >&2; exit 2; }
  else
    test -s "$checkpoint/paper_baseline_meta.json" || {
      echo "Missing baseline metadata: $checkpoint/paper_baseline_meta.json" >&2; exit 2; }
  fi
done

evaluate_checkpoint() {
  local gpu=$1 checkpoint=$2 tasks=$3 out=$4 log=$5
  mkdir -p "$out" "$(dirname "$log")"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" evaluate_Ours_LoRA_MoE.py \
    --checkpoint_dir "$checkpoint" \
    --base_model_name_or_path "$MODEL" \
    --data_path data/LLM-CL-Benchmark_5000 \
    --inference_tasks "$tasks" \
    --inference_output_path "$out" \
    --max_prompt_len 1536 --max_ans_len 512 \
    --per_device_eval_batch_size "$EVAL_BATCH" \
    --temperature "$TEMPERATURE" \
    >"$log" 2>&1
}

run_method() {
  local gpu=$1 method=$2 task index result
  local method_log="$EVAL_ROOT/logs/${method}_driver.log"
  echo "[START] method=$method GPU=$gpu" | tee "$method_log"

  local final_out="$EVAL_ROOT/$method/final"
  local complete=true
  for task in "${TASKS[@]}"; do
    [ -s "$final_out/results-$task.json" ] || complete=false
  done
  if $complete; then
    echo "[SKIP] final checkpoint already complete" | tee -a "$method_log"
  else
    evaluate_checkpoint "$gpu" "$TRAIN_ROOT/$method/7" "$ALL_TASKS" \
      "$final_out" "$EVAL_ROOT/logs/${method}_final.log"
  fi

  for index in 0 1 2 3 4 5 6; do
    task=${TASKS[$index]}
    local diagonal_out="$EVAL_ROOT/$method/diagonal/$index"
    result="$diagonal_out/results-$task.json"
    if [ -s "$result" ]; then
      echo "[SKIP] diagonal=$index task=$task" | tee -a "$method_log"
      continue
    fi
    echo "[EVAL] diagonal=$index task=$task" | tee -a "$method_log"
    evaluate_checkpoint "$gpu" "$TRAIN_ROOT/$method/$index" "$task" \
      "$diagonal_out" \
      "$EVAL_ROOT/logs/${method}_diagonal_${index}_${task}.log"
  done
  echo "[DONE] method=$method GPU=$gpu" | tee -a "$method_log"
}

declare -A PID_GPU PID_METHOD
next=0
running=0

launch_next() {
  local gpu=$1 method pid
  if [ "$next" -ge "${#QUEUE[@]}" ]; then return 1; fi
  method=${QUEUE[$next]}
  next=$((next + 1))
  echo "=== QUEUE: $method -> GPU$gpu ==="
  run_method "$gpu" "$method" & pid=$!
  PID_GPU[$pid]=$gpu
  PID_METHOD[$pid]=$method
  running=$((running + 1))
}

launch_next 0
launch_next 1
while [ "$running" -gt 0 ]; do
  finished_pid=""; rc=0
  wait -n -p finished_pid || rc=$?
  gpu=${PID_GPU[$finished_pid]}
  method=${PID_METHOD[$finished_pid]}
  unset 'PID_GPU[$finished_pid]' 'PID_METHOD[$finished_pid]'
  running=$((running - 1))
  if [ "$rc" -ne 0 ]; then
    echo "Evaluation failed: method=$method GPU=$gpu rc=$rc" >&2
    exit "$rc"
  fi
  launch_next "$gpu" || true
done

"$PYTHON_BIN" scripts/summarize_five_baselines.py \
  --eval_root "$EVAL_ROOT" \
  --methods "$TABLE_METHODS" \
  --output_prefix "$EVAL_ROOT/six_model_results" \
  | tee "$EVAL_ROOT/logs/summary.log"

echo "ALL SIX MODELS EVALUATED: $EVAL_ROOT"
