#!/bin/bash
# Efficient TRACE evaluation for the five completed paper baselines.
# Per method: final checkpoint on all 8 tasks (OP) + checkpoints 0..6 on their
# just-learned task (BWT diagonal), i.e. 15 task evaluations instead of 8x8=64.
set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_ROOT=${TRAIN_ROOT:-output/qwen06_2epoch_fast6}
EVAL_ROOT=${EVAL_ROOT:-eval_out/qwen06_2epoch_fast6_five_baselines}
MODEL=${MODEL:-/home/work/Agent_HJ/00_models/Qwen3-0.6B}
PYTHON_BIN=${PYTHON_BIN:-/home/work/Agent_HJ/30_flame_agent/envs/train_env/bin/python}
GPU=${GPU:-0}
EVAL_BATCH=${EVAL_BATCH:-64}
TEMPERATURE=${TEMPERATURE:-0.0}
METHODS=(seqlora loramoe ewc gem olora)
TASKS=(C-STANCE FOMC MeetingBank Py150 ScienceQA NumGLUE-cm NumGLUE-ds 20Minuten)
ALL_TASKS=$(IFS=,; echo "${TASKS[*]}")

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p "$EVAL_ROOT/logs"

for method in "${METHODS[@]}"; do
  checkpoint="$TRAIN_ROOT/$method/7"
  if [ ! -s "$checkpoint/paper_baseline_meta.json" ]; then
    echo "Missing completed checkpoint: $checkpoint" >&2
    exit 2
  fi
done

evaluate() {
  local checkpoint=$1 tasks=$2 out=$3 log=$4
  mkdir -p "$out" "$(dirname "$log")"
  "$PYTHON_BIN" evaluate_Ours_LoRA_MoE.py \
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

for method in "${METHODS[@]}"; do
  echo "=== EVAL $method: final checkpoint / OP ==="
  final_out="$EVAL_ROOT/$method/final"
  complete=true
  for task in "${TASKS[@]}"; do
    [ -s "$final_out/results-$task.json" ] || complete=false
  done
  if $complete; then
    echo "[SKIP] $method final results already complete"
  else
    evaluate "$TRAIN_ROOT/$method/7" "$ALL_TASKS" "$final_out" \
      "$EVAL_ROOT/logs/${method}_final.log"
  fi

  echo "=== EVAL $method: BWT diagonal checkpoints ==="
  for index in 0 1 2 3 4 5 6; do
    task=${TASKS[$index]}
    diagonal_out="$EVAL_ROOT/$method/diagonal/$index"
    result="$diagonal_out/results-$task.json"
    if [ -s "$result" ]; then
      echo "[SKIP] $method checkpoint $index / $task"
      continue
    fi
    evaluate "$TRAIN_ROOT/$method/$index" "$task" "$diagonal_out" \
      "$EVAL_ROOT/logs/${method}_diagonal_${index}_${task}.log"
  done
done

"$PYTHON_BIN" scripts/summarize_five_baselines.py \
  --eval_root "$EVAL_ROOT" \
  --output_prefix "$EVAL_ROOT/five_baseline_results" \
  | tee "$EVAL_ROOT/logs/summary.log"

echo "ALL FIVE BASELINES EVALUATED: $EVAL_ROOT"
