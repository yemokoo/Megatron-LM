#!/bin/bash
# Qwen3-0.6B zero-shot TRACE evaluation alongside memory-heavy GPU jobs.
# One base-model replica per GPU; four disjoint tasks each. Existing result files
# are skipped so a failed worker can be restarted safely with a smaller batch.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN=${PYTHON_BIN:-/home/work/Agent_HJ/30_flame_agent/envs/train_env/bin/python}
MODEL=${MODEL:-/home/work/Agent_HJ/00_models/Qwen3-0.6B}
DATA=${DATA:-data/LLM-CL-Benchmark_5000}
OUT=${OUT:-eval_out/qwen06_zeroshot_trace}
GPUS=${GPUS:-0,1}
EVAL_BATCH=${EVAL_BATCH:-8}
TASKS=(C-STANCE FOMC MeetingBank Py150 ScienceQA NumGLUE-cm NumGLUE-ds 20Minuten)

IFS=',' read -r -a gpu_array <<<"$GPUS"
[ "${#gpu_array[@]}" -eq 2 ] || { echo "GPUS must contain two IDs" >&2; exit 2; }
[ -x "$PYTHON_BIN" ] || { echo "Python not executable: $PYTHON_BIN" >&2; exit 2; }
[ -d "$MODEL" ] || { echo "Model missing: $MODEL" >&2; exit 2; }

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$OUT/logs"

run_worker() {
  local slot=$1 gpu=$2
  local assigned=() pending=() task
  for ((i=slot; i<${#TASKS[@]}; i+=2)); do assigned+=("${TASKS[$i]}"); done
  for task in "${assigned[@]}"; do
    [ -s "$OUT/results-$task.json" ] || pending+=("$task")
  done
  if [ "${#pending[@]}" -eq 0 ]; then
    echo "[GPU $gpu] all assigned tasks already complete"
    return 0
  fi
  local csv
  csv=$(IFS=,; echo "${pending[*]}")
  echo "[GPU $gpu] batch=$EVAL_BATCH tasks=$csv"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" evaluate_Ours_LoRA_MoE.py \
    --base_only \
    --base_model_name_or_path "$MODEL" \
    --data_path "$DATA" \
    --inference_tasks "$csv" \
    --inference_output_path "$OUT" \
    --max_prompt_len 1536 --max_ans_len 512 \
    --per_device_eval_batch_size "$EVAL_BATCH" \
    --temperature 0.0 \
    --summary_filename "summary-worker$slot.json" \
    >"$OUT/logs/worker$slot.log" 2>&1
}

run_worker 0 "${gpu_array[0]}" & p0=$!
run_worker 1 "${gpu_array[1]}" & p1=$!
status=0
wait "$p0" || status=1
wait "$p1" || status=1
[ "$status" -eq 0 ] || {
  echo "A worker failed. Inspect $OUT/logs; retry with EVAL_BATCH=4." >&2
  exit 1
}

"$PYTHON_BIN" - "$OUT" "${TASKS[@]}" <<'PY'
import json, os, sys
out, tasks = sys.argv[1], sys.argv[2:]
primary = {"C-STANCE":"accuracy", "FOMC":"accuracy", "MeetingBank":"rouge-L",
           "Py150":"similarity", "ScienceQA":"accuracy", "NumGLUE-cm":"accuracy",
           "NumGLUE-ds":"accuracy", "20Minuten":"rouge-L"}
summary, scores = {}, {}
for task in tasks:
    path = os.path.join(out, f"results-{task}.json")
    if not os.path.isfile(path): raise SystemExit(f"missing result: {path}")
    result = json.load(open(path, encoding="utf-8"))["eval"]
    summary[task] = result
    value = result[primary[task]]
    scores[task] = value if task == "Py150" else value * 100
payload = {"method":"Qwen3-0.6B zero-shot", "final_scores":scores,
           "OP":sum(scores.values())/len(scores), "metrics":summary}
with open(os.path.join(out,"zeroshot_results.json"),"w",encoding="utf-8") as f:
    json.dump(payload,f,ensure_ascii=False,indent=2)
print(json.dumps(payload,ensure_ascii=False,indent=2))
PY
