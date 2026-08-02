#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-output/track2_OLMoE_ept1_force_upper_5k_seed1234}"
EVAL="${EVAL:-eval_out/track2_OLMoE_ept1_force_upper_5k_seed1234_sparse}"
MODEL="${MODEL:-/home/work/Agent_HJ/00_models/OLMoE-1B-7B-0125}"
DATA="${DATA:-data/LLM-CL-Benchmark_5000}"
PY="${PYTHON_BIN:-/home/work/Agent_HJ/30_flame_agent/envs/train_env/bin/python}"
BATCH="${EVAL_BATCH:-4}"

TASKS=(C-STANCE FOMC MeetingBank Py150 ScienceQA NumGLUE-cm NumGLUE-ds 20Minuten)
mkdir -p "$EVAL/final" "$EVAL/diagonal" "$EVAL/logs"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

run_one() {
  local gpu="$1" ckpt="$2" task="$3" out="$4" log="$5"
  if [[ -s "$out/results-$task.json" ]]; then
    echo "[SKIP] $out/results-$task.json"
    return
  fi
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" evaluate_Ours_MoE_FFN.py \
    --checkpoint_dir "$ckpt" \
    --base_model_name_or_path "$MODEL" \
    --data_path "$DATA" \
    --inference_tasks "$task" \
    --inference_output_path "$out" \
    --max_prompt_len 1536 --max_ans_len 512 \
    --per_device_eval_batch_size "$BATCH" \
    --task_generation_limits --with_sari \
    --temperature 0 --length_bucketing \
    >"$log" 2>&1
}

# Lane 0: load the final checkpoint once and evaluate all eight tasks.
CUDA_VISIBLE_DEVICES=0 "$PY" evaluate_Ours_MoE_FFN.py \
  --checkpoint_dir "$ROOT/7" \
  --base_model_name_or_path "$MODEL" \
  --data_path "$DATA" \
  --inference_tasks "$(IFS=,; echo "${TASKS[*]}")" \
  --inference_output_path "$EVAL/final" \
  --max_prompt_len 1536 --max_ans_len 512 \
  --per_device_eval_batch_size "$BATCH" \
  --task_generation_limits --with_sari \
  --temperature 0 --length_bucketing \
  >"$EVAL/logs/final.log" 2>&1 &
p0=$!

# Lane 1: only the score immediately after each task was learned.
(
  for i in {0..6}; do
    task="${TASKS[$i]}"
    run_one 1 "$ROOT/$i" "$task" "$EVAL/diagonal/$i" \
      "$EVAL/logs/diagonal_${i}_${task}.log"
  done
) &
p1=$!

rc=0
wait "$p0" || rc=$?
wait "$p1" || rc=$?
if (( rc != 0 )); then
  echo "OLMoE sparse evaluation failed (rc=$rc)" >&2
  exit "$rc"
fi

"$PY" - "$EVAL" <<'PY'
import json, os, sys
root = sys.argv[1]
tasks = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
         "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]
metric = {"C-STANCE":"accuracy", "FOMC":"accuracy", "MeetingBank":"rouge-L",
          "Py150":"similarity", "ScienceQA":"accuracy", "NumGLUE-cm":"accuracy",
          "NumGLUE-ds":"accuracy", "20Minuten":"sari"}
def scalar(task, result):
    value = float(result[metric[task]])
    return value if task in ("Py150", "20Minuten") else value * 100.0
final, learned, full = {}, {}, {}
for i, task in enumerate(tasks):
    path = os.path.join(root, "final", f"results-{task}.json")
    result = json.load(open(path, encoding="utf-8"))["eval"]
    full[task] = result
    final[task] = scalar(task, result)
    if i < 7:
        path = os.path.join(root, "diagonal", str(i), f"results-{task}.json")
        result = json.load(open(path, encoding="utf-8"))["eval"]
        learned[task] = scalar(task, result)
bwt_terms = {t: final[t] - learned[t] for t in tasks[:-1]}
payload = {
    "protocol": {"max_prompt_len": 1536, "temperature": 0,
                 "task_generation_limits": True, "20Minuten_metric": "SARI",
                 "BWT": "mean(final-after_learned), tasks 0..6"},
    "OP": sum(final.values()) / len(tasks),
    "BWT": sum(bwt_terms.values()) / len(bwt_terms),
    "final_scores": final, "learned_scores": learned,
    "bwt_per_task": bwt_terms, "full_final_metrics": full,
}
with open(os.path.join(root, "cl_summary_sparse.json"), "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

echo "OLMOE_SPARSE_EVALUATION_COMPLETE"
