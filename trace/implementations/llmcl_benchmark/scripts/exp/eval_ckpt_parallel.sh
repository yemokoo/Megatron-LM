#!/bin/bash
# Evaluate ONE grown Ours_LoRA_MoE checkpoint on several TRACE tasks. Each GPU
# loads one model copy and handles its assigned tasks sequentially. This avoids
# launching multiple 8B model processes on the same GPU when tasks > GPUs and
# amortizes base/checkpoint loading across those tasks.
#
# Usage: bash scripts/exp/eval_ckpt_parallel.sh <checkpoint_dir> <base_model> <out_dir> [task1,task2,...] [GPUS]
#   e.g. bash scripts/exp/eval_ckpt_parallel.sh \
#          output/track1_Qwen3-8B_ept1/3 \
#          /home/work/Agent_HJ/00_models/Qwen3-8B \
#          eval_out/track1_8b_round3 \
#          C-STANCE,FOMC,MeetingBank,Py150
#
# Watch progress:  tail -f <out_dir>/eval_gpu<id>.log
set -e
cd "$(dirname "$0")/../.."

CKPT="$1"
BASE="$2"
OUT="$3"
TASKS="${4:-C-STANCE,FOMC,MeetingBank,Py150}"
GPUS="${5:-0,1,2,3}"

if [ -z "$CKPT" ] || [ -z "$BASE" ] || [ -z "$OUT" ]; then
  echo "Usage: $0 <checkpoint_dir> <base_model> <out_dir> [tasks] [gpus]"
  exit 1
fi

export PATH=/home/work/Agent_HJ/30_flame_agent/envs/train_env/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p "$OUT"

IFS=',' read -ra TASK_ARR <<< "$TASKS"
IFS=',' read -ra GPU_ARR  <<< "$GPUS"
NGPU=${#GPU_ARR[@]}

echo "checkpoint : $CKPT"
echo "base model : $BASE"
echo "tasks      : ${TASK_ARR[*]}"
echo "gpus       : ${GPU_ARR[*]} ($NGPU)"
echo "out        : $OUT"
echo

# Assign tasks round-robin, then launch exactly one evaluator per GPU. A worker
# receives a comma-separated task list and therefore loads the model only once.
declare -a GPU_TASKS
for i in "${!TASK_ARR[@]}"; do
  slot=$((i % NGPU))
  if [ -n "${GPU_TASKS[$slot]}" ]; then
    GPU_TASKS[$slot]="${GPU_TASKS[$slot]},${TASK_ARR[$i]}"
  else
    GPU_TASKS[$slot]="${TASK_ARR[$i]}"
  fi
done

pids=()
for slot in "${!GPU_ARR[@]}"; do
  assigned="${GPU_TASKS[$slot]}"
  [ -z "$assigned" ] && continue
  gpu="${GPU_ARR[$slot]}"
  log="$OUT/eval_gpu${gpu}.log"
  echo "[GPU $gpu] -> $assigned   (log: $log)"
  CUDA_VISIBLE_DEVICES="$gpu" python evaluate_Ours_LoRA_MoE.py \
    --checkpoint_dir "$CKPT" \
    --base_model_name_or_path "$BASE" \
    --data_path data/LLM-CL-Benchmark_5000 \
    --inference_tasks "$assigned" \
    --inference_output_path "$OUT" \
    --summary_filename "summary_gpu${gpu}.json" \
    ${LIMIT_FRAC:+--limit_frac "$LIMIT_FRAC"} \
    > "$log" 2>&1 &
  pids+=($!)
done

echo
echo "launched ${#pids[@]} eval jobs; waiting..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

echo
echo "================ RESULTS ($OUT) ================"
python - "$OUT" "${TASK_ARR[@]}" <<'PY'
import json, os, sys
out = sys.argv[1]; tasks = sys.argv[2:]
summary = {}
for t in tasks:
    fp = os.path.join(out, f"results-{t}.json")
    if os.path.exists(fp):
        d = json.load(open(fp))
        summary[t] = d.get("eval")
        print(f"{t:14s} n={len(d.get('labels',[]))}  {d.get('eval')}")
    else:
        print(f"{t:14s} (missing -- check {out}/eval_gpu*.log)")
json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2, ensure_ascii=False)
vals = [v.get("accuracy") for v in summary.values()
        if isinstance(v, dict) and isinstance(v.get("accuracy"), (int, float))]
if vals:
    print(f"\nmean accuracy (tasks with 'accuracy'): {sum(vals)/len(vals):.4f}")
print(f"\nsaved -> {os.path.join(out, 'summary.json')}")
PY
[ "$fail" -eq 0 ] && echo "ALL EVAL JOBS OK" || echo "SOME EVAL JOBS FAILED (see per-task logs)"
