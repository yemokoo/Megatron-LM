#!/bin/bash
# Task-wise OLMoE full-parameter fine-tuning micro-batch survey under ZeRO-2.
# Each task/batch candidate gets a fresh torchrun so a failed CUDA/NCCL process
# cannot contaminate the next measurement.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

MODEL=${MODEL:-/home/work/Agent_HJ/00_models/OLMoE-1B-7B-0125}
GPUS=${GPUS:-0,1}
PYTHON_BIN=${PYTHON_BIN:-/home/work/Agent_HJ/30_flame_agent/envs/train_env/bin/python}
TASKS=${TASKS:-C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten}
MB_CANDIDATES=${MB_CANDIDATES:-1,2,3,4}
STEPS=${STEPS:-2}
BUDGET_MIB=${BUDGET_MIB:-74000}
OUT=${OUT:-eval_out/batch_survey_olmoe_full_ft_zero2.json}
PARTS=${PARTS:-eval_out/batch_survey_olmoe_full_ft_zero2_parts}
LOG_DIR=${LOG_DIR:-logs/batch_survey_olmoe_full_ft_zero2}
PORT_BASE=${PORT_BASE:-29620}
OFFLOAD_OPTIMIZER=${OFFLOAD_OPTIMIZER:-0}

export CUDA_VISIBLE_DEVICES="$GPUS"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
NPROC=$(tr ',' '\n' <<<"$GPUS" | sed '/^$/d' | wc -l)

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [ "$NPROC" -lt 2 ]; then
  echo "ZeRO-2 survey requires at least two GPUs; GPUS=$GPUS" >&2
  exit 1
fi
if pgrep -f 'pretrain_gpt.py|main_Ours_MoE_FFN.py|main_baseline.py|batch_survey_olmoe_full_ft_zero2.py' >/dev/null; then
  echo "Refusing to survey while a known training process is active." >&2
  echo "Set ALLOW_BUSY_GPUS=1 only if the selected GPUS are known to be free." >&2
  if [ "${ALLOW_BUSY_GPUS:-0}" != "1" ]; then
    exit 2
  fi
fi

"$PYTHON_BIN" - <<'PY'
import sys
try:
    import torch, transformers, deepspeed
except Exception as exc:
    raise SystemExit(f"survey environment is incomplete: {exc}")
print(f"environment: python={sys.executable} torch={torch.__version__} "
      f"transformers={transformers.__version__} deepspeed={deepspeed.__version__}")
PY

mkdir -p "$PARTS" "$LOG_DIR" "$(dirname "$OUT")"
rm -f "$PARTS"/*.json
IFS=',' read -r -a task_array <<<"$TASKS"
IFS=',' read -r -a mb_array <<<"$MB_CANDIDATES"
probe_index=0

for task in "${task_array[@]}"; do
  task_safe=${task//[^A-Za-z0-9_.-]/_}
  last_ok=0
  echo "===== $task ====="
  for mb in "${mb_array[@]}"; do
    result="$PARTS/${task_safe}_mb${mb}.json"
    log="$LOG_DIR/${task_safe}_mb${mb}.log"
    port=$((PORT_BASE + probe_index))
    probe_index=$((probe_index + 1))
    offload_args=()
    if [ "$OFFLOAD_OPTIMIZER" = "1" ]; then
      offload_args+=(--offload_optimizer)
    fi
    echo "[$task] probing per-device MB=$mb on $NPROC GPUs (log=$log)"
    set +e
    "$PYTHON_BIN" -m torch.distributed.run \
      --standalone --nnodes=1 --nproc_per_node="$NPROC" \
      --master_port="$port" \
      scripts/batch_survey_olmoe_full_ft_zero2.py \
      --model_name_or_path "$MODEL" \
      --data_path data/LLM-CL-Benchmark_5000 \
      --data_output_path "/tmp/data_files_olmoe_full_ft_survey_${task_safe}_mb${mb}/" \
      --task "$task" --micro_batch "$mb" --steps "$STEPS" \
      --result "$result" "${offload_args[@]}" \
      >"$log" 2>&1
    status=$?
    set -e

    if [ "$status" -eq 0 ] && [ -s "$result" ]; then
      peak=$("$PYTHON_BIN" -c 'import json,sys; print(round(json.load(open(sys.argv[1]))["peak_reserved_mib_max_rank"]))' "$result")
      echo "[$task] MB=$mb OK, peak_reserved=${peak} MiB"
      if "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)' "$peak" "$BUDGET_MIB"; then
        last_ok=$mb
        continue
      fi
      echo "[$task] MB=$mb exceeds safety budget ${BUDGET_MIB} MiB; stopping task"
      break
    fi

    if rg -qi 'CUDA out of memory|OutOfMemoryError|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED' "$log"; then
      echo "[$task] MB=$mb OOM; stopping task"
      break
    fi
    echo "[$task] MB=$mb failed for a non-OOM reason; inspect $log" >&2
    tail -n 60 "$log" >&2
    exit "$status"
  done
  echo "[$task] recommended maximum within budget: MB=$last_ok"
done

"$PYTHON_BIN" - "$PARTS" "$OUT" "$BUDGET_MIB" "$MODEL" "$GPUS" <<'PY'
import glob, json, os, sys
parts, output, budget, model, gpus = sys.argv[1:]
rows = [json.load(open(path)) for path in sorted(glob.glob(os.path.join(parts, "*.json")))]
tasks = {}
for row in rows:
    tasks.setdefault(row["task"], []).append(row)
for probes in tasks.values():
    probes.sort(key=lambda item: item["micro_batch_per_gpu"])
summary = {}
for task, probes in tasks.items():
    safe = [p for p in probes if p["peak_reserved_mib_max_rank"] <= float(budget)]
    summary[task] = {
        "recommended_micro_batch": max((p["micro_batch_per_gpu"] for p in safe), default=0),
        "successful_probes": probes,
    }
payload = {
    "model": model,
    "gpus": gpus,
    "zero_stage": 2,
    "gradient_checkpointing": True,
    "budget_mib": float(budget),
    "tasks": summary,
}
os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
with open(output, "w") as handle:
    json.dump(payload, handle, indent=2)
print("\n===== SUMMARY =====")
for task, value in summary.items():
    print(f"{task:12s} MB={value['recommended_micro_batch']}")
print(f"saved: {output}")
PY
