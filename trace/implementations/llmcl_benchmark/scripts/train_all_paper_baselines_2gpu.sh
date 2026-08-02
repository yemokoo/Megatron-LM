#!/bin/bash
# Sequentially train the five non-MH-MoE TRACE baselines on two GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN=${PYTHON_BIN:-/home/work/Agent_HJ/30_flame_agent/envs/train_env/bin/python}
MODEL=${MODEL:-/home/work/Agent_HJ/00_models/Qwen3-8B}
GPUS=${GPUS:-0,1}
EPOCHS=${EPOCHS:-2,2,2,2,2,2,2,2}
BATCH=${BATCH:-10,6,8,8,12,18,26,8}
CKPT_TASKS=${CKPT_TASKS:-MeetingBank,Py150,ScienceQA,20Minuten}
METHODS_CSV=${METHODS:-seqlora,loramoe,ewc,gem,olora}
ROOT=${ROOT:-output/paper_baselines_Qwen3-8B_r8_2epoch_2gpu}
LOG_DIR=${LOG_DIR:-$ROOT/logs}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
PLAN_ONLY=${PLAN_ONLY:-0}

if [ "$GPUS" != "0,1" ]; then
  echo "warning: requested GPUS=$GPUS (the validated/default setting is 0,1)"
fi
IFS=',' read -r -a METHODS_ARRAY <<< "$METHODS_CSV"
mkdir -p "$ROOT" "$LOG_DIR"

write_aggregate_result() {
  "$PYTHON_BIN" - "$ROOT" "$MODEL" "$GPUS" "$EPOCHS" "$BATCH" "$METHODS_CSV" <<'PY'
import json, os, sys
from datetime import datetime, timezone

root, model, gpus, epochs, batch, methods_csv = sys.argv[1:]
methods = [item for item in methods_csv.split(',') if item]
entries = []
for method in methods:
    path = os.path.join(root, method, 'result.json')
    if os.path.isfile(path):
        result = json.load(open(path))
        entries.append({
            'method': method,
            'status': result['status'],
            'result': path,
            'final_checkpoint': os.path.join(root, method, '7'),
            'totals': result['totals'],
        })
    else:
        entries.append({'method': method, 'status': 'pending', 'result': path})
completed = [item for item in entries if item['status'] == 'completed']
payload = {
    'schema_version': 1,
    'status': 'completed' if len(completed) == len(methods) else 'running',
    'model_name_or_path': model,
    'gpus': gpus.split(','),
    'world_size': len(gpus.split(',')),
    'epochs': epochs,
    'per_device_batch_size': batch,
    'method_order': methods,
    'methods': entries,
    'totals': {
        'training_seconds': sum(x['totals']['training_seconds'] for x in completed),
        'gpu_seconds': sum(x['totals']['gpu_seconds'] for x in completed),
        'tokens': sum(x['totals']['tokens'] for x in completed),
        'estimated_flops': sum(x['totals']['estimated_flops'] for x in completed),
        'estimated_tflops': sum(x['totals']['estimated_tflops'] for x in completed),
    },
    'updated_at_utc': datetime.now(timezone.utc).isoformat(),
}
path = os.path.join(root, 'result.json')
temporary = path + '.tmp'
with open(temporary, 'w') as handle:
    json.dump(payload, handle, indent=2)
os.replace(temporary, path)
PY
}

verify_method() {
  local method=$1
  local out=$2
  "$PYTHON_BIN" - "$method" "$out" <<'PY'
import json, os, sys
method, out = sys.argv[1:]
path = os.path.join(out, 'result.json')
result = json.load(open(path))
assert result['status'] == 'completed', result['status']
assert result['method'] == method and result['world_size'] == 2
assert len(result['tasks']) == 8
assert result['totals']['training_seconds'] > 0
assert result['totals']['tokens'] > 0
assert result['totals']['estimated_flops'] > 0
for task_index in range(8):
    checkpoint = os.path.join(out, str(task_index))
    assert os.path.isfile(os.path.join(checkpoint, 'pytorch_model.bin')), checkpoint
    assert os.path.isfile(os.path.join(checkpoint, 'paper_baseline_meta.json')), checkpoint
print('FULL_RESULT_OK', method, result['totals'])
PY
}

write_aggregate_result
for METHOD in "${METHODS_ARRAY[@]}"; do
  OUT="$ROOT/$METHOD"
  LOG="$LOG_DIR/$METHOD.log"
  if [ "$SKIP_COMPLETED" = 1 ] && [ -f "$OUT/result.json" ] && \
     "$PYTHON_BIN" - "$OUT/result.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get('status') == 'completed' else 1)
PY
  then
    echo "[SKIP] $METHOD already completed: $OUT"
    verify_method "$METHOD" "$OUT"
    continue
  fi

  echo "[START] $METHOD at $(date --iso-8601=seconds)"
  echo "        output=$OUT log=$LOG"
  if [ "$PLAN_ONLY" = 1 ]; then
    echo "PLAN_ONLY GPUS=$GPUS EPOCHS=$EPOCHS BATCH=$BATCH method=$METHOD"
    continue
  fi

  RESUME_CHECKPOINT=""
  if [ -f "$OUT/result.json" ]; then
    RESUME_CHECKPOINT=$(
      "$PYTHON_BIN" - "$METHOD" "$OUT" <<'PY'
import json, os, sys
method, out = sys.argv[1:]
result = json.load(open(os.path.join(out, 'result.json')))
assert result.get('method') == method
tasks = result.get('tasks', [])
assert [task.get('task_index') for task in tasks] == list(range(len(tasks)))
if tasks:
    checkpoint = os.path.join(out, str(len(tasks) - 1))
    assert os.path.isfile(os.path.join(checkpoint, 'pytorch_model.bin'))
    assert os.path.isfile(os.path.join(checkpoint, 'paper_baseline_meta.json'))
    print(checkpoint)
PY
    )
  fi
  if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "[RESUME] $METHOD from $RESUME_CHECKPOINT"
  fi

  GPUS="$GPUS" PYTHON_BIN="$PYTHON_BIN" EPOCHS="$EPOCHS" BATCH="$BATCH" \
    CKPT_TASKS="$CKPT_TASKS" OUT="$OUT" \
    RESUME_CHECKPOINT="$RESUME_CHECKPOINT" \
    bash scripts/train_paper_baseline.sh "$METHOD" "$MODEL" 2>&1 | tee "$LOG"
  verify_method "$METHOD" "$OUT"
  write_aggregate_result
  echo "[DONE] $METHOD at $(date --iso-8601=seconds)"
done

write_aggregate_result
if [ "$PLAN_ONLY" = 1 ]; then
  echo "[PLAN DONE] no training was started"
else
  "$PYTHON_BIN" - "$ROOT/result.json" <<'PY'
import json, sys
result=json.load(open(sys.argv[1]))
assert result['status']=='completed', result
print('[ALL DONE]', result['totals'])
PY
fi
