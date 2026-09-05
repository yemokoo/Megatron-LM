#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace"
RUN="/data2/seonghyeonnoh/LLM-continual-learning-runs/v2_new_retrain_20260807"
LOG="${RUN}/task_expert_py150_8way_then_gpu_hold.log"
mkdir -p "${RUN}/eval_task_expert_logs"
exec >>"${LOG}" 2>&1

cd "${ROOT}"
source "${ROOT}/.venv-runtime/bin/activate"
export PYTHONNOUSERSITE=1
export WANDB_MODE=offline
export TRACE_DATA_ROOT="${ROOT}/data/trace"
export OURS_LORAMOE_OUTPUT_ROOT="${RUN}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[PY150 8WAY START] $(date '+%F %T %Z')"
python "${ROOT}/scripts/run_ours_py150_4way.py" \
  --run-dir "${RUN}" --round 8 --task Py150 --batch 4 \
  --gpus 0,1,2,3,4,5,6,7 \
  --output-subdir evaluation_task_expert/order8 \
  --force-expert-index 3

python - <<'PY'
import json
from pathlib import Path

run = Path('/data2/seonghyeonnoh/LLM-continual-learning-runs/v2_new_retrain_20260807')
out = run / 'evaluation_task_expert/order8'
tasks = ['C-STANCE', 'FOMC', 'MeetingBank', 'Py150', 'ScienceQA',
         'NumGLUE-cm', 'NumGLUE-ds', '20Minuten']
summary = {}
for expert, task in enumerate(tasks):
    path = out / f'results-{task}.json'
    payload = json.loads(path.read_text())
    expected = len(json.loads((Path('/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace/data/trace') / task / 'test.json').read_text()))
    if len(payload.get('results', [])) != expected:
        raise SystemExit(f'incomplete {task}: {len(payload.get("results", []))}/{expected}')
    if payload.get('forced_expert_index') != expert:
        raise SystemExit(f'wrong forced expert for {task}: {payload.get("forced_expert_index")} != {expert}')
    summary[task] = {'forced_expert_index': expert,
                     'eval': payload['eval'], 'samples': expected}
(run / 'task_expert_final_summary.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2))
print('[TASK-EXPERT COMPLETE] all 8 results verified')
PY

echo "[GPU HOLD START] $(date '+%F %T %Z')"
exec python /home/seonghyeonnoh/yemokoo/gpu_hold.py
