#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace"
RUN_DIR="/data2/seonghyeonnoh/LLM-continual-learning-runs/v2_new_retrain_20260807"
LOG_DIR="${RUN_DIR}/eval_queue_logs"
CHAIN_LOG="${RUN_DIR}/eval_sparse15_then_gpu_hold.log"
SUMMARY="${RUN_DIR}/sparse15_summary.json"

mkdir -p "${LOG_DIR}"
exec >>"${CHAIN_LOG}" 2>&1

echo "[CHAIN START] $(date '+%F %T %Z') V2-new sparse-15 evaluation"
cd "${ROOT}"
source "${ROOT}/.venv-runtime/bin/activate"
export PYTHONNOUSERSITE=1
export WANDB_MODE=offline
export TRACE_DATA_ROOT="${ROOT}/data/trace"
export OURS_LORAMOE_OUTPUT_ROOT="${RUN_DIR}"
export SPARSE15_METHODS="ours_lora_moe_v2_new"
export SPARSE15_GPUS="0,1,2,3,4,5,6,7"
export SPARSE15_LOG_ROOT="${LOG_DIR}"
export SPARSE15_EVAL_BATCH=32
export SPARSE15_SCIENCEQA_BATCH=128
export SPARSE15_20MINUTEN_BATCH=32
export SPARSE15_MEETINGBANK_BATCH=1
export SPARSE15_PY150_BATCH=8
export SPARSE15_CPU_THREADS=4

python "${ROOT}/scripts/run_ours_sparse15_efficient.py"

if [[ ! -s "${SUMMARY}" ]]; then
  echo "[CHAIN ERROR] sparse-15 evaluator returned without ${SUMMARY}" >&2
  exit 1
fi

echo "[EVAL COMPLETE] $(date '+%F %T %Z') all 15 evaluations verified"
echo "[TASK-EXPERT START] $(date '+%F %T %Z') final checkpoint task-only routing"
python "${ROOT}/scripts/run_v2_new_task_expert_eval.py"
if [[ ! -s "${RUN_DIR}/task_expert_final_summary.json" ]]; then
  echo "[CHAIN ERROR] task-expert evaluator returned without summary" >&2
  exit 1
fi
echo "[TASK-EXPERT COMPLETE] $(date '+%F %T %Z') all 8 forced-routing evaluations verified"
echo "[GPU HOLD START] $(date '+%F %T %Z')"
exec python /home/seonghyeonnoh/yemokoo/gpu_hold.py
