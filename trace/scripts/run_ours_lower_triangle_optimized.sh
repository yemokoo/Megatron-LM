#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:?usage: run_ours_lower_triangle_optimized.sh RUN_DIR METHOD GPUS [LOG]}"
METHOD="${2:?usage: run_ours_lower_triangle_optimized.sh RUN_DIR METHOD GPUS [LOG]}"
GPUS="${3:?usage: run_ours_lower_triangle_optimized.sh RUN_DIR METHOD GPUS [LOG]}"
LOG="${4:-${RUN_DIR}/evaluation/lower_triangle_queue.log}"
PYTHON="${TRACE_PYTHON:-${ROOT}/.venv-runtime/bin/python}"
MODEL="${SLORA_LLAMA31_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}"

mkdir -p "$(dirname "${LOG}")"
echo "[LOWER START] $(date --iso-8601=seconds) run=${RUN_DIR} method=${METHOD} gpus=${GPUS}" | tee -a "${LOG}"

SLORA_LLAMA31_PATH="${MODEL}" \
OURS_LORAMOE_OUTPUT_ROOT="${RUN_DIR}" \
SPARSE15_METHODS="${METHOD}" \
SPARSE15_GPUS="${GPUS}" \
SPARSE15_EXACT_STOP_MARKERS=1 \
SPARSE15_EVAL_BATCH="${SPARSE15_EVAL_BATCH:-32}" \
SPARSE15_SCIENCEQA_BATCH="${SPARSE15_SCIENCEQA_BATCH:-128}" \
SPARSE15_MEETINGBANK_BATCH="${SPARSE15_MEETINGBANK_BATCH:-1}" \
SPARSE15_PY150_BATCH="${SPARSE15_PY150_BATCH:-8}" \
SPARSE15_20MINUTEN_BATCH="${SPARSE15_20MINUTEN_BATCH:-8}" \
TRACE_PYTHON="${PYTHON}" \
PYTHONNOUSERSITE=1 \
TOKENIZERS_PARALLELISM=false \
  "${PYTHON}" -u "${ROOT}/scripts/run_ours_sparse15_optimized.py" \
    --matrix-mode lower_triangle 2>&1 | tee -a "${LOG}"

if [[ -s "${RUN_DIR}/lower_triangle_summary.json" ]]; then
  echo "[LOWER COMPLETE] $(date --iso-8601=seconds) summary=${RUN_DIR}/lower_triangle_summary.json" | tee -a "${LOG}"
else
  echo "[LOWER PARTIAL] $(date --iso-8601=seconds) summary=${RUN_DIR}/lower_triangle_partial_summary.json failures=${RUN_DIR}/evaluation/lower_triangle_failures.json" | tee -a "${LOG}"
fi
