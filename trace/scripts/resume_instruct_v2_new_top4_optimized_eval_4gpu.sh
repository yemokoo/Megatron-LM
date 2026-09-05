#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${INSTRUCT_TOP4_OUT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/instruct_priority_fourway_20260812/v2_new_top4}"
MODEL_PATH="${INSTRUCT_MODEL_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}"
PYTHON_BIN="${TRACE_PYTHON:-${ROOT}/.venv-runtime/bin/python}"

export OURS_LORAMOE_OUTPUT_ROOT="${RUN_DIR}"
export SLORA_LLAMA31_PATH="${MODEL_PATH}"
export SPARSE15_METHODS="ours_lora_moe_v2_new_top4"
export SPARSE15_GPUS="${SPARSE15_GPUS:-0,1,2,3}"
export SPARSE15_EXACT_STOP_MARKERS="${SPARSE15_EXACT_STOP_MARKERS:-1}"
export SPARSE15_EVAL_BATCH="${SPARSE15_EVAL_BATCH:-32}"
export SPARSE15_SCIENCEQA_BATCH="${SPARSE15_SCIENCEQA_BATCH:-128}"
export SPARSE15_MEETINGBANK_BATCH="${SPARSE15_MEETINGBANK_BATCH:-1}"
export SPARSE15_PY150_BATCH="${SPARSE15_PY150_BATCH:-8}"
export SPARSE15_20MINUTEN_BATCH="${SPARSE15_20MINUTEN_BATCH:-8}"
export SPARSE15_STATUS_INTERVAL="${SPARSE15_STATUS_INTERVAL:-60}"
export SPARSE15_CPU_THREADS="${SPARSE15_CPU_THREADS:-4}"
export TRACE_PYTHON="${PYTHON_BIN}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

echo "[EVAL RESUME START] $(date --iso-8601=seconds) run=${RUN_DIR} gpus=${SPARSE15_GPUS}"
exec "${PYTHON_BIN}" -u "${ROOT}/scripts/run_ours_sparse15_optimized.py"
