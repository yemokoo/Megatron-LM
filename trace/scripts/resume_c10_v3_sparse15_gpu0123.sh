#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${TRACE_PYTHON:-${ROOT}/.venv-runtime/bin/python}"
MODEL="${SLORA_LLAMA31_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}"
DATA_ROOT="${TRACE_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace}"
RUN_ROOT="${TRACE_RUN_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace}"
TAG="${TRACE_HIDDEN_MSE_C10_TAG:-instruct_hidden_mse_postkd_c10_20260814}"
V3_OUT="${RUN_ROOT}/v3/${TAG}/v3_new_hidden_mse_c10"
LOG_ROOT="${RUN_ROOT}/logs/${TAG}/v3_sparse15_resume_4gpu_20260816"

mkdir -p "${LOG_ROOT}"
[[ -s "${V3_OUT}/7/pytorch_model.bin" && -s "${V3_OUT}/7/lora_moe_meta.json" ]] || {
  echo "[ERROR] incomplete V3 model: ${V3_OUT}" >&2
  exit 2
}

echo "[START] $(date --iso-8601=seconds) V3 c10 sparse15 GPUs=0,1,2,3 shards=2" \
  | tee "${LOG_ROOT}/status.log"

env \
  SLORA_LLAMA31_PATH="${MODEL}" \
  TRACE_DATA_ROOT="${DATA_ROOT}" \
  OURS_LORAMOE_OUTPUT_ROOT="${V3_OUT}" \
  SPARSE15_METHODS=ours_lora_moe_v3_new_hidden_mse \
  SPARSE15_GPUS=0,1,2,3 \
  SPARSE15_NUM_SAMPLE_SHARDS=2 \
  SPARSE15_EXACT_STOP_MARKERS=1 \
  SPARSE15_EVAL_BATCH=32 \
  SPARSE15_SCIENCEQA_BATCH=128 \
  SPARSE15_MEETINGBANK_BATCH=1 \
  SPARSE15_PY150_BATCH=2 \
  SPARSE15_20MINUTEN_BATCH=2 \
  SPARSE15_STATUS_INTERVAL=60 \
  SPARSE15_CONTINUE_ON_CELL_ERROR=1 \
  TRACE_PYTHON="${PYTHON}" \
  PYTHONNOUSERSITE=1 \
  TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 \
  OPENBLAS_NUM_THREADS=8 \
  "${PYTHON}" -u "${ROOT}/scripts/run_ours_sparse15_optimized.py" \
    --matrix-mode sparse15 >"${LOG_ROOT}/v3_eval.log" 2>&1

if [[ -s "${V3_OUT}/sparse15_summary.json" ]]; then
  echo "[COMPLETE] $(date --iso-8601=seconds) V3 c10 sparse15" \
    | tee -a "${LOG_ROOT}/status.log"
elif [[ -s "${V3_OUT}/sparse15_partial_summary.json" && \
        -s "${V3_OUT}/evaluation/sparse15_failures.json" ]]; then
  echo "[PARTIAL COMPLETE] $(date --iso-8601=seconds) V3 c10 sparse15 failed cells skipped" \
    | tee -a "${LOG_ROOT}/status.log"
else
  echo "[ERROR] final or partial sparse15 summary missing" | tee -a "${LOG_ROOT}/status.log"
  exit 3
fi
