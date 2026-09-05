#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${TRACE_PYTHON:-${ROOT}/.venv-runtime/bin/python}"
MODEL="${SLORA_LLAMA31_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}"
DATA_ROOT="${TRACE_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace}"
RUN_ROOT="${TRACE_RUN_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace}"
TAG="${TRACE_HIDDEN_MSE_C10_TAG:-instruct_hidden_mse_postkd_c10_20260814}"
V2_OUT="${RUN_ROOT}/v2/${TAG}/v2_new_hidden_mse_c10"
V3_OUT="${RUN_ROOT}/v3/${TAG}/v3_new_hidden_mse_c10"
LOG_ROOT="${RUN_ROOT}/logs/${TAG}/sparse15_resume_gpu0123_20260816"
STATUS_LOG="${LOG_ROOT}/status.log"

mkdir -p "${LOG_ROOT}"
: >"${STATUS_LOG}"

status() {
  echo "$*" | tee -a "${STATUS_LOG}"
}

complete_model() {
  [[ -s "$1/7/pytorch_model.bin" && -s "$1/7/lora_moe_meta.json" ]]
}

for output in "${V2_OUT}" "${V3_OUT}"; do
  if ! complete_model "${output}"; then
    status "[ERROR] incomplete model: ${output}"
    exit 2
  fi
done

run_eval() {
  local output="$1" method="$2" gpus="$3" log="$4"
  env \
    SLORA_LLAMA31_PATH="${MODEL}" \
    TRACE_DATA_ROOT="${DATA_ROOT}" \
    OURS_LORAMOE_OUTPUT_ROOT="${output}" \
    SPARSE15_METHODS="${method}" \
    SPARSE15_GPUS="${gpus}" \
    SPARSE15_NUM_SAMPLE_SHARDS=2 \
    SPARSE15_EXACT_STOP_MARKERS=1 \
    SPARSE15_EVAL_BATCH=32 \
    SPARSE15_SCIENCEQA_BATCH=128 \
    SPARSE15_MEETINGBANK_BATCH=1 \
    SPARSE15_PY150_BATCH=2 \
    SPARSE15_20MINUTEN_BATCH=2 \
    SPARSE15_STATUS_INTERVAL=60 \
    TRACE_PYTHON="${PYTHON}" \
    PYTHONNOUSERSITE=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    "${PYTHON}" -u "${ROOT}/scripts/run_ours_sparse15_optimized.py" \
      --matrix-mode sparse15 >"${log}" 2>&1
}

status "[START] $(date --iso-8601=seconds) V2=GPU0,1 V3=GPU2,3 shards=2 resume-complete-cells"
run_eval "${V2_OUT}" ours_lora_moe_v2_new_hidden_mse 0,1 "${LOG_ROOT}/v2_eval.log" &
v2_pid=$!
run_eval "${V3_OUT}" ours_lora_moe_v3_new_hidden_mse 2,3 "${LOG_ROOT}/v3_eval.log" &
v3_pid=$!
status "[PIDS] parent=$$ v2=${v2_pid} v3=${v3_pid}"

failed=0
wait "${v2_pid}" || {
  status "[ERROR] $(date --iso-8601=seconds) V2 evaluation failed; log=${LOG_ROOT}/v2_eval.log"
  failed=1
}
wait "${v3_pid}" || {
  status "[ERROR] $(date --iso-8601=seconds) V3 evaluation failed; log=${LOG_ROOT}/v3_eval.log"
  failed=1
}
if [[ "${failed}" != 0 ]]; then
  exit 3
fi

for summary in "${V2_OUT}/sparse15_summary.json" "${V3_OUT}/sparse15_summary.json"; do
  [[ -s "${summary}" ]] || {
    status "[ERROR] missing final summary: ${summary}"
    exit 4
  }
done
status "[COMPLETE] $(date --iso-8601=seconds) V2/V3 c10 sparse15 evaluation complete"
