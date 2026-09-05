#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${TRACE_PYTHON:-${ROOT}/.venv-runtime/bin/python}"
MODEL="${SLORA_LLAMA31_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}"
DATA_ROOT="${TRACE_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace}"
RUN_ROOT="${TRACE_RUN_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace}"
UPSTREAM_UNIT="${TRACE_V3_C10_EVAL_UNIT:-trace-c10-v3-sparse15-resume-4gpu-20260816.service}"
C10_TAG="${TRACE_HIDDEN_MSE_C10_TAG:-instruct_hidden_mse_postkd_c10_20260814}"
V3_C10="${RUN_ROOT}/v3/${C10_TAG}/v3_new_hidden_mse_c10"
V3_C10_STATUS="${RUN_ROOT}/logs/${C10_TAG}/v3_sparse15_resume_4gpu_20260816/status.log"
TOP4_OUTPUT="${RUN_ROOT}/v3/instruct_priority_fourway_20260812/v3_new_top4"
SLORA_TAG="${TRACE_HIDDEN_MSE_TAG:-instruct_hidden_mse_postkd_20260813}"
SLORA_ROOT="${RUN_ROOT}/slora/${SLORA_TAG}"
SLORA_RUN="${SLORA_ROOT}/llama31/pre"
LOG_ROOT="${RUN_ROOT}/logs/${C10_TAG}/after_v3_c10_top4_lower_then_slora_pre_20260816"
STATUS_LOG="${LOG_ROOT}/status.log"

mkdir -p "${LOG_ROOT}"
touch "${STATUS_LOG}"
status() { echo "$*" | tee -a "${STATUS_LOG}"; }

wait_gpu0123_free() {
  local attempt gpu busy
  for attempt in $(seq 1 180); do
    busy=0
    for gpu in 0 1 2 3; do
      if nvidia-smi -i "${gpu}" --query-compute-apps=pid \
          --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; then
        busy=1
      fi
    done
    [[ "${busy}" == 0 ]] && return 0
    sleep 2
  done
  status "[ERROR] GPU 0-3 did not become free within 360 seconds"
  return 1
}

wait_top4_gpus_free() {
  local attempt gpu busy
  for attempt in $(seq 1 180); do
    busy=0
    for gpu in 0 1 2 3 4 5; do
      if nvidia-smi -i "${gpu}" --query-compute-apps=pid \
          --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; then
        busy=1
      fi
    done
    [[ "${busy}" == 0 ]] && return 0
    sleep 2
  done
  status "[ERROR] top4 GPUs 0,1,2,3,4,5 did not become free within 360 seconds"
  return 1
}

wait_gpu0_5_free() {
  local attempt gpu busy
  for attempt in $(seq 1 180); do
    busy=0
    for gpu in 0 1 2 3 4 5; do
      if nvidia-smi -i "${gpu}" --query-compute-apps=pid \
          --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; then
        busy=1
      fi
    done
    [[ "${busy}" == 0 ]] && return 0
    sleep 2
  done
  status "[ERROR] GPU 0-5 did not become free within 360 seconds"
  return 1
}

status "[WAIT] $(date --iso-8601=seconds) upstream=${UPSTREAM_UNIT}"
while systemctl --user is-active --quiet "${UPSTREAM_UNIT}"; do
  sleep 20
done
if ! grep -Eq '^\[(COMPLETE|PARTIAL COMPLETE)\].*V3 c10 sparse15' "${V3_C10_STATUS}"; then
  status "[RECOVER] current V3 evaluator stopped before a summary; resuming with per-cell skip policy"
  wait_gpu0123_free
  bash "${ROOT}/scripts/resume_c10_v3_sparse15_gpu0123.sh" \
    >>"${LOG_ROOT}/00_v3_c10_recovery.log" 2>&1 || {
      status "[ERROR] V3 recovery failed outside an evaluation cell; log=${LOG_ROOT}/00_v3_c10_recovery.log"
      exit 2
    }
fi
if [[ -s "${V3_C10}/sparse15_summary.json" ]]; then
  status "[UPSTREAM COMPLETE] V3 c10 full summary"
elif [[ -s "${V3_C10}/sparse15_partial_summary.json" && \
        -s "${V3_C10}/evaluation/sparse15_failures.json" ]]; then
  status "[UPSTREAM PARTIAL] V3 c10 failed cells skipped"
else
  status "[ERROR] V3 c10 full/partial summary missing"
  exit 2
fi
wait_top4_gpus_free

status "[1/2 START] $(date --iso-8601=seconds) v3_new_top4 lower triangle GPUs=0,1,2,3,4,5 shards=4"
SPARSE15_EVAL_BATCH=32 \
SPARSE15_SCIENCEQA_BATCH=128 \
SPARSE15_MEETINGBANK_BATCH=1 \
SPARSE15_PY150_BATCH=2 \
SPARSE15_20MINUTEN_BATCH=2 \
SPARSE15_NUM_SAMPLE_SHARDS=4 \
SPARSE15_CONTINUE_ON_CELL_ERROR=1 \
TRACE_DATA_ROOT="${DATA_ROOT}" \
TRACE_PYTHON="${PYTHON}" \
SLORA_LLAMA31_PATH="${MODEL}" \
  bash "${ROOT}/scripts/run_ours_lower_triangle_optimized.sh" \
    "${TOP4_OUTPUT}" ours_lora_moe_v3_new_top4 0,1,2,3,4,5 \
    "${LOG_ROOT}/01_v3_new_top4_lower_triangle.log"
if [[ -s "${TOP4_OUTPUT}/lower_triangle_summary.json" ]]; then
  status "[1/2 COMPLETE] $(date --iso-8601=seconds) top4 lower triangle full"
elif [[ -s "${TOP4_OUTPUT}/lower_triangle_partial_summary.json" && \
        -s "${TOP4_OUTPUT}/evaluation/lower_triangle_failures.json" ]]; then
  status "[1/2 PARTIAL] $(date --iso-8601=seconds) top4 failed cells were skipped; summary=${TOP4_OUTPUT}/lower_triangle_partial_summary.json"
else
  status "[ERROR] top4 evaluator failed outside a task cell; continuing to SLoRA"
fi
wait_gpu0_5_free

for round in $(seq 1 8); do
  [[ -s "${SLORA_RUN}/order${round}/adapter_model.safetensors" && \
     -s "${SLORA_RUN}/order${round}/adapter_config.json" ]] || {
    status "[ERROR] trained SLoRA adapter missing: ${SLORA_RUN}/order${round}"
    exit 4
  }
done

status "[2/2 START] $(date --iso-8601=seconds) trained SLoRA-Pre sparse15 GPUs=0,1,2,3,4,5"
pids=()
failed=0
for index in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES="${index}" \
  SLORA_OUTPUT_ROOT="${SLORA_ROOT}" \
  SLORA_LLAMA31_PATH="${MODEL}" \
  SLORA_EVAL_PYTHON="${PYTHON}" \
  TRACE_DATA_ROOT="${DATA_ROOT}" \
  EVAL_ALL_ROUNDS=1 \
  EVAL_SPARSE_15=1 \
  EVAL_SHARD_COUNT=6 \
  EVAL_SHARD_INDEX="${index}" \
  EVAL_CONTINUE_ON_CELL_ERROR=1 \
  SLORA_EVAL_BATCH=4 \
  PYTHONNOUSERSITE=1 \
  TOKENIZERS_PARALLELISM=false \
    bash "${ROOT}/implementations/SLoRA-repro/scripts/repro/eval_trace.sh" \
      pre llama31 >"${LOG_ROOT}/02_slora_pre_eval_shard${index}.log" 2>&1 &
  pids+=("$!")
done
for index in 0 1 2 3 4 5; do
  wait "${pids[$index]}" || {
    status "[ERROR] SLoRA-Pre shard process ${index} failed outside/after a task cell; other shards continue; log=${LOG_ROOT}/02_slora_pre_eval_shard${index}.log"
    failed=1
  }
done

"${PYTHON}" "${ROOT}/scripts/collect_results.py" \
  --method slora_pre --model llama31 --sparse-15 --family slora \
  --allow-partial \
  --run-dir "${SLORA_RUN}" \
  --output "${SLORA_RUN}/sparse15_partial_or_full_summary.json" \
  2>&1 | tee "${LOG_ROOT}/02_slora_pre_collect.log"
[[ -s "${SLORA_RUN}/sparse15_partial_or_full_summary.json" ]] || {
  status "[ERROR] SLoRA-Pre summary generation failed"
  exit 6
}
if "${PYTHON}" - "${SLORA_RUN}/sparse15_partial_or_full_summary.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["complete"] else 1)
PY
then
  cp -f "${SLORA_RUN}/sparse15_partial_or_full_summary.json" "${SLORA_RUN}/sparse15_summary.json"
  status "[2/2 COMPLETE] $(date --iso-8601=seconds) SLoRA-Pre sparse15 full"
else
  status "[2/2 PARTIAL] $(date --iso-8601=seconds) SLoRA-Pre failed cells skipped; summary=${SLORA_RUN}/sparse15_partial_or_full_summary.json shard_process_failure=${failed}"
fi
status "[CHAIN COMPLETE] $(date --iso-8601=seconds)"
