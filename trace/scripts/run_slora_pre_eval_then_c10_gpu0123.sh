#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${TRACE_PYTHON:-${ROOT}/.venv-runtime/bin/python}"
MODEL="${INSTRUCT_MODEL_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}"
RUN_ROOT="${TRACE_RUN_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace}"
TAG="${TRACE_HIDDEN_MSE_TAG:-instruct_hidden_mse_postkd_20260813}"
GPUS="${TRACE_CHAIN_GPUS:-0,1,2,3}"
C10_TAG="${TRACE_HIDDEN_MSE_C10_TAG:-instruct_hidden_mse_postkd_c10_20260814}"
SLORA_ROOT="${RUN_ROOT}/slora/${TAG}"
LOG_ROOT="${RUN_ROOT}/logs/${TAG}/pre_eval_c10_no_post_20260815"
TOP4_OUTPUT="${RUN_ROOT}/v3/instruct_priority_fourway_20260812/v3_new_top4"
GRPO_QUEUE="${GRPO_QUEUE_SCRIPT:-/home/seonghyeonnoh/yemokoo/androidflux/rl/train/queue_parallel_grpo.sh}"
mkdir -p "${LOG_ROOT}"
status() { echo "$*" | tee -a "${LOG_ROOT}/status.log"; }

eval_slora_pre() {
  local pids=() index gpu failed=0
  IFS=',' read -r -a gpu_list <<< "${GPUS}"
  status "[1/4] $(date --iso-8601=seconds) SLoRA-Pre sparse15 evaluation (training already complete)"
  for index in 0 1 2 3; do
    gpu="${gpu_list[$index]}"
    CUDA_VISIBLE_DEVICES="${gpu}" SLORA_OUTPUT_ROOT="${SLORA_ROOT}" \
    SLORA_LLAMA31_PATH="${MODEL}" SLORA_EVAL_PYTHON="${PYTHON}" \
    EVAL_ALL_ROUNDS=1 EVAL_SPARSE_15=1 EVAL_SHARD_COUNT=4 \
    EVAL_SHARD_INDEX="${index}" SLORA_EVAL_BATCH=4 PYTHONNOUSERSITE=1 \
      bash "${ROOT}/implementations/SLoRA-repro/scripts/repro/eval_trace.sh" \
        pre llama31 >"${LOG_ROOT}/01_slora_pre_eval_shard${index}.log" 2>&1 &
    pids+=("$!")
  done
  for index in 0 1 2 3; do
    wait "${pids[$index]}" || { status "[ERROR] SLoRA-Pre shard ${index} failed"; failed=1; }
  done
  [[ "${failed}" == 0 ]] || exit 4
  "${PYTHON}" "${ROOT}/scripts/collect_results.py" --method slora_pre \
    --model llama31 --sparse-15 --family slora \
    --run-dir "${SLORA_ROOT}/llama31/pre" \
    --output "${SLORA_ROOT}/llama31/pre/sparse15_summary.json" \
    2>&1 | tee -a "${LOG_ROOT}/01_slora_pre_collect.log"
}

: > "${LOG_ROOT}/status.log"
status "[POST REMOVED] $(date --iso-8601=seconds) SLoRA-Post train/eval intentionally omitted"
eval_slora_pre

status "[2/4] $(date --iso-8601=seconds) V2/V3 hidden-MSE c=10 train+eval"
TRACE_CHAIN_GPUS="${GPUS}" TRACE_HIDDEN_MSE_C10_TAG="${C10_TAG}" \
  bash "${ROOT}/scripts/run_trace_hidden_mse_c10_gpu0123_chain.sh" \
  2>&1 | tee -a "${LOG_ROOT}/02_c10_chain.log"

status "[3/4] $(date --iso-8601=seconds) V3-new-top4 lower triangle"
bash "${ROOT}/scripts/run_ours_lower_triangle_optimized.sh" \
  "${TOP4_OUTPUT}" ours_lora_moe_v3_new_top4 "${GPUS}" \
  "${LOG_ROOT}/03_v3_new_top4_lower_triangle.log"

status "[4/4] $(date --iso-8601=seconds) AndroidFlux GRPO"
GRPO_QUEUE_SKIP_SOURCE_WAIT=1 GRPO_QUEUE_TARGET_GPUS="${GPUS}" \
  bash "${GRPO_QUEUE}" 2>&1 | tee -a "${LOG_ROOT}/04_androidflux_grpo.log"
status "[CHAIN COMPLETE] $(date --iso-8601=seconds)"
