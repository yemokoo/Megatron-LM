#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${TRACE_PYTHON:-${ROOT}/.venv-runtime/bin/python}"
MODEL="${INSTRUCT_MODEL_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}"
RUN_ROOT="${TRACE_RUN_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace}"
TAG="${TRACE_HIDDEN_MSE_TAG:-instruct_hidden_mse_postkd_20260813}"
GPUS="${TRACE_CHAIN_GPUS:-0,1,2,3}"
SLORA_ROOT="${RUN_ROOT}/slora/${TAG}"
LOG_ROOT="${RUN_ROOT}/logs/${TAG}/recovery_20260815"
POST_LOG_ROOT="${RUN_ROOT}/logs/${TAG}/slora_post_followup"
C10_TAG="${TRACE_HIDDEN_MSE_C10_TAG:-instruct_hidden_mse_postkd_c10_20260814}"
TOP4_OUTPUT="${RUN_ROOT}/v3/instruct_priority_fourway_20260812/v3_new_top4"
GRPO_QUEUE="${GRPO_QUEUE_SCRIPT:-/home/seonghyeonnoh/yemokoo/androidflux/rl/train/queue_parallel_grpo.sh}"
mkdir -p "${LOG_ROOT}"
status() { echo "$*" | tee -a "${LOG_ROOT}/status.log"; }

eval_slora_pre() {
  local pids=() index gpu failed=0
  IFS=',' read -r -a gpu_list <<< "${GPUS}"
  status "[1/6] $(date --iso-8601=seconds) resume SLoRA-Pre sparse15 evaluation"
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
eval_slora_pre

status "[2/6] $(date --iso-8601=seconds) SLoRA-Post train/prepare/eval"
printf '[CHAIN COMPLETE] recovery prerequisite\n' > "${LOG_ROOT}/pre_recovered.complete"
TRACE_UPSTREAM_UNIT=trace-recovery-already-complete.service \
TRACE_UPSTREAM_STATUS="${LOG_ROOT}/pre_recovered.complete" \
TRACE_CHAIN_GPUS="${GPUS}" TRACE_SLORA_POST_LOG_ROOT="${POST_LOG_ROOT}" \
SLORA_EVAL_PYTHON="${PYTHON}" \
  bash "${ROOT}/scripts/wait_hidden_mse_chain_then_slora_post_gpu0123.sh" \
  2>&1 | tee -a "${LOG_ROOT}/02_slora_post_chain.log"

status "[3/6] $(date --iso-8601=seconds) V2/V3 hidden-MSE c=10 train+eval"
TRACE_CHAIN_GPUS="${GPUS}" TRACE_HIDDEN_MSE_C10_TAG="${C10_TAG}" \
  bash "${ROOT}/scripts/run_trace_hidden_mse_c10_gpu0123_chain.sh" \
  2>&1 | tee -a "${LOG_ROOT}/03_c10_chain.log"

status "[4/6] $(date --iso-8601=seconds) V3-new-top4 lower triangle"
bash "${ROOT}/scripts/run_ours_lower_triangle_optimized.sh" \
  "${TOP4_OUTPUT}" ours_lora_moe_v3_new_top4 "${GPUS}" \
  "${LOG_ROOT}/04_v3_new_top4_lower_triangle.log"

status "[5/6] $(date --iso-8601=seconds) AndroidFlux GRPO"
GRPO_QUEUE_SKIP_SOURCE_WAIT=1 GRPO_QUEUE_TARGET_GPUS="${GPUS}" \
  bash "${GRPO_QUEUE}" 2>&1 | tee -a "${LOG_ROOT}/05_androidflux_grpo.log"
status "[6/6 COMPLETE] $(date --iso-8601=seconds)"
