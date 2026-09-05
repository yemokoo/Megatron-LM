#!/usr/bin/env bash
set -euo pipefail

TRACE_ROOT="/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace"
GRPO_ROOT="/home/seonghyeonnoh/yemokoo/androidflux/rl/train"
GRPO_RUNTIME="${ANDROIDFLUX_RL_RUNTIME_ROOT:-/data2/seonghyeonnoh/androidflux-rl}"
TRACE_RUN_ROOT="${TRACE_RUN_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace}"
TAG="${TRACE_HIDDEN_MSE_C10_TAG:-instruct_hidden_mse_postkd_c10_20260814}"
CHAIN_LOG_ROOT="${TRACE_RUN_ROOT}/logs/${TAG}/grpo_first_then_trace_20260815"
V3_OUT="${TRACE_RUN_ROOT}/v3/${TAG}/v3_new_hidden_mse_c10"

mkdir -p "${CHAIN_LOG_ROOT}"
status() { printf '%s\n' "$*" | tee -a "${CHAIN_LOG_ROOT}/status.log"; }

: >"${CHAIN_LOG_ROOT}/status.log"
status "[CHAIN START] $(date --iso-8601=seconds) GPUs 0-3"
status "[1/3 GRPO START] scalar=(RM0,policy1), discrete=(RM2,policy3)"

GRPO_QUEUE_SKIP_SOURCE_WAIT=1 GRPO_QUEUE_TARGET_GPUS=0,1,2,3 \
  ANDROIDFLUX_RL_RUNTIME_ROOT="${GRPO_RUNTIME}" \
  bash "${GRPO_ROOT}/queue_parallel_grpo.sh" \
  2>&1 | tee -a "${CHAIN_LOG_ROOT}/01_grpo_parallel.log"

GRPO_RUN_ROOT="$(<"${GRPO_RUNTIME}/queue/latest_run_root")"
for mode in scalar discrete; do
  [[ "$(<"${GRPO_RUN_ROOT}/${mode}/exit_status")" == "0" ]] || {
    status "[ERROR] ${mode} GRPO exit status is not zero"
    exit 2
  }
  adapter="$(find "${GRPO_RUN_ROOT}/${mode}/outputs" -type f -name adapter_model.safetensors -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
  [[ -n "${adapter}" && -s "${adapter}" ]] || {
    status "[ERROR] ${mode} GRPO adapter missing"
    exit 3
  }
  status "[GRPO VERIFIED] ${mode} adapter=${adapter}"
done
status "[1/3 GRPO COMPLETE] $(date --iso-8601=seconds) root=${GRPO_RUN_ROOT}"

if [[ ! -s "${V3_OUT}/7/pytorch_model.bin" && -d "${V3_OUT}" ]] && \
   find "${V3_OUT}" -mindepth 1 -print -quit | grep -q .; then
  archive="${V3_OUT}.interrupted_$(date +%Y%m%d_%H%M%S)"
  mv "${V3_OUT}" "${archive}"
  status "[V3 PARTIAL ARCHIVED] ${archive}"
fi

status "[2/3 TRACE START] V2 c10 will skip as complete; V3 c10 starts fresh"
TRACE_SKIP_FINAL_GRPO=1 TRACE_RUN_ROOT="${TRACE_RUN_ROOT}" \
  TRACE_HIDDEN_MSE_C10_TAG="${TAG}" \
  bash "${TRACE_ROOT}/scripts/run_c10_train_then_parallel_eval_gpu0123.sh" \
  2>&1 | tee -a "${CHAIN_LOG_ROOT}/02_trace_c10_and_eval.log"

status "[3/3 CHAIN COMPLETE] $(date --iso-8601=seconds)"
