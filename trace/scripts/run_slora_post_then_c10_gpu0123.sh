#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${TRACE_PYTHON:-${ROOT}/.venv-runtime/bin/python}"
RUN_ROOT="${TRACE_RUN_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace}"
TAG="${TRACE_HIDDEN_MSE_TAG:-instruct_hidden_mse_postkd_20260813}"
GPUS="${TRACE_CHAIN_GPUS:-0,1,2,3}"
C10_TAG="${TRACE_HIDDEN_MSE_C10_TAG:-instruct_hidden_mse_postkd_c10_20260814}"
LOG_ROOT="${RUN_ROOT}/logs/${TAG}/post_c10_no_pre_20260815"
POST_LOG_ROOT="${RUN_ROOT}/logs/${TAG}/slora_post_followup"
TOP4_OUTPUT="${RUN_ROOT}/v3/instruct_priority_fourway_20260812/v3_new_top4"
GRPO_QUEUE="${GRPO_QUEUE_SCRIPT:-/home/seonghyeonnoh/yemokoo/androidflux/rl/train/queue_parallel_grpo.sh}"
mkdir -p "${LOG_ROOT}"
status() { echo "$*" | tee -a "${LOG_ROOT}/status.log"; }

: > "${LOG_ROOT}/status.log"
status "[PRE REMOVED] $(date --iso-8601=seconds) SLoRA-Pre train/eval intentionally skipped"

status "[1/4] $(date --iso-8601=seconds) SLoRA-Post train/prepare/eval"
printf '[CHAIN COMPLETE] pre intentionally omitted\n' > "${LOG_ROOT}/upstream.complete"
TRACE_UPSTREAM_UNIT=trace-pre-intentionally-omitted.service \
TRACE_UPSTREAM_STATUS="${LOG_ROOT}/upstream.complete" \
TRACE_CHAIN_GPUS="${GPUS}" TRACE_SLORA_POST_LOG_ROOT="${POST_LOG_ROOT}" \
SLORA_EVAL_PYTHON="${PYTHON}" \
  bash "${ROOT}/scripts/wait_hidden_mse_chain_then_slora_post_gpu0123.sh" \
  2>&1 | tee -a "${LOG_ROOT}/01_slora_post_chain.log"

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
