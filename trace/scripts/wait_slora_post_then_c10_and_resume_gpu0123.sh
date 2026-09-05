#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${TRACE_RUN_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace}"
GPUS="${TRACE_RESUMED_GPUS:-0,1,2,3}"
UPSTREAM_UNIT="${TRACE_UPSTREAM_UNIT:-trace-slora-post-followup-gpu0123-20260813.service}"
OLD_TAG="${TRACE_HIDDEN_MSE_TAG:-instruct_hidden_mse_postkd_20260813}"
UPSTREAM_STATUS="${RUN_ROOT}/logs/${OLD_TAG}/slora_post_followup/status.log"
C10_TAG="${TRACE_HIDDEN_MSE_C10_TAG:-instruct_hidden_mse_postkd_c10_20260814}"
C10_STATUS="${RUN_ROOT}/logs/${C10_TAG}/status.log"
TOP4_OUTPUT="${RUN_ROOT}/v3/instruct_priority_fourway_20260812/v3_new_top4"
LOG_ROOT="${RUN_ROOT}/logs/${C10_TAG}/post_c10_resume"
GRPO_QUEUE="${GRPO_QUEUE_SCRIPT:-/home/seonghyeonnoh/yemokoo/androidflux/rl/train/queue_parallel_grpo.sh}"
mkdir -p "${LOG_ROOT}"
status() { echo "$*" | tee -a "${LOG_ROOT}/status.log"; }

status "[WAIT START] $(date --iso-8601=seconds) upstream=${UPSTREAM_UNIT}"
while systemctl --user is-active --quiet "${UPSTREAM_UNIT}"; do
  status "[WAIT] $(date --iso-8601=seconds) SLoRA-Post chain active"
  sleep 60
done
grep -q '^\[SLoRA-POST CHAIN COMPLETE\]' "${UPSTREAM_STATUS}" || {
  status "[ERROR] upstream incomplete: ${UPSTREAM_STATUS}"; exit 3;
}

status "[1/3] $(date --iso-8601=seconds) V2/V3 hidden-MSE c=10 train+eval"
TRACE_CHAIN_GPUS="${GPUS}" TRACE_HIDDEN_MSE_C10_TAG="${C10_TAG}" \
  bash "${ROOT}/scripts/run_trace_hidden_mse_c10_gpu0123_chain.sh" \
  2>&1 | tee -a "${LOG_ROOT}/01_c10_chain.log"
grep -q '^\[C10 CHAIN COMPLETE\]' "${C10_STATUS}" || {
  status "[ERROR] c10 chain incomplete: ${C10_STATUS}"; exit 4;
}

status "[2/3] $(date --iso-8601=seconds) V3-new-top4 lower triangle"
bash "${ROOT}/scripts/run_ours_lower_triangle_optimized.sh" \
  "${TOP4_OUTPUT}" ours_lora_moe_v3_new_top4 "${GPUS}" \
  "${LOG_ROOT}/02_v3_new_top4_lower_triangle.log"

status "[3/3] $(date --iso-8601=seconds) AndroidFlux GRPO"
GRPO_QUEUE_SKIP_SOURCE_WAIT=1 GRPO_QUEUE_TARGET_GPUS="${GPUS}" \
  bash "${GRPO_QUEUE}" 2>&1 | tee -a "${LOG_ROOT}/03_androidflux_grpo.log"
status "[ALL COMPLETE] $(date --iso-8601=seconds)"
