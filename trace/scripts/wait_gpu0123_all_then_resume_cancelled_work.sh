#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${TRACE_RUN_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace}"
TAG="${TRACE_HIDDEN_MSE_TAG:-instruct_hidden_mse_postkd_20260813}"
GPUS="${TRACE_RESUMED_GPUS:-0,1,2,3}"
UPSTREAM_UNIT="${TRACE_UPSTREAM_UNIT:-trace-slora-post-followup-gpu0123-20260813.service}"
UPSTREAM_STATUS="${TRACE_UPSTREAM_STATUS:-${RUN_ROOT}/logs/${TAG}/slora_post_followup/status.log}"
TOP4_OUTPUT="${TRACE_V3_TOP4_OUTPUT:-${RUN_ROOT}/v3/instruct_priority_fourway_20260812/v3_new_top4}"
LOG_ROOT="${TRACE_RESUMED_LOG_ROOT:-${RUN_ROOT}/logs/${TAG}/resumed_cancelled_after_gpu0123}"
STATUS_LOG="${LOG_ROOT}/status.log"
GRPO_QUEUE="${GRPO_QUEUE_SCRIPT:-/home/seonghyeonnoh/yemokoo/androidflux/rl/train/queue_parallel_grpo.sh}"

mkdir -p "${LOG_ROOT}"
status() { echo "$*" | tee -a "${STATUS_LOG}"; }

status "[WAIT START] $(date --iso-8601=seconds) upstream=${UPSTREAM_UNIT} gpus=${GPUS}"
while systemctl --user is-active --quiet "${UPSTREAM_UNIT}"; do
  status "[WAIT] $(date --iso-8601=seconds) GPU 0-3 chain still active"
  sleep 60
done

if ! grep -q '^\[SLoRA-POST CHAIN COMPLETE\]' "${UPSTREAM_STATUS}"; then
  status "[ERROR] upstream stopped without SLoRA-POST CHAIN COMPLETE: ${UPSTREAM_STATUS}"
  exit 3
fi
if [[ ! -s "${TOP4_OUTPUT}/7/pytorch_model.bin" || \
      ! -s "${TOP4_OUTPUT}/7/lora_moe_meta.json" ]]; then
  status "[ERROR] V3-new-top4 final checkpoint is incomplete: ${TOP4_OUTPUT}/7"
  exit 4
fi

status "[1/2] $(date --iso-8601=seconds) resume V3-new-top4 full lower triangle on GPUs ${GPUS}"
bash "${ROOT}/scripts/run_ours_lower_triangle_optimized.sh" \
  "${TOP4_OUTPUT}" ours_lora_moe_v3_new_top4 "${GPUS}" \
  "${LOG_ROOT}/01_v3_new_top4_lower_triangle.log"

status "[2/2] $(date --iso-8601=seconds) launch queued AndroidFlux GRPO on GPUs ${GPUS}"
GRPO_QUEUE_SKIP_SOURCE_WAIT=1 \
GRPO_QUEUE_TARGET_GPUS="${GPUS}" \
  bash "${GRPO_QUEUE}" 2>&1 | tee -a "${LOG_ROOT}/02_androidflux_grpo.log"

status "[RESUMED CANCELLED WORK COMPLETE] $(date --iso-8601=seconds)"
