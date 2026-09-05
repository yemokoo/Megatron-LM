#!/bin/bash
# Drain the three single-axis ablations through one 4-GPU lane, train+eval each.
#
# Each unit changes exactly one thing against v3_replay1to1, and run_v3_job.sh
# gates that with assert_matches_v3_new_baseline.py before it takes a GPU, so a
# config mistake fails in seconds instead of six hours in.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/ablation_lanes
QUEUE="${RUN_DIR}/queue.txt"; LOCK="${RUN_DIR}/queue.lock"; FAILURES="${RUN_DIR}/failures.txt"
GPUS="${ABLATION_GPUS:-4,5,6,7}"
LOG="${RUN_DIR}/lane.log"
export SLORA_LLAMA31_PATH="${SLORA_LLAMA31_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}"
export TRACE_DATA_ROOT="${TRACE_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace}"

pop() {
  local u=""
  { flock 9
    u="$(head -n 1 "${QUEUE}" 2>/dev/null || true)"
    [[ -n "${u}" ]] && sed -i '1d' "${QUEUE}"
  } 9>"${LOCK}"
  printf '%s' "${u}"
}

while :; do
  unit="$(pop)"; [[ -z "${unit}" ]] && break
  name="${unit%%|*}"; cmd="${unit#*|}"; cmd="${cmd//__GPUS__/${GPUS}}"
  printf '[ABL gpus=%s] START %s %s\n' "${GPUS}" "${name}" "$(date '+%F %T')" | tee -a "${LOG}"
  CUDA_DEVICE_ORDER=PCI_BUS_ID bash -c "${cmd}" >> "${LOG}" 2>&1
  status=$?
  printf '[ABL] %s exit=%s %s\n' "${name}" "${status}" "$(date '+%F %T')" | tee -a "${LOG}"
  [[ "${status}" -ne 0 ]] && printf '%s exit=%s\n' "${name}" "${status}" >> "${FAILURES}"
done
printf '[ABL gpus=%s] queue empty %s\n' "${GPUS}" "$(date '+%F %T')" | tee -a "${LOG}"
