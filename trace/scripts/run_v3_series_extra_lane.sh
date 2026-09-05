#!/bin/bash
# A second worker on the same flock-guarded queue as run_v3_series_lanes.sh.
#
# The original lane 1 exited when three units failed in a row and drained the
# queue.  Its phase loop cannot be restarted without also restarting lane 0,
# which is mid-way through SLoRA, so this attaches a worker to the same queue
# file instead.  The protocol is identical -- pop under flock, run, record --
# so the two workers cannot hand out the same unit.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/series_lanes"
QUEUE="${RUN_DIR}/queue.txt"; LOCK="${RUN_DIR}/queue.lock"; FAILURES="${RUN_DIR}/failures.txt"
GPUS="${EXTRA_LANE_GPUS:-4,5,6,7}"
LOG="${RUN_DIR}/train_lane_extra.log"

export SLORA_LLAMA31_PATH="${SLORA_LLAMA31_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}"
export TRACE_DATA_ROOT="${TRACE_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace}"

pop_unit() {
  local unit=""
  { flock 9
    unit="$(head -n 1 "${QUEUE}" 2>/dev/null || true)"
    [[ -n "${unit}" ]] && sed -i '1d' "${QUEUE}"
  } 9>"${LOCK}"
  printf '%s' "${unit}"
}

while :; do
  unit="$(pop_unit)"
  [[ -z "${unit}" ]] && break
  name="${unit%%|*}"; cmd="${unit#*|}"; cmd="${cmd//__GPUS__/${GPUS}}"
  printf '[extra gpus=%s] START %s %s\n' "${GPUS}" "${name}" "$(date '+%F %T')" | tee -a "${LOG}"
  CUDA_DEVICE_ORDER=PCI_BUS_ID bash -c "${cmd}" >> "${LOG}" 2>&1
  status=$?
  printf '[extra] %s exit=%s %s\n' "${name}" "${status}" "$(date '+%F %T')" | tee -a "${LOG}"
  [[ "${status}" -ne 0 ]] && printf 'extra %s exit=%s\n' "${name}" "${status}" >> "${FAILURES}"
done
printf '[extra gpus=%s] queue empty %s\n' "${GPUS}" "$(date '+%F %T')" | tee -a "${LOG}"
