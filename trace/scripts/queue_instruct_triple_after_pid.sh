#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAIT_PID="${WAIT_PID:?set WAIT_PID to the root process that currently owns the GPUs}"
POLL_SECONDS="${POLL_SECONDS:-30}"
CHAIN_SCRIPT="${CHAIN_SCRIPT:-${ROOT}/scripts/run_instruct_priority_fourway_train_eval_chain.sh}"

[[ -f "${CHAIN_SCRIPT}" ]] || {
  echo "[ERROR] missing queued chain: ${CHAIN_SCRIPT}" >&2
  exit 2
}

echo "[QUEUE START] $(date --iso-8601=seconds) waiting_for_pid=${WAIT_PID} chain=${CHAIN_SCRIPT}"
polls=0
while kill -0 "${WAIT_PID}" 2>/dev/null; do
  polls=$((polls + 1))
  if (( polls == 1 || polls % 10 == 0 )); then
    echo "[QUEUE WAIT] $(date --iso-8601=seconds) pid=${WAIT_PID} still_running"
  fi
  sleep "${POLL_SECONDS}"
done

# Avoid racing with a child process that outlived the parent. Require two
# consecutive observations with no GPU compute process before claiming all
# eight devices.
empty_observations=0
while (( empty_observations < 2 )); do
  gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    2>/dev/null | sed '/^[[:space:]]*$/d' | sort -u || true)"
  if [[ -z "${gpu_pids}" ]]; then
    empty_observations=$((empty_observations + 1))
  else
    empty_observations=0
    echo "[QUEUE WAIT] $(date --iso-8601=seconds) gpu_pids=${gpu_pids//$'\n'/,}"
  fi
  sleep "${POLL_SECONDS}"
done

echo "[QUEUE RELEASE] $(date --iso-8601=seconds) GPUs are idle"
exec bash "${CHAIN_SCRIPT}"
