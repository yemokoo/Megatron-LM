#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WAIT_PID="${1:?usage: $0 <slora-parent-pid> <proc-start-ticks> <process-group-id>}"
WAIT_START_TICKS="${2:?usage: $0 <slora-parent-pid> <proc-start-ticks> <process-group-id>}"
WAIT_PGID="${3:?usage: $0 <slora-parent-pid> <proc-start-ticks> <process-group-id>}"
POLL_SECONDS="${SLORA_TRIGGER_POLL_SECONDS:-60}"
SOURCE_RUN_DIR="${SLORA_TRIGGER_SOURCE_RUN_DIR:-${ROOT}/results/full_runs_upstream_code/llama31/pre}"
SOURCE_LOG="${SLORA_TRIGGER_SOURCE_LOG:-${ROOT}/logs/slora_pre_upstream_order2-8.log}"
CHAIN_SCRIPT="${SLORA_TRIGGER_CHAIN_SCRIPT:-${ROOT}/scripts/chains/run_llama31_seq_ours_v1_v2.sh}"
TRIGGER_ROOT="${SLORA_TRIGGER_STATE_ROOT:-${ROOT}/results/chains/llama31_seq_ours_v1_v2/trigger}"
TRIGGER_ID="${SLORA_TRIGGER_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${TRIGGER_ROOT}/runs/${TRIGGER_ID}"
STATUS_FILE="${RUN_DIR}/status.log"
LOCK_FILE="${TRIGGER_ROOT}/trigger.lock"

case "${WAIT_PID}:${WAIT_START_TICKS}:${WAIT_PGID}:${POLL_SECONDS}" in
  *[!0-9:]*)
    echo "[ERROR] PID, start ticks, process group, and poll seconds must be integers" >&2
    exit 2
    ;;
esac

mkdir -p "${TRIGGER_ROOT}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[ERROR] another SLoRA trigger holds ${LOCK_FILE}" >&2
  exit 9
fi

mkdir -p "${RUN_DIR}"
: > "${STATUS_FILE}"
printf '%s\n' "${RUN_DIR}" > "${TRIGGER_ROOT}/latest_trigger.txt"
printf '%s\n' "$$" > "${RUN_DIR}/watcher.pid"
printf '%s\n' "${WAIT_PID}" > "${RUN_DIR}/source.pid"
printf '%s\n' "${WAIT_START_TICKS}" > "${RUN_DIR}/source_start_ticks"
printf '%s\n' "${WAIT_PGID}" > "${RUN_DIR}/source.pgid"

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

record() {
  local state="$1"
  local detail="${2:-}"
  printf '%s | %s | %s\n' "$(now)" "${state}" "${detail}" | tee -a "${STATUS_FILE}"
}

process_start_ticks() {
  local pid="$1"
  [[ -r "/proc/${pid}/stat" ]] || return 1
  awk '{print $22}' "/proc/${pid}/stat"
}

source_process_is_same() {
  local current
  current="$(process_start_ticks "${WAIT_PID}")" || return 1
  [[ "${current}" == "${WAIT_START_TICKS}" ]]
}

source_group_has_members() {
  ps -eo pgid= | awk -v expected="${WAIT_PGID}" '
    $1 == expected { found = 1 }
    END { exit(found ? 0 : 1) }
  '
}

source_outputs_complete() {
  local order task_dir
  for order in {1..8}; do
    task_dir="${SOURCE_RUN_DIR}/order${order}"
    [[ -s "${task_dir}/adapter_config.json" ]] || return 1
    [[ -s "${task_dir}/adapter_model.safetensors" ]] || return 1
  done
}

list_missing_outputs() {
  local order task_dir
  for order in {1..8}; do
    task_dir="${SOURCE_RUN_DIR}/order${order}"
    [[ -s "${task_dir}/adapter_config.json" ]] ||
      printf 'missing=%s\n' "${task_dir}/adapter_config.json"
    [[ -s "${task_dir}/adapter_model.safetensors" ]] ||
      printf 'missing=%s\n' "${task_dir}/adapter_model.safetensors"
  done
}

if ! source_process_is_same; then
  record "arm_rejected" \
    "source_pid_identity_mismatch expected_pid=${WAIT_PID} expected_start_ticks=${WAIT_START_TICKS}"
  touch "${RUN_DIR}/ARM_REJECTED"
  exit 23
fi

if ! source_group_has_members; then
  record "arm_rejected" "source_process_group_missing expected_pgid=${WAIT_PGID}"
  touch "${RUN_DIR}/ARM_REJECTED"
  exit 23
fi

record "armed" \
  "watcher_pid=$$ source_pid=${WAIT_PID} source_start_ticks=${WAIT_START_TICKS} source_pgid=${WAIT_PGID} poll_seconds=${POLL_SECONDS}"

while source_process_is_same; do
  record "waiting_for_slora" "source_pid=${WAIT_PID} source_log=${SOURCE_LOG}"
  sleep "${POLL_SECONDS}"
done

while source_group_has_members; do
  record "waiting_for_source_process_group" "source_pgid=${WAIT_PGID}"
  sleep "${POLL_SECONDS}"
done

record "source_process_finished" "source_pid=${WAIT_PID}"

if ! source_outputs_complete; then
  list_missing_outputs | tee -a "${STATUS_FILE}"
  record "source_incomplete" "chain_not_started"
  touch "${RUN_DIR}/SOURCE_INCOMPLETE"
  exit 21
fi

touch "${RUN_DIR}/SOURCE_COMPLETE"
record "source_complete" "orders=8/8"

record "chain_validation_started" "script=${CHAIN_SCRIPT}"
if ! "${CHAIN_SCRIPT}" validate 2>&1 | tee -a "${RUN_DIR}/chain_validate.log"; then
  record "chain_validation_failed" "chain_not_started"
  touch "${RUN_DIR}/CHAIN_VALIDATE_FAILED"
  exit 22
fi
record "chain_validation_completed" "exit_code=0"

CHAIN_RUN_ID="after_slora_${TRIGGER_ID}"
touch "${RUN_DIR}/CHAIN_STARTED"
record "chain_started" "chain_run_id=${CHAIN_RUN_ID}"

set +e
env CHAIN_RUN_ID="${CHAIN_RUN_ID}" "${CHAIN_SCRIPT}" train \
  2>&1 | tee -a "${RUN_DIR}/chain_train.log"
chain_rc=${PIPESTATUS[0]}
set -e

printf '%s\n' "${chain_rc}" > "${RUN_DIR}/chain.exit_code"
if [[ "${chain_rc}" -ne 0 ]]; then
  record "chain_failed" "exit_code=${chain_rc}"
  touch "${RUN_DIR}/CHAIN_FAILED"
  exit "${chain_rc}"
fi

touch "${RUN_DIR}/CHAIN_SUCCEEDED"
record "chain_succeeded" "exit_code=0"
