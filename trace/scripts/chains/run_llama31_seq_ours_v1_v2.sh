#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTION="${1:-validate}"
CHAIN_NAME="llama31_seq_ours_v1_v2"
FULL_RUN_ROOT="${CHAIN_FULL_RUN_ROOT:-${ROOT}/results/full_runs}"
CHAIN_ROOT="${CHAIN_STATE_ROOT:-${ROOT}/results/chains/${CHAIN_NAME}}"
RUN_ID="${CHAIN_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${CHAIN_LOG_DIR:-${CHAIN_ROOT}/runs/${RUN_ID}}"
STATUS_FILE="${LOG_DIR}/status.log"
LOCK_FILE="${CHAIN_ROOT}/chain.lock"
SEQ_RUN_DIR="${FULL_RUN_ROOT}/llama31/seq"
V1_RUN_DIR="${FULL_RUN_ROOT}/llama31/ours_lora_moe_v1"
V2_RUN_DIR="${FULL_RUN_ROOT}/llama31/ours_lora_moe_v2"
SEQ_WRAPPER="${CHAIN_SEQ_WRAPPER:-${ROOT}/scripts/baselines/llama31/seq_lora.sh}"
V1_WRAPPER="${CHAIN_V1_WRAPPER:-${ROOT}/scripts/baselines/llama31/ours_lora_moe_v1.sh}"
V2_WRAPPER="${CHAIN_V2_WRAPPER:-${ROOT}/scripts/baselines/llama31/ours_lora_moe_v2.sh}"
TOKEN_CACHE="${ROOT}/cache/tokenized/llama31_8b/slora_chat_full_len1024"
REPLAY_MANIFEST="${ROOT}/manifests/replay/trace_seed2025_random50_per_task.json"

case "${ACTION}" in
  validate|status|plan|train) ;;
  *) echo "usage: $0 <validate|status|plan|train>" >&2; exit 2 ;;
esac

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
record() {
  local stage="$1"
  local state="$2"
  local detail="${3:-}"
  local line="$(now) | ${stage} | ${state} | ${detail}"
  echo "${line}" | tee -a "${STATUS_FILE}"
}

seq_completed_orders() {
  local order count=0
  for order in {1..8}; do
    if [[ -s "${SEQ_RUN_DIR}/order${order}/adapter_config.json" && -s "${SEQ_RUN_DIR}/order${order}/adapter_model.safetensors" ]]; then
      count=$((count + 1))
    fi
  done
  echo "${count}"
}

seq_complete() { [[ "$(seq_completed_orders)" == "8" ]]; }

ours_completed_rounds() {
  local run_dir="$1"
  local round count=0
  for round in {0..7}; do
    if [[ -s "${run_dir}/${round}/lora_moe_meta.json" && -s "${run_dir}/${round}/pytorch_model.bin" ]]; then
      count=$((count + 1))
    fi
  done
  echo "${count}"
}

ours_complete() { [[ "$(ours_completed_rounds "$1")" == "8" ]]; }

latest_ours_round() {
  local run_dir="$1"
  local round latest=-1
  for ((round=0; round<=7; round++)); do
    if [[ -s "${run_dir}/${round}/lora_moe_meta.json" && -s "${run_dir}/${round}/pytorch_model.bin" ]]; then
      latest="${round}"
    else
      break
    fi
  done
  echo "${latest}"
}

show_status() {
  echo "chain=${CHAIN_NAME}"
  echo "seq_lora=$(seq_completed_orders)/8 dir=${SEQ_RUN_DIR}"
  echo "ours_v1=$(ours_completed_rounds "${V1_RUN_DIR}")/8 dir=${V1_RUN_DIR}"
  echo "ours_v2=$(ours_completed_rounds "${V2_RUN_DIR}")/8 dir=${V2_RUN_DIR}"
  echo "chain_logs=${CHAIN_ROOT}"
}

preflight_files() {
  local path task
  for path in "${SEQ_WRAPPER}" "${V1_WRAPPER}" "${V2_WRAPPER}"; do
    [[ -x "${path}" ]] || { echo "[ERROR] executable missing: ${path}" >&2; return 1; }
  done
  [[ -f "${TOKEN_CACHE}/manifest.json" ]] || { echo "[ERROR] token cache manifest missing" >&2; return 1; }
  [[ -f "${REPLAY_MANIFEST}" ]] || { echo "[ERROR] replay manifest missing" >&2; return 1; }
  for task in C-STANCE FOMC MeetingBank Py150 ScienceQA NumGLUE-cm NumGLUE-ds 20Minuten; do
    [[ -s "${TOKEN_CACHE}/${task}.parquet" ]] || { echo "[ERROR] token cache missing: ${task}" >&2; return 1; }
  done
}

run_stage() {
  local stage="$1"
  shift
  local log_file="${LOG_DIR}/${stage}.log"
  local rc
  record "${stage}" "running" "log=${log_file}"
  set +e
  "$@" 2>&1 | tee -a "${log_file}"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    record "${stage}" "failed" "exit_code=${rc}"
    return "${rc}"
  fi
  record "${stage}" "command_finished" "exit_code=0"
}

run_ours_stage() {
  local version="$1"
  local run_dir="$2"
  local wrapper="$3"
  local stage="ours_${version}"
  local latest
  local command
  if ours_complete "${run_dir}"; then
    record "${stage}" "skipped" "already_complete=8/8"
    return 0
  fi
  latest="$(latest_ours_round "${run_dir}")"
  command=(env "OURS_LORAMOE_OUTPUT_ROOT=${run_dir}")
  if [[ "${latest}" -ge 0 ]]; then
    command+=("OURS_LORAMOE_RESUME_CHECKPOINT=${run_dir}/${latest}")
    record "${stage}" "resume_selected" "round=${latest}"
  else
    command+=("OURS_LORAMOE_RESUME_CHECKPOINT=")
    record "${stage}" "fresh_start" "round=-1"
  fi
  command+=("${wrapper}" train)
  run_stage "${stage}" "${command[@]}" || return $?
  if ! ours_complete "${run_dir}"; then
    record "${stage}" "failed_completion_check" "completed=$(ours_completed_rounds "${run_dir}")/8"
    return 20
  fi
  record "${stage}" "completed" "rounds=8/8"
}

if [[ "${ACTION}" == "status" ]]; then
  show_status
  exit 0
fi

preflight_files

if [[ "${ACTION}" == "plan" ]]; then
  if seq_complete; then
    echo "seq_lora=skip completed=8/8"
  else
    echo "seq_lora=run completed=$(seq_completed_orders)/8 skip_completed=1"
  fi
  if ours_complete "${V1_RUN_DIR}"; then
    echo "ours_v1=skip completed=8/8"
  elif [[ "$(latest_ours_round "${V1_RUN_DIR}")" -ge 0 ]]; then
    echo "ours_v1=resume round=$(latest_ours_round "${V1_RUN_DIR}")"
  else
    echo "ours_v1=fresh"
  fi
  if ours_complete "${V2_RUN_DIR}"; then
    echo "ours_v2=skip completed=8/8"
  elif [[ "$(latest_ours_round "${V2_RUN_DIR}")" -ge 0 ]]; then
    echo "ours_v2=resume round=$(latest_ours_round "${V2_RUN_DIR}")"
  else
    echo "ours_v2=fresh"
  fi
  exit 0
fi
mkdir -p "${CHAIN_ROOT}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[ERROR] another chain process holds ${LOCK_FILE}" >&2
  exit 9
fi
mkdir -p "${LOG_DIR}"
: > "${STATUS_FILE}"
printf '%s\n' "${LOG_DIR}" > "${CHAIN_ROOT}/latest_run.txt"

if [[ "${ACTION}" == "validate" ]]; then
  bash -n "${ROOT}/implementations/SLoRA-repro/scripts/repro/train_trace.sh"
  bash -n "${ROOT}/scripts/baselines/_run_ours_lora_moe.sh"
  run_stage "seq_lora_validate" "${SEQ_WRAPPER}" validate
  run_stage "ours_v1_validate" "${V1_WRAPPER}" validate
  run_stage "ours_v2_validate" "${V2_WRAPPER}" validate
  record "chain" "validated" "no_training_started"
  show_status
  exit 0
fi

printf '%s\n' "$$" > "${LOG_DIR}/chain.pid"
record "chain" "running" "pid=$$ order=seq_lora,ours_v1,ours_v2"

if seq_complete; then
  record "seq_lora" "skipped" "already_complete=8/8"
else
  run_stage "seq_lora" env "SLORA_OUTPUT_ROOT=${FULL_RUN_ROOT}" "SLORA_SKIP_COMPLETED=1" "${SEQ_WRAPPER}" train || exit $?
  if ! seq_complete; then
    record "seq_lora" "failed_completion_check" "completed=$(seq_completed_orders)/8"
    exit 20
  fi
  record "seq_lora" "completed" "orders=8/8"
fi

run_ours_stage "v1" "${V1_RUN_DIR}" "${V1_WRAPPER}" || exit $?
run_ours_stage "v2" "${V2_RUN_DIR}" "${V2_WRAPPER}" || exit $?

record "chain" "completed" "seq_lora=8/8 ours_v1=8/8 ours_v2=8/8"
printf '%s\n' "$(now)" > "${LOG_DIR}/CHAIN_COMPLETE"
show_status
