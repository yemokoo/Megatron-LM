#!/bin/bash
# Drain the hyperparameter series through two 4-GPU lanes, in two phases.
#
# Phase 1 trains all six variants, phase 2 scores all six sparse-15.  The
# barrier between them is deliberate: comparing six models is far easier when
# none of them is still moving, and it keeps a late-finishing training from
# queueing behind an earlier model's evaluation.
#
# Within a phase a queue unit is one variant, so a lane that finishes early
# pulls the next unit instead of idling behind a slower one.  The queue is a
# file guarded by flock, which is what makes two workers on it safe.
#
# A failed unit is recorded and the lane moves on -- one broken config must not
# strand the other five.  A variant that failed to train is skipped in phase 2
# by run_v3_job.sh, which refuses to evaluate a missing round-7 checkpoint.
#
#   V3_SERIES_LANES   GPU groups, default "0,1,2,3 4,5,6,7"
#   V3_SERIES_PHASES  which phases to run, default "train eval"
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${DIR}/.." && pwd)"

LANES="${V3_SERIES_LANES:-0,1,2,3 4,5,6,7}"
PHASES="${V3_SERIES_PHASES:-train eval}"
STAGGER="${V3_SERIES_STAGGER:-30}"
RUN_DIR="/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/series_lanes"
QUEUE="${RUN_DIR}/queue.txt"
LOCK="${RUN_DIR}/queue.lock"
FAILURES="${RUN_DIR}/failures.txt"
mkdir -p "${RUN_DIR}"

export SLORA_LLAMA31_PATH="${SLORA_LLAMA31_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}"
export TRACE_DATA_ROOT="${TRACE_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace}"
if ! grep -q chat_template "${SLORA_LLAMA31_PATH}/tokenizer_config.json" 2>/dev/null; then
  echo "[ERROR] ${SLORA_LLAMA31_PATH} has no chat_template; that is the BASE model" >&2
  exit 2
fi

# Order inside a phase: SLoRA first because every v3 number is read against it
# and it is the shortest unit, then the two headline 1:1 runs, then the stored
# memory variations, and recency last -- it is the one change whose value
# depends on what 1:1 turns out to do.  Paths are expanded here because each
# unit runs in a fresh `bash -c` that would not inherit this shell.
UNITS=(
  "slora_pre_upstream"
  "v3_replay1to1:v3_new_replay1to1"
  "v3_hidden_mse_1to1:v3_new_hidden_mse_1to1"
  "v3_replay1to1_p5k:v3_new_replay1to1_p5k"
  "v3_hidden_mse_1to1_p5k:v3_new_hidden_mse_1to1_p5k"
  "v3_replay1to1_recency:v3_new_replay1to1_recency"
)

write_queue() {
  local phase="$1" unit slug version
  : > "${QUEUE}"
  for unit in "${UNITS[@]}"; do
    if [[ "${unit}" == "slora_pre_upstream" ]]; then
      if [[ "${phase}" == "train" ]]; then
        printf '%s|SLORA_SKIP_EVAL=1 SLORA_GPUS=__GPUS__ bash %q\n' \
          "${unit}" "${DIR}/run_slora_pre_upstream_4gpu.sh" >> "${QUEUE}"
      else
        printf '%s|SLORA_SKIP_TRAIN=1 SLORA_GPUS=__GPUS__ bash %q\n' \
          "${unit}" "${DIR}/run_slora_pre_upstream_4gpu.sh" >> "${QUEUE}"
      fi
      continue
    fi
    slug="${unit%%:*}"
    version="${unit#*:}"
    printf '%s|bash %q %s %s __GPUS__ %s\n' \
      "${slug}" "${DIR}/run_v3_job.sh" "${slug}" "${version}" "${phase}" >> "${QUEUE}"
  done
}

pop_unit() {
  local unit=""
  {
    flock 9
    unit="$(head -n 1 "${QUEUE}" 2>/dev/null || true)"
    [[ -n "${unit}" ]] && sed -i '1d' "${QUEUE}"
  } 9>"${LOCK}"
  printf '%s' "${unit}"
}

lane_worker() {
  local gpus="$1" lane_id="$2" phase="$3"
  local log="${RUN_DIR}/${phase}_lane${lane_id}.log"
  local unit name cmd status
  while :; do
    unit="$(pop_unit)"
    [[ -z "${unit}" ]] && break
    name="${unit%%|*}"
    cmd="${unit#*|}"
    cmd="${cmd//__GPUS__/${gpus}}"
    printf '[%s LANE %s gpus=%s] START %s %s\n' \
      "${phase}" "${lane_id}" "${gpus}" "${name}" "$(date '+%F %T')" | tee -a "${log}"
    if [[ "${V3_SERIES_DRYRUN:-0}" == "1" ]]; then
      # Exercise the queue, the lock and the GPU substitution with no GPU work.
      bash -c "sleep $(( (RANDOM % 3) + 1 )); echo \"[dry] ${cmd}\"" >> "${log}" 2>&1
    else
      CUDA_DEVICE_ORDER=PCI_BUS_ID bash -c "${cmd}" >> "${log}" 2>&1
    fi
    status=$?
    printf '[%s LANE %s] %s exit=%s %s\n' \
      "${phase}" "${lane_id}" "${name}" "${status}" "$(date '+%F %T')" | tee -a "${log}"
    [[ "${status}" -ne 0 ]] && printf '%s %s exit=%s\n' "${phase}" "${name}" "${status}" >> "${FAILURES}"
  done
  printf '[%s LANE %s gpus=%s] queue empty %s\n' \
    "${phase}" "${lane_id}" "${gpus}" "$(date '+%F %T')" | tee -a "${log}"
}

: > "${FAILURES}"
for phase in ${PHASES}; do
  write_queue "${phase}"
  echo "[LANES] === phase ${phase} === $(date '+%F %T')"
  sed 's/^/  /' "${QUEUE}"
  lane_id="${V3_SERIES_LANE_ID_BASE:-0}"
  for gpus in ${LANES}; do
    lane_worker "${gpus}" "${lane_id}" "${phase}" &
    lane_id=$(( lane_id + 1 ))
    sleep "${STAGGER}"
  done
  # Barrier: every lane must drain this phase before the next one is queued.
  wait
  echo "[LANES] phase ${phase} finished $(date '+%F %T')"
done

echo "[LANES] all phases finished $(date '+%F %T')"
if [[ -s "${FAILURES}" ]]; then
  echo "[LANES] FAILURES:"; sed 's/^/  /' "${FAILURES}"; exit 1
fi
echo "[LANES] every queued unit exited 0"
