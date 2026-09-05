#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLD_SERVICE="${OLD_CHAIN_SERVICE:-trace-instruct-priority-resume-after-top4-20260813.service}"
OLD_EVAL_PID="${OLD_EVAL_PID:-2791500}"
V3_OUTPUT="${INSTRUCT_V3_NEW_OUT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/instruct_priority_fourway_20260812/v3_new}"
SUMMARY="${V3_OUTPUT}/sparse15_summary.json"
POLL_SECONDS="${POLL_SECONDS:-15}"

echo "[HANDOFF START] $(date --iso-8601=seconds) waiting=${SUMMARY}"
polls=0
eval_pid_running() {
  [[ -r "/proc/${OLD_EVAL_PID}/stat" ]] || return 1
  # A stopped parent cannot reap its completed child.  kill -0 still succeeds
  # for that zombie, so inspect proc state and treat Z/X as finished.
  local state
  state="$(awk '{print $3}' "/proc/${OLD_EVAL_PID}/stat" 2>/dev/null || true)"
  [[ "${state}" != "Z" && "${state}" != "X" && -n "${state}" ]]
}

while [[ ! -s "${SUMMARY}" ]] || eval_pid_running; do
  polls=$((polls + 1))
  if (( polls == 1 || polls % 4 == 0 )); then
    echo "[HANDOFF WAIT] $(date --iso-8601=seconds) summary=$([[ -s "${SUMMARY}" ]] && echo ready || echo pending) eval_pid=$(eval_pid_running && echo running || echo finished)"
  fi
  sleep "${POLL_SECONDS}"
done

echo "[HANDOFF RELEASE] $(date --iso-8601=seconds) V3 evaluation complete"
systemctl --user stop "${OLD_SERVICE}"

export INSTRUCT_GPUS="0,1,2,3"
exec bash "${ROOT}/scripts/run_instruct_priority_fourway_fourgpu_continue.sh"
