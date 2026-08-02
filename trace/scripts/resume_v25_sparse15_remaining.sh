#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${TRACE_PYTHON:-${ROOT}/.venv-runtime/bin/python}"
RUNNER="${ROOT}/scripts/run_ours_py150_4way.py"
LOG="${ROOT}/results/full_runs/llama31/ours_lora_moe_v2_5/eval_workers/sparse15_remaining_batch8.log"

run_cell() {
  local round="$1"
  local task="$2"
  echo "[RESUME START] $(date -u +'%Y-%m-%d %H:%M:%S UTC') order=${round} task=${task} shards=4 batch_per_shard=8"
  "${PYTHON}" -u "${RUNNER}" \
    --method ours_lora_moe_v2_5 \
    --round "${round}" \
    --task "${task}" \
    --batch 8
  echo "[RESUME DONE] $(date -u +'%Y-%m-%d %H:%M:%S UTC') order=${round} task=${task}"
}

{
  run_cell 8 MeetingBank
  run_cell 4 Py150
  run_cell 8 Py150
  echo "[RESUME ALL COMPLETE] $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
} >>"${LOG}" 2>&1
