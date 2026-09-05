#!/bin/bash
# Run a command once the whole v3 series -- training, evaluation and the
# repair pass that fills OOM-skipped cells -- has finished and released its
# GPUs.
#
#   chain_after_v3_series.sh <command...>
#
# Three conditions have to agree, because any one of them alone lies:
#   - the orchestrator pid is gone: it owns the phase-eval repair pass, so it
#     exits last
#   - the extra lane pid is gone: it drains the same queue but is not a child
#     of the orchestrator
#   - no cell is still missing: a repair pass can end with gaps if a cell OOMs
#     again, and starting new GPU work on top of that hides the problem
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace
LANES_PID="${LANES_PID:-22547}"
EXTRA_PID="${EXTRA_PID:-484897}"
POLL="${POLL_SECONDS:-60}"
PER_GPU_FREE_MB="${PER_GPU_FREE_MB:-2000}"
stamp() { date '+%F %T'; }

[[ "$#" -ge 1 ]] || { echo "usage: chain_after_v3_series.sh <command...>" >&2; exit 2; }

echo "[CHAIN] waiting for the series orchestrator ${LANES_PID} $(stamp)"
while kill -0 "${LANES_PID}" 2>/dev/null; do sleep "${POLL}"; done
echo "[CHAIN] orchestrator exited $(stamp)"

while kill -0 "${EXTRA_PID}" 2>/dev/null; do
  echo "[CHAIN] extra lane ${EXTRA_PID} still draining $(stamp)"
  sleep "${POLL}"
done
echo "[CHAIN] extra lane exited $(stamp)"

# Count with wc: pgrep prints 0 and exits 1 when nothing matches.
while :; do
  alive="$(pgrep -f 'run_ours_sparse15_optimized|evaluate_Ours_LoRA_MoE|main_Ours_LoRA_MoE|cl_train_slora' 2>/dev/null | wc -l)"
  [[ "${alive}" -eq 0 ]] && break
  echo "[CHAIN] ${alive} worker process(es) still alive $(stamp)"
  sleep "${POLL}"
done

while :; do
  busy=""
  for gpu in 0 1 2 3 4 5 6 7; do
    used="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)"
    [[ "${used}" -gt "${PER_GPU_FREE_MB}" ]] && busy="${busy} ${gpu}(${used}MB)"
  done
  [[ -z "${busy}" ]] && break
  echo "[CHAIN] GPUs still holding memory:${busy} $(stamp)"
  sleep 30
done
echo "[CHAIN] all eight GPUs free $(stamp)"

echo "[CHAIN] final cell audit:"
bash "${DIR}/check_eval_oom.sh" || echo "[CHAIN] cells are missing; the chained command runs anyway, but see above"

echo "[CHAIN] starting: $* $(stamp)"
exec "$@"
