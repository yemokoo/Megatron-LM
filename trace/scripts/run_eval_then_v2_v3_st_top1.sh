#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_RUN="${OURS_EVAL_RUN:-/data3/seonghyeonnoh/LLM-continual-learning-runs/local/trace/results/full_runs/llama31/ours_lora_moe_v3_fixed1000_epochprobe64}"

export OURS_LORAMOE_OUTPUT_ROOT="${EVAL_RUN}"
export SPARSE15_METHODS=ours_lora_moe_v3
export SPARSE15_GPUS=0,1,2,3,4,5,6,7
export SPARSE15_EVAL_BATCH=32
export SPARSE15_SCIENCEQA_BATCH=192
export SPARSE15_20MINUTEN_BATCH=32
export SPARSE15_MEETINGBANK_BATCH=2
export SPARSE15_PY150_BATCH=16
export SPARSE15_CPU_THREADS=4
export SPARSE15_LOG_ROOT="${EVAL_RUN}/eval_queue_logs"
export TRACE_PYTHON="${ROOT}/.venv-runtime/bin/python"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

echo "[CHAIN] resuming sparse-15 evaluation"
"${TRACE_PYTHON}" -u "${ROOT}/scripts/run_ours_sparse15_efficient.py"
[[ -f "${EVAL_RUN}/sparse15_summary.json" ]] || {
  echo "[ERROR] sparse-15 summary missing after evaluator success" >&2
  exit 3
}
echo "[CHAIN] sparse-15 complete; starting corrected V2/V3 training"
bash "${ROOT}/scripts/run_v2_v3_st_top1_4plus4.sh"
