#!/usr/bin/env bash
set -euo pipefail

unset PYTHONPATH
unset BNB_CUDA_VERSION

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${SLORA_AUDIT_PYTHON:-}" ]]; then
  PYTHON_BIN="${SLORA_AUDIT_PYTHON}"
elif [[ -x "${ROOT}/.venv-runtime/bin/python" ]]; then
  PYTHON_BIN="${ROOT}/.venv-runtime/bin/python"
else
  PYTHON_BIN="python3"
fi

"${PYTHON_BIN}" "${ROOT}/scripts/preflight.py" --mode scaffold
TRAIN_JSON="${TRACE_DATA_ROOT:-${ROOT}/data/trace}/C-STANCE/train.json"
if [[ -f "${TRAIN_JSON}" ]]; then
  mkdir -p "${ROOT}/results"
  "${PYTHON_BIN}" "${ROOT}/scripts/run_contract.py" \
    --train-json "${TRAIN_JSON}" \
    --task C-STANCE --expected-samples 5000 --micro-batch 2 --world-size 1 \
    --gradient-accumulation 8 --epochs 5 --logging-steps 1 \
    > "${ROOT}/results/smoke_run_contract.json"
else
  echo "[SKIP] TRACE data is not installed; run scripts/data/download_trace_from_hf.sh"
fi
"${PYTHON_BIN}" -m unittest discover -s "${ROOT}/tests" -v
"${PYTHON_BIN}" "${ROOT}/scripts/compare_results.py" \
  "${ROOT}/tests/fixtures/smoke_result.json" \
  --output "${ROOT}/results/smoke_summary.json"
echo "Offline scaffold smoke test: PASS"
