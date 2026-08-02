#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ID="${TRACE_DATASET_REPO:-YeMoKoo/flamedata2}"
LOCAL_DIR="${TRACE_HF_LOCAL_DIR:-${ROOT}/data}"

if [[ -n "${HF_BIN:-}" ]]; then
  HF="${HF_BIN}"
elif [[ -x "${ROOT}/.venv-runtime/bin/hf" ]]; then
  HF="${ROOT}/.venv-runtime/bin/hf"
else
  HF="$(command -v hf || true)"
fi

[[ -n "${HF}" ]] || {
  echo "[ERROR] hf CLI is missing. Install the runtime or set HF_BIN." >&2
  exit 2
}

mkdir -p "${LOCAL_DIR}"
"${HF}" download "${REPO_ID}" \
  --repo-type dataset \
  --include "trace/**" \
  --local-dir "${LOCAL_DIR}"

DATA_ROOT="${LOCAL_DIR}/trace"
for task in C-STANCE FOMC MeetingBank Py150 ScienceQA NumGLUE-cm NumGLUE-ds 20Minuten; do
  for split in train eval test; do
    [[ -f "${DATA_ROOT}/${task}/${split}.json" ]] || {
      echo "[ERROR] missing downloaded file: ${DATA_ROOT}/${task}/${split}.json" >&2
      exit 3
    }
  done
done

echo "[OK] TRACE dataset: ${DATA_ROOT}"
echo "export TRACE_DATA_ROOT=${DATA_ROOT}"
