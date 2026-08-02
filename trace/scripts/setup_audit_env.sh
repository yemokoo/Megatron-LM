#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${SLORA_AUDIT_ENV:-${ROOT}/.venv-audit}"
TARGET_DIR="${SLORA_AUDIT_TARGET:-${ROOT}/.audit-packages}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -d "${TARGET_DIR}" ]] && PYTHONPATH="${TARGET_DIR}" "${PYTHON_BIN}" -c \
  'import numpy; assert numpy.__version__ == "1.26.4"' 2>/dev/null; then
  echo "Audit environment ready (target mode): ${TARGET_DIR}"
  echo "Run with: ${ROOT}/scripts/audit_python.sh"
  exit 0
fi

if [[ ! -x "${ENV_DIR}/bin/python" ]] || ! "${ENV_DIR}/bin/python" -m pip --version >/dev/null 2>&1; then
  if "${PYTHON_BIN}" -m venv "${ENV_DIR}" 2>/dev/null; then
    :
  else
    echo "ensurepip unavailable; using isolated --target packages at ${TARGET_DIR}"
    "${PYTHON_BIN}" -m pip install --disable-pip-version-check \
      --target "${TARGET_DIR}" --upgrade \
      -r "${ROOT}/config/requirements-audit.lock"
    echo "Audit environment ready (target mode): ${TARGET_DIR}"
    echo "Run with: ${ROOT}/scripts/audit_python.sh"
    exit 0
  fi
fi
"${ENV_DIR}/bin/python" -m pip install --disable-pip-version-check \
  -r "${ROOT}/config/requirements-audit.lock"
echo "Audit environment ready: ${ENV_DIR}"
echo "Activate with: source ${ENV_DIR}/bin/activate"
