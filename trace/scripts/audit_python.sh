#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${SLORA_AUDIT_TARGET:-${ROOT}/.audit-packages}"
if [[ ! -d "${TARGET_DIR}" ]]; then
  echo "Missing ${TARGET_DIR}; run scripts/setup_audit_env.sh first." >&2
  exit 2
fi
export PYTHONPATH="${TARGET_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN:-python3}" "$@"
