#!/usr/bin/env bash
set -euo pipefail

# Prevent host user packages from shadowing the project virtualenv.
unset PYTHONPATH

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${SLORA_VENV:-${ROOT}/.venv-runtime}"
BOOTSTRAP_VIRTUALENV="${ROOT}/.bootstrap-tools/bin/virtualenv"
PYTHON="${VENV}/bin/python"
PIP="${VENV}/bin/pip"
WHEEL_CACHE="${ROOT}/.wheel-cache"
TORCH_WHEEL="${WHEEL_CACHE}/torch-2.4.1+cu124-cp310-cp310-linux_x86_64.whl"
TORCH_WHEEL_PART="${TORCH_WHEEL}.part"
TORCH_WHEEL_URL="https://download.pytorch.org/whl/cu124/torch-2.4.1%2Bcu124-cp310-cp310-linux_x86_64.whl"
retry() {
  until "$@"; do
    echo "[RETRY] package download failed; retrying in ${PIP_RETRY_DELAY:-5}s" >&2
    sleep "${PIP_RETRY_DELAY:-5}"
  done
}


if [[ ! -x "${PYTHON}" ]]; then
  if [[ -x "${BOOTSTRAP_VIRTUALENV}" ]]; then
    "${BOOTSTRAP_VIRTUALENV}" --python python3 "${VENV}"
  else
    python3 -m venv "${VENV}"
  fi
fi

retry env PIP_CONFIG_FILE=/dev/null "${PIP}" install --timeout 60 --upgrade "pip==24.3.1" "setuptools==75.6.0" "wheel==0.45.1"
mkdir -p "${WHEEL_CACHE}"
if ! "${PYTHON}" -c 'import torch; assert torch.__version__ == "2.4.1+cu124"' 2>/dev/null; then
  if [[ ! -s "${TORCH_WHEEL}" ]]; then
    retry curl --fail --location --continue-at - --connect-timeout 20 --output "${TORCH_WHEEL_PART}" "${TORCH_WHEEL_URL}"
    mv "${TORCH_WHEEL_PART}" "${TORCH_WHEEL}"
  fi
  retry env PIP_CONFIG_FILE=/dev/null "${PIP}" install --timeout 60 "${TORCH_WHEEL}"
fi
retry env PIP_CONFIG_FILE=/dev/null "${PIP}" install --timeout 60 -r "${ROOT}/config/requirements-runtime.txt"

"${PYTHON}" -m pip freeze \
  | sed -E "s#^torch @ file:.*#torch==2.4.1+cu124#" \
  > "${ROOT}/config/requirements-runtime.lock"
"${PYTHON}" - <<'PY'
import importlib

packages = [
    "torch",
    "transformers",
    "peft",
    "trl",
    "datasets",
    "accelerate",
    "safetensors",
    "deepspeed",
    "evaluate",
    "rouge",
    "qpth",
    "quadprog",
    "sacrebleu",
    "sacremoses",
]
for package in packages:
    module = importlib.import_module(package)
    print(package, getattr(module, "__version__", "import-ok"))
PY
