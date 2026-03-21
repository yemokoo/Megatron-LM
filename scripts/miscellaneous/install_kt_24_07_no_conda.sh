#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

if command -v python >/dev/null 2>&1; then
  SYSTEM_PYTHON="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  SYSTEM_PYTHON="$(command -v python3)"
else
  echo "ERROR: python/python3 not found in this KT 24.07 session"
  exit 1
fi

VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv-kt2407}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"
USE_VENV="${USE_VENV:-1}"

"$SYSTEM_PYTHON" --version

if [ "$USE_VENV" = "1" ]; then
  "$SYSTEM_PYTHON" -m venv "$VENV_DIR"
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  PYTHON_BIN="$(command -v python)"
else
  PYTHON_BIN="$SYSTEM_PYTHON"
fi

"$PYTHON_BIN" --version
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

# Reinstall a known-good PyTorch stack close to the previously validated local env.
"$PYTHON_BIN" -m pip install --upgrade --index-url "$PYTORCH_INDEX_URL" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}"

# Megatron requirements, excluding the unpinned torch entry above.
FILTERED_REQS="$(mktemp)"
"$PYTHON_BIN" - <<'PY' > "$FILTERED_REQS"
from pathlib import Path
req_path = Path('Megatron-LM/requirements/pytorch_24.10/requirements.txt')
for line in req_path.read_text().splitlines():
    if line.strip() == 'torch':
        continue
    print(line)
PY
"$PYTHON_BIN" -m pip install -r "$FILTERED_REQS"
rm -f "$FILTERED_REQS"

"$PYTHON_BIN" -m pip install transformers pybind11 tensorboard numpy==1.26.4 wandb

pushd apex >/dev/null
"$PYTHON_BIN" -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
  --config-settings "--build-option=--cpp_ext" \
  --config-settings "--build-option=--cuda_ext" \
  ./
popd >/dev/null

pushd TransformerEngine >/dev/null
export NVTE_FRAMEWORK=pytorch
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
"$PYTHON_BIN" -m pip install .
popd >/dev/null

echo "KT 24.07 environment setup completed."
if [ "$USE_VENV" = "1" ]; then
  echo "Activate with: source $VENV_DIR/bin/activate"
fi

echo "Recommended smoke tests:"
echo "  python -c 'import torch; print(torch.__version__, torch.version.cuda)'"
echo "  python -c 'import transformer_engine, apex; print(\"ok\")'"
echo "  python Megatron-LM/pretrain_gpt.py --help >/tmp/pretrain_help.txt && tail -n 5 /tmp/pretrain_help.txt"
