#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

ENV_NAME="${ENV_NAME:-flame-moe-a100}"
ENV_FILE="${ENV_FILE:-environment.a100.yml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found. Activate your conda installation first."
    exit 1
fi

conda env create -n "$ENV_NAME" -f "$ENV_FILE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

$PYTHON_BIN -m pip install -r Megatron-LM/requirements/pytorch_24.10/requirements.txt
$PYTHON_BIN -m pip install transformers pybind11 tensorboard numpy==1.26.4 wandb

pushd apex >/dev/null
$PYTHON_BIN -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
  --config-settings "--build-option=--cpp_ext" \
  --config-settings "--build-option=--cuda_ext" \
  ./
popd >/dev/null

pushd TransformerEngine >/dev/null
export NVTE_FRAMEWORK=pytorch
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
$PYTHON_BIN -m pip install .
popd >/dev/null

echo "Environment '$ENV_NAME' is ready."
echo "Run a quick smoke test with:"
echo "  conda activate $ENV_NAME && python -c 'import torch; print(torch.cuda.get_device_name(0))'"
