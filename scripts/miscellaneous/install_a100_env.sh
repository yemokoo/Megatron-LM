#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${ENV_NAME:-flame-megatron-a100}"
ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/environment.a100.yml}"
REQ_FILE="${REQ_FILE:-${PROJECT_ROOT}/environments/flame-megatron/requirements.txt}"
MAX_JOBS="${MAX_JOBS:-8}"
SKIP_EXTENSIONS=0

usage() {
  cat <<'EOF'
Usage: scripts/miscellaneous/install_a100_env.sh [--skip-extensions]

Creates/updates an isolated Conda environment for FLAME/Megatron. The default
build installs torch 2.4.1+cu124 and compiles grouped-gemm, Apex,
TransformerEngine, and flash-attn against that same torch/CUDA ABI.

Environment overrides: ENV_NAME, ENV_FILE, REQ_FILE, MAX_JOBS, CUDA_HOME.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-extensions) SKIP_EXTENSIONS=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install Miniforge/Miniconda first." >&2
  exit 1
fi
if [[ "$SKIP_EXTENSIONS" -eq 0 ]]; then
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not found. Install a CUDA 12.4/12.5 toolkit before building extensions." >&2
    exit 1
  fi
  if ! command -v gcc >/dev/null 2>&1; then
    echo "ERROR: gcc not found. GCC/G++ 11 is recommended." >&2
    exit 1
  fi
  CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
  export CUDA_HOME
fi

export MAX_JOBS
export PYTHONNOUSERSITE=1
unset PYTHONPATH

source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda env update --name "$ENV_NAME" --file "$ENV_FILE"
else
  conda env create --name "$ENV_NAME" --file "$ENV_FILE"
fi
conda activate "$ENV_NAME"

python -m pip install --disable-pip-version-check \
  --index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.4.1+cu124" "torchvision==0.19.1+cu124" "torchaudio==2.4.1+cu124"
python -m pip install --disable-pip-version-check -r "$REQ_FILE"

if [[ "$SKIP_EXTENSIONS" -eq 0 ]]; then
  # G2 FFN+attention expert runners enable both grouped-GEMM paths.
  python -m pip install --disable-pip-version-check --no-build-isolation \
    "git+https://github.com/fanshiqing/grouped_gemm@172fada89fa7364fe5d026b3a0dfab58b591ffdd"

  python -m pip install -v --disable-pip-version-check --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    "$PROJECT_ROOT/apex"

  NVTE_FRAMEWORK=pytorch \
    python -m pip install -v --disable-pip-version-check --no-build-isolation --no-deps \
    "$PROJECT_ROOT/TransformerEngine"

  # Optional in the current local-attention G2 path, but required by other
  # committed Megatron attention entrypoints.
  FLASH_ATTENTION_FORCE_BUILD=TRUE \
    python -m pip install -v --disable-pip-version-check --no-build-isolation \
    "flash-attn==2.4.2"
fi

echo
echo "Environment '$ENV_NAME' is ready."
echo "Activate and validate it with:"
echo "  conda activate $ENV_NAME"
echo "  source $PROJECT_ROOT/scripts/miscellaneous/activate_flame_env.sh"
echo "  python $PROJECT_ROOT/scripts/miscellaneous/verify_flame_env.py --require-gpu"
