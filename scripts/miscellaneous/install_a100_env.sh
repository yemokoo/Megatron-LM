#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET_GPU="${TARGET_GPU:-a100}"
ENV_NAME="${ENV_NAME:-flame-megatron-${TARGET_GPU}}"
ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/environment.a100.yml}"
REQ_FILE="${REQ_FILE:-${PROJECT_ROOT}/environments/flame-megatron/requirements.txt}"
MAX_JOBS="${MAX_JOBS:-8}"
CONDA_CUDA_TOOLKIT_VERSION="${CONDA_CUDA_TOOLKIT_VERSION:-12.4.1}"
SKIP_EXTENSIONS=0

usage() {
  cat <<'EOF'
Usage: scripts/miscellaneous/install_a100_env.sh [--skip-extensions]

Creates/updates an isolated Conda environment for FLAME/Megatron. The default
build installs torch 2.4.1+cu124 and compiles grouped-gemm, Apex,
TransformerEngine, and flash-attn against that same torch/CUDA ABI.

Environment overrides: TARGET_GPU (a100 or h100), ENV_NAME, ENV_FILE, REQ_FILE,
MAX_JOBS, CUDA_HOME, CONDA_CUDA_TOOLKIT_VERSION.
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

case "${TARGET_GPU,,}" in
  a100)
    EXPECTED_GPU_PATTERN="A100"
    CUDA_ARCH_LIST="8.0"
    ;;
  h100)
    EXPECTED_GPU_PATTERN="H100"
    CUDA_ARCH_LIST="9.0"
    ;;
  *)
    echo "ERROR: TARGET_GPU must be 'a100' or 'h100', got '$TARGET_GPU'." >&2
    exit 2
    ;;
esac

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install Miniforge/Miniconda first." >&2
  exit 1
fi
export MAX_JOBS
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$CUDA_ARCH_LIST}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH

echo "FLAME target GPU: $TARGET_GPU (expected name: $EXPECTED_GPU_PATTERN, CUDA arch: $TORCH_CUDA_ARCH_LIST)"

source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda env update --name "$ENV_NAME" --file "$ENV_FILE"
else
  conda env create --name "$ENV_NAME" --file "$ENV_FILE"
fi
conda activate "$ENV_NAME"

if [[ "$SKIP_EXTENSIONS" -eq 0 ]]; then
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc not found; installing CUDA toolkit $CONDA_CUDA_TOOLKIT_VERSION inside $ENV_NAME."
    conda install --yes --name "$ENV_NAME" \
      --channel "nvidia/label/cuda-${CONDA_CUDA_TOOLKIT_VERSION}" \
      "cuda-toolkit=${CONDA_CUDA_TOOLKIT_VERSION}"
    hash -r
  fi
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc is unavailable after the project-local CUDA toolkit install." >&2
    exit 1
  fi
  if ! command -v gcc >/dev/null 2>&1; then
    echo "ERROR: gcc not found. GCC/G++ 11 is recommended." >&2
    exit 1
  fi
  NVCC_PATH="$(command -v nvcc)"
  CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$NVCC_PATH")")}"
  export CUDA_HOME
  NVCC_RELEASE="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | tail -n 1)"
  case "$NVCC_RELEASE" in
    12.4|12.5) ;;
    *)
      echo "ERROR: CUDA toolkit 12.4 or 12.5 is required, got '${NVCC_RELEASE:-unknown}'." >&2
      exit 1
      ;;
  esac
fi

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

  CUDNN_PATH="$(python -c 'from pathlib import Path; import nvidia.cudnn; print(Path(nvidia.cudnn.__file__).resolve().parent)')"
  export CUDNN_PATH
  export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-${CUDA_ARCH_LIST/./}}"
  export NVTE_CMAKE_BUILD_DIR="${NVTE_CMAKE_BUILD_DIR:-${TMPDIR:-$PROJECT_ROOT/.local/build}/transformer-engine-${TARGET_GPU}}"
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
