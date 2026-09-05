#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: activate the FLAME Conda environment first." >&2
  return 1 2>/dev/null || exit 1
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_DIR/Megatron-LM"
export FLAME_MOE_REPO_DIR="$REPO_DIR"
if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -x "$CONDA_PREFIX/bin/nvcc" ]]; then
    export CUDA_HOME="$CONDA_PREFIX"
  else
    export CUDA_HOME=/usr/local/cuda
  fi
fi

echo "FLAME/Megatron runtime activated"
echo "repo:   $REPO_DIR"
echo "conda:  $CONDA_PREFIX"
echo "python: $(command -v python)"
