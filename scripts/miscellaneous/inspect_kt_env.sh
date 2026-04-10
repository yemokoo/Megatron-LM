#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${1:-$REPO_DIR/.local/diagnostics/kt-env-$STAMP}"

mkdir -p "$OUT_DIR"

write_cmd() {
  local name="$1"
  shift
  {
    echo "# $*"
    "$@"
  } >"$OUT_DIR/$name" 2>&1 || true
}

cat >"$OUT_DIR/summary.txt" <<EOF
timestamp=$(date -Is)
repo_dir=$REPO_DIR
out_dir=$OUT_DIR
host=$(hostname)
user=$(whoami)
pwd=$(pwd)
EOF

write_cmd os.txt uname -a
write_cmd date.txt date
write_cmd whoami.txt whoami
write_cmd hostname.txt hostname
write_cmd shell.txt bash -lc 'echo "$SHELL"'
write_cmd path.txt bash -lc 'printf "%s\n" "$PATH"'
write_cmd env_filtered.txt bash -lc 'env | sort | grep -E "^(CUDA|CUDNN|NCCL|LD_LIBRARY_PATH|PATH|PYTHON|PIP|VIRTUAL_ENV|CONDA|WANDB|BNB|HF_|TRANSFORMERS_|TOKENIZERS_)" || true'
write_cmd env_all.txt bash -lc 'env | sort'
write_cmd os_release.txt bash -lc 'cat /etc/os-release'
write_cmd cpu.txt bash -lc 'lscpu'
write_cmd mem.txt bash -lc 'free -h'
write_cmd disk.txt bash -lc 'df -h'
write_cmd ulimit.txt bash -lc 'ulimit -a'
write_cmd nvidia_smi.txt nvidia-smi
write_cmd nvidia_query.txt nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,temperature.gpu,power.draw --format=csv,noheader
write_cmd nvcc.txt bash -lc 'nvcc --version'
write_cmd which.txt bash -lc 'which python || true; which python3 || true; which pip || true; which pip3 || true; which git || true'
write_cmd git_status.txt git -C "$REPO_DIR" status --short
write_cmd git_rev.txt git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD
write_cmd git_head.txt git -C "$REPO_DIR" rev-parse HEAD
write_cmd git_submodule.txt git -C "$REPO_DIR" submodule status --recursive
write_cmd python_version.txt bash -lc 'python --version || true; python3 --version || true'
write_cmd pip_version.txt bash -lc 'python -m pip --version || true; python3 -m pip --version || true'
write_cmd pip_freeze.txt bash -lc 'python -m pip freeze || true'
write_cmd venvs.txt bash -lc 'ls -ld "$REPO_DIR"/.venv* 2>/dev/null || true'

python_info() {
  python - <<'PY'
import importlib
import json
import os
import site
import sys

packages = [
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "apex",
    "transformer_engine",
    "flash_attn",
    "wandb",
    "bitsandbytes",
]

info = {
    "sys_executable": sys.executable,
    "sys_version": sys.version,
    "sys_prefix": sys.prefix,
    "base_prefix": getattr(sys, "base_prefix", ""),
    "virtual_env": os.environ.get("VIRTUAL_ENV", ""),
    "site_packages": site.getsitepackages() if hasattr(site, "getsitepackages") else [],
}

for name in packages:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        info[name] = {"ok": True, "version": version}
    except Exception as exc:
        info[name] = {"ok": False, "error": repr(exc)}

try:
    import torch
    info["torch_runtime"] = {
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    }
except Exception as exc:
    info["torch_runtime"] = {"ok": False, "error": repr(exc)}

print(json.dumps(info, indent=2, sort_keys=True))
PY
}

python_info >"$OUT_DIR/python_imports.json" 2>&1 || true

echo "KT environment snapshot written to: $OUT_DIR"
