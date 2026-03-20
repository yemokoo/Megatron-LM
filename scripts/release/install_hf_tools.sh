#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.conda/envs/flame3090/bin/python}"
PIP_BIN="${PIP_BIN:-$PYTHON_BIN -m pip}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: project python not found under .conda/envs/flame3090"
  exit 1
fi

"$PYTHON_BIN" --version
"$PYTHON_BIN" -m pip install --upgrade "huggingface_hub[cli]>=0.30"
"$PYTHON_BIN" - <<'PY'
import huggingface_hub
print('huggingface_hub', huggingface_hub.__version__)
PY

echo "HF tooling is ready."
echo "Next steps:"
echo "  1. ~/.local/bin/hf auth login"
echo "  2. .conda/envs/flame3090/bin/python scripts/release/upload_models_to_hf.py --dry-run"
echo "  3. .conda/envs/flame3090/bin/python scripts/release/upload_models_to_hf.py"
