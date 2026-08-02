#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${HF_BIN:-}" ]]; then
    HF="${HF_BIN}"
elif [[ -x "${ROOT}/.venv-runtime/bin/hf" ]]; then
    HF="${ROOT}/.venv-runtime/bin/hf"
else
    HF="$(command -v hf || true)"
fi
[[ -n "${HF}" ]] || { echo "[ERROR] hf CLI is missing; set HF_BIN" >&2; exit 1; }
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
RETRY_DELAY="${HF_RETRY_DELAY:-2}"

if ! "${HF}" auth whoami >/dev/null 2>&1; then
    echo "[ERROR] This shell is not authenticated with Hugging Face." >&2
    echo "Run: ${HF} auth login" >&2
    exit 2
fi

download_model() {
    local repo_id="$1"
    local local_dir="$2"

    mkdir -p "${local_dir}"
    echo "[DOWNLOAD] ${repo_id} -> ${local_dir}"
    until "${HF}" download "${repo_id}" --local-dir "${local_dir}" --max-workers 1; do
        echo "[RETRY] ${repo_id}: restarting in ${RETRY_DELAY}s" >&2
        sleep "${RETRY_DELAY}"
    done
}

download_model \
    "meta-llama/Llama-3.1-8B-Instruct" \
    "${ROOT}/models/Llama-3.1-8B-Instruct"

download_model \
    "Qwen/Qwen2.5-7B-Instruct" \
    "${ROOT}/models/Qwen2.5-7B-Instruct"

python3 "${ROOT}/scripts/preflight.py" \
    --mode full \
    --models llama31_8b_instruct qwen25_7b_instruct
