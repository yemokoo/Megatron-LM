#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI="${HF_LEGACY_CLI:-huggingface-cli}"
RETRY_DELAY="${HF_RETRY_DELAY:-2}"

export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1

if ! "${CLI}" whoami >/dev/null 2>&1; then
  echo "[ERROR] The legacy Hugging Face CLI is not authenticated." >&2
  echo "Run: huggingface-cli login" >&2
  exit 2
fi

download_file() {
  local repo_id="$1"
  local local_dir="$2"
  local filename="$3"

  mkdir -p "${local_dir}"
  while true; do
    echo "[DOWNLOAD] ${repo_id}/${filename}"
    if "${CLI}" download \
      "${repo_id}" \
      "${filename}" \
      --local-dir "${local_dir}" \
      --resume-download \
      --local-dir-use-symlinks False; then
      return 0
    fi
    echo "[RETRY] ${filename}: restarting in ${RETRY_DELAY}s" >&2
    sleep "${RETRY_DELAY}"
  done
}

download_repo_files() {
  local repo_id="$1"
  local local_dir="$2"
  shift 2
  local filename
  for filename in "$@"; do
    download_file "${repo_id}" "${local_dir}" "${filename}"
  done
}

download_repo_files \
  "meta-llama/Llama-3.1-8B-Instruct" \
  "${ROOT}/models/Llama-3.1-8B-Instruct" \
  config.json \
  generation_config.json \
  model.safetensors.index.json \
  model-00001-of-00004.safetensors \
  model-00002-of-00004.safetensors \
  model-00003-of-00004.safetensors \
  model-00004-of-00004.safetensors \
  tokenizer.json \
  tokenizer_config.json \
  special_tokens_map.json

download_repo_files \
  "Qwen/Qwen2.5-7B-Instruct" \
  "${ROOT}/models/Qwen2.5-7B-Instruct" \
  config.json \
  generation_config.json \
  model.safetensors.index.json \
  model-00001-of-00004.safetensors \
  model-00002-of-00004.safetensors \
  model-00003-of-00004.safetensors \
  model-00004-of-00004.safetensors \
  tokenizer.json \
  tokenizer_config.json \
  merges.txt \
  vocab.json

python3 "${ROOT}/scripts/preflight.py" \
  --mode full \
  --models llama31_8b_instruct qwen25_7b_instruct
