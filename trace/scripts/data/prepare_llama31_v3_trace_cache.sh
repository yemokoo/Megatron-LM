#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export SLORA_LLAMA31_PATH="${SLORA_LLAMA31_PATH:?export SLORA_LLAMA31_PATH (Llama-3.1-8B-Instruct)}"
export OURS_LLAMA31_TOKEN_CACHE="${OURS_LLAMA31_TOKEN_CACHE:-${ROOT}/cache/tokenized/llama31_8b_base/slora_chat_full_len1024}"

exec "${ROOT}/scripts/data/prepare_llama31_trace_cache.sh" "$@"
