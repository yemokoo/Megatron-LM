#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OURS_ROOT="${OURS_LORAMOE_ROOT:-${ROOT}/implementations/llmcl_benchmark}"
PYTHON_BIN="${OURS_LORAMOE_PYTHON:-${TRACE_PYTHON:-${ROOT}/.venv-runtime/bin/python}}"
DATA_ROOT="${TRACE_DATA_ROOT:-${ROOT}/data/trace}"
MODEL_PATH="${SLORA_LLAMA31_PATH:?export SLORA_LLAMA31_PATH (Llama-3.1-8B-Instruct)}"
CACHE_ROOT="${OURS_LLAMA31_TOKEN_CACHE:-${ROOT}/cache/tokenized/llama31_8b_base/slora_chat_full_len1024}"
REPLAY_MANIFEST="${OURS_REPLAY_MANIFEST:-${ROOT}/manifests/replay/trace_seed2025_random50_per_task.json}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec "${PYTHON_BIN}" "${OURS_ROOT}/scripts/prepare_trace_token_cache.py" \
  --data-root "${DATA_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${CACHE_ROOT}" \
  --replay-manifest "${REPLAY_MANIFEST}" \
  --seed "${OURS_REPLAY_SUBSET_SEED:-2025}" \
  --samples-per-task 50 \
  --max-length 1024 \
  --batch-size "${TOKEN_CACHE_BATCH_SIZE:-64}" \
  "$@"
