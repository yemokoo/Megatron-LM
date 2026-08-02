#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "SLoRA reproduction status"
echo "========================="
echo

for model in Llama-3.1-8B-Instruct Qwen2.5-7B-Instruct; do
  path="${ROOT}/models/${model}"
  size="$(du -sh "${path}" 2>/dev/null | cut -f1)"
  complete="$(find "${path}" -maxdepth 1 -type f -name 'model-*.safetensors' 2>/dev/null | wc -l)"
  incomplete="$(find "${path}/.cache/huggingface/download" -type f -name '*.incomplete' -size +0c 2>/dev/null | wc -l)"
  echo "model=${model} size=${size:-0} complete_shards=${complete}/4 cache_partial_files=${incomplete}"
done

echo
if pgrep -af 'hf download|huggingface-cli download' >/dev/null; then
  echo "downloads:"
  pgrep -af 'hf download|huggingface-cli download'
else
  echo "downloads: none"
fi

echo
if [[ -x "${ROOT}/.venv-runtime/bin/python" ]]; then
  if "${ROOT}/.venv-runtime/bin/python" -c 'import torch, transformers, peft, trl, deepspeed' >/dev/null 2>&1; then
    echo "runtime: ready"
  else
    echo "runtime: incomplete (run scripts/setup_runtime.sh)"
  fi
else
  echo "runtime: missing (run scripts/setup_runtime.sh)"
fi

echo
echo "GPUs:"
nvidia-smi \
  --query-gpu=index,name,memory.used,memory.total \
  --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"

echo
echo "Methods:"
echo "  seq_lora slora_pre_released slora_pre slora_post"
echo "  ewc lwf"
echo "  gem_upstream gem_corrected"
echo "  olora_upstream olora_corrected"
echo "  loramoe (local compatible port)"
echo "  catalog: scripts/baselines/<model>.sh list"
