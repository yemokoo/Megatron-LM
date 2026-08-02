#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:?usage: run_suite.sh <validate|train|eval|all> <llama31|qwen25_7b>}"
MODEL="${2:?usage: run_suite.sh <validate|train|eval|all> <llama31|qwen25_7b>}"

if [[ -n "${SUITE_METHODS:-}" ]]; then
  read -r -a METHODS <<< "${SUITE_METHODS}"
else
  METHODS=(
    seq_lora
    slora_pre
    slora_post
    ewc
    lwf
    gem_upstream
    gem_corrected
    olora_upstream
    olora_corrected
  )
fi

for method in "${METHODS[@]}"; do
  echo
  echo "================================================================"
  echo "[SUITE] action=${ACTION} model=${MODEL} method=${method}"
  echo "================================================================"
  "${ROOT}/scripts/baselines/_run_model.sh" "${MODEL}" "${ACTION}" "${method}"
done
