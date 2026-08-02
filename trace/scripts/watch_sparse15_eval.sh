#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHAIN_SESSION="${CHAIN_SESSION:-slora-eval-sparse15-five}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
CHAIN_LOG="${CHAIN_LOG:-${ROOT}/results/eval_chains/llama31_sparse15_five_20260728/full.log}"
SUMMARY_DIR="${ROOT}/results/summaries/llama31"
METHODS=(slora_pre_released slora_post seq_lora ours_lora_moe_v1 ours_lora_moe_v2)

while true; do
  echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
  if tmux has-session -t "${CHAIN_SESSION}" 2>/dev/null; then
    echo "chain_session=alive"
  else
    echo "chain_session=dead"
  fi

  ps -eo pid,pgid,etime,cmd --sort=pgid |
    rg 'run_suite.sh eval llama31|model_diverse_gen_batch|evaluate_Ours_LoRA_MoE' ||
    true

  if [[ -f "${CHAIN_LOG}" ]]; then
    error_count="$(
      rg -c 'CUDA out of memory|OutOfMemoryError|Traceback|\[ERROR\]|NaN|malformed|Killed' \
        "${CHAIN_LOG}" 2>/dev/null || true
    )"
    echo "error_matches=${error_count:-0}"
    tail -n 8 "${CHAIN_LOG}"
  else
    echo "chain_log=missing"
  fi

  summary_count=0
  for method in "${METHODS[@]}"; do
    if [[ -s "${SUMMARY_DIR}/${method}.json" ]]; then
      summary_count=$((summary_count + 1))
    fi
  done
  echo "required_summaries=${summary_count}/5"

  nvidia-smi \
    --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null || true

  if [[ "${summary_count}" -eq 5 ]]; then
    echo "watch_status=all_summaries_present"
    exit 0
  fi
  sleep "${INTERVAL_SECONDS}"
done
