#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${1:?internal usage: _run_model.sh <model> <command> [method]}"
COMMAND="${2:-list}"
METHOD="${3:-}"
CATALOG="${ROOT}/scripts/baselines/methods.tsv"

case "${MODEL}" in
  llama31|qwen25_7b) ;;
  *)
    echo "[ERROR] Unknown model: ${MODEL}" >&2
    exit 2
    ;;
esac

show_list() {
  awk -F '\t' '
    NR == 1 { printf "%-20s %-10s %-16s %s\n", $1, $2, $3, $5; next }
    { printf "%-20s %-10s %-16s %s\n", $1, $2, $3, $5 }
  ' "${CATALOG}"
}

status_of() {
  awk -F '\t' -v method="$1" '
    NR > 1 && $1 == method { print $3; found = 1; exit }
    END { if (!found) exit 1 }
  ' "${CATALOG}"
}

run_one() {
  local action="$1"
  local method="$2"
  local status

  case "${method}" in
    seqlora) method="seq_lora" ;;
    gem)
      echo "[ERROR] gem은 모호합니다: gem_upstream 또는 gem_corrected를 지정하세요." >&2
      return 2
      ;;
    olora)
      echo "[ERROR] olora는 모호합니다: olora_upstream 또는 olora_corrected를 지정하세요." >&2
      return 2
      ;;
  esac

  status="$(status_of "${method}")" || {
    echo "[ERROR] Unknown method: ${method}" >&2
    show_list >&2
    return 2
  }
  if [[ "${status}" == "unavailable" ]]; then
    echo "[ERROR] ${method}: 공개 실행 코드가 없어 목록에만 기록된 방법입니다." >&2
    return 3
  fi

  echo "[BASELINE] model=${MODEL} action=${action} method=${method} status=${status}"
  if [[ "${action}" == "validate" ]]; then
    local validation_root="${BASELINE_VALIDATE_ROOT:-/tmp/slora_repro_validate}/${MODEL}/${method}"
    if [[ "${method}" == ours_lora_moe_v* ]]; then
      OURS_LORAMOE_OUTPUT_ROOT="${validation_root}" \
        "${ROOT}/scripts/baselines/_run_ours_lora_moe.sh" \
        "${action}" "${MODEL}" "${method##ours_lora_moe_}"
    elif [[ "${method}" == "loramoe" ]]; then
      LORAMOE_OUTPUT_ROOT="${validation_root}" \
        "${ROOT}/scripts/baselines/_run_loramoe.sh" "${action}" "${MODEL}"
    else
      SLORA_OUTPUT_ROOT="${validation_root}" \
      SLORA_RELEASED_OUTPUT_ROOT="${validation_root}" \
      TRACE_OUTPUT_ROOT="${validation_root}" \
        "${ROOT}/scripts/run_experiment.sh" "${action}" "${method}" "${MODEL}"
    fi
  elif [[ "${method}" == ours_lora_moe_v* ]]; then
    "${ROOT}/scripts/baselines/_run_ours_lora_moe.sh" \
      "${action}" "${MODEL}" "${method##ours_lora_moe_}"
  elif [[ "${method}" == "loramoe" ]]; then
    "${ROOT}/scripts/baselines/_run_loramoe.sh" "${action}" "${MODEL}"
  else
    "${ROOT}/scripts/run_experiment.sh" "${action}" "${method}" "${MODEL}"
  fi
}

case "${COMMAND}" in
  list)
    show_list
    ;;
  validate|train|eval|all)
    [[ -n "${METHOD}" ]] || {
      echo "usage: $(basename "$0") ${COMMAND} <method>" >&2
      exit 2
    }
    run_one "${COMMAND}" "${METHOD}"
    ;;
  suite)
    ACTION="${METHOD:-validate}"
    case "${ACTION}" in
      validate|train|eval|all) ;;
      *)
        echo "[ERROR] suite action must be validate, train, eval, or all" >&2
        exit 2
        ;;
    esac
    METHODS=(
      seq_lora
      slora_pre_released
      slora_pre
      slora_post
      ewc
      lwf
      gem_upstream
      gem_corrected
      olora_upstream
      olora_corrected
      ours_lora_moe_v1
      ours_lora_moe_v1_expert_first
      ours_lora_moe_v2
      ours_lora_moe_v2_new
      ours_lora_moe_v2_new_top4
      ours_lora_moe_v2_5
      ours_lora_moe_v3
      ours_lora_moe_v3_new_top4
      loramoe
    )
    for method in "${METHODS[@]}"; do
      run_one "${ACTION}" "${method}"
    done
    ;;
  *)
    echo "usage: $(basename "$0") <list|validate|train|eval|all|suite> [method|suite-action]" >&2
    exit 2
    ;;
esac
