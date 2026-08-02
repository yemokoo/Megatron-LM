#!/usr/bin/env bash
set -euo pipefail

# Do not allow host user packages to shadow the project virtualenv.
unset PYTHONPATH

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="${ROOT}/.venv-runtime/bin"
if [[ -x "${VENV_BIN}/python" ]]; then
  export PATH="${VENV_BIN}:${PATH}"
fi


ACTION="${1:?usage: run_experiment.sh <validate|train|eval|all> <method> <llama31|qwen25_7b>}"
METHOD="${2:?usage: run_experiment.sh <validate|train|eval|all> <method> <llama31|qwen25_7b>}"
MODEL="${3:?usage: run_experiment.sh <validate|train|eval|all> <method> <llama31|qwen25_7b>}"

case "${MODEL}" in
  llama31) VERIFY_MODEL_PATH="${ROOT}/models/Llama-3.1-8B-Instruct" ;;
  qwen25_7b) VERIFY_MODEL_PATH="${ROOT}/models/Qwen2.5-7B-Instruct" ;;
  *)
    echo "[ERROR] Unknown model: ${MODEL}" >&2
    exit 2
    ;;
esac

case "${METHOD}" in
  seq_lora)
    FAMILY="slora"
    TRAIN_ARGS=(seq "${MODEL}")
    EVAL_ARGS=(seq "${MODEL}")
    ;;
  slora_pre)
    FAMILY="slora"
    TRAIN_ARGS=(pre "${MODEL}")
    EVAL_ARGS=(pre "${MODEL}")
    ;;
  slora_pre_released)
    FAMILY="slora_released"
    TRAIN_ARGS=(pre "${MODEL}")
    EVAL_ARGS=(pre "${MODEL}")
    ;;
  slora_post)
    FAMILY="slora"
    TRAIN_ARGS=(post "${MODEL}")
    EVAL_ARGS=(post "${MODEL}")
    ;;
  ewc)
    FAMILY="trace"
    TRAIN_ARGS=(ewc upstream "${MODEL}")
    EVAL_ARGS=(ewc upstream "${MODEL}")
    ;;
  lwf)
    FAMILY="trace"
    TRAIN_ARGS=(lwf upstream "${MODEL}")
    EVAL_ARGS=(lwf upstream "${MODEL}")
    ;;
  gem_upstream)
    FAMILY="trace"
    TRAIN_ARGS=(gem upstream "${MODEL}")
    EVAL_ARGS=(gem upstream "${MODEL}")
    ;;
  gem_corrected)
    FAMILY="trace"
    TRAIN_ARGS=(gem corrected "${MODEL}")
    EVAL_ARGS=(gem corrected "${MODEL}")
    ;;
  olora_upstream)
    FAMILY="trace"
    TRAIN_ARGS=(olora upstream "${MODEL}")
    EVAL_ARGS=(olora upstream "${MODEL}")
    ;;
  olora_corrected)
    FAMILY="trace"
    TRAIN_ARGS=(olora corrected "${MODEL}")
    EVAL_ARGS=(olora corrected "${MODEL}")
    ;;
  *)
    echo "[ERROR] Unknown method: ${METHOD}" >&2
    exit 2
    ;;
esac

case "${FAMILY}" in
  slora)
    TRAIN_SCRIPT="${ROOT}/implementations/SLoRA-repro/scripts/repro/train_trace.sh"
    EVAL_SCRIPT="${ROOT}/implementations/SLoRA-repro/scripts/repro/eval_trace.sh"
    ;;
  slora_released)
    TRAIN_SCRIPT="${ROOT}/implementations/SLoRA-upstream-port/scripts/repro/train_trace.sh"
    EVAL_SCRIPT="${ROOT}/implementations/SLoRA-upstream-port/scripts/repro/eval_trace.sh"
    export SLORA_OUTPUT_ROOT="${SLORA_RELEASED_OUTPUT_ROOT:-${ROOT}/results/full_runs_upstream_code}"
    ;;
  trace)
    TRAIN_SCRIPT="${ROOT}/implementations/TRACE-repro/scripts/repro/train_trace.sh"
    EVAL_SCRIPT="${ROOT}/implementations/TRACE-repro/scripts/repro/eval_trace.sh"
    ;;
esac

run_train() {
  "${TRAIN_SCRIPT}" "${TRAIN_ARGS[@]}"
}

run_eval() {
  "${EVAL_SCRIPT}" "${EVAL_ARGS[@]}"
}

run_collect() {
  local command=(python3 "${ROOT}/scripts/collect_results.py" --method "${METHOD}" --model "${MODEL}")
  if [[ "${EVAL_SPARSE_15:-0}" == "1" ]]; then command+=(--sparse-15); fi
  "${command[@]}"
}

if [[ "${ACTION}" != "validate" ]]; then
  [[ -x "${VENV_BIN}/python" ]] || {
    echo "[ERROR] Runtime is missing. Run: ${ROOT}/scripts/setup_runtime.sh" >&2
    exit 4
  }
  python3 "${ROOT}/scripts/verify_model_files.py" "${VERIFY_MODEL_PATH}"
  python3 "${ROOT}/scripts/runtime_preflight.py" --model "${MODEL}" --world-size "${WORLD_SIZE:-4}"
fi

case "${ACTION}" in
  validate)
    echo "[VALIDATE] ${METHOD} / ${MODEL}"
    DRY_RUN=1 run_train
    DRY_RUN=1 run_eval
    ;;
  train)
    run_train
    ;;
  eval)
    run_eval
    run_collect
    ;;
  all)
    run_train
    run_eval
    run_collect
    ;;
  *)
    echo "[ERROR] Unknown action: ${ACTION}" >&2
    exit 2
    ;;
esac
