#!/usr/bin/env bash
set -euo pipefail

# Do not allow host user packages to shadow the project virtualenv.
unset PYTHONPATH

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}"
SCAFFOLD="$(cd "${ROOT}/../.." && pwd)"

METHOD="${1:?usage: eval_trace.sh <ewc|lwf|gem|olora> <upstream|corrected> <llama31|qwen25_7b>}"
VARIANT="${2:?usage: eval_trace.sh <ewc|lwf|gem|olora> <upstream|corrected> <llama31|qwen25_7b>}"
MODEL_KEY="${3:?usage: eval_trace.sh <ewc|lwf|gem|olora> <upstream|corrected> <llama31|qwen25_7b>}"

DATA_ROOT="${TRACE_DATA_ROOT:-${SCAFFOLD}/data/trace}"
OUTPUT_ROOT="${TRACE_OUTPUT_ROOT:-${SCAFFOLD}/results/full_runs}"
RUN_NAME="${RUN_NAME:-${MODEL_KEY}/${METHOD}_${VARIANT}}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
EVAL_DIR="${RUN_DIR}/evaluation"
INFERENCE_BATCH="${INFERENCE_BATCH:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DRY_RUN="${DRY_RUN:-0}"
export CUDA_VISIBLE_DEVICES

TASKS=(C-STANCE FOMC MeetingBank Py150 ScienceQA NumGLUE-cm NumGLUE-ds 20Minuten)
TASK_CSV="$(IFS=,; echo "${TASKS[*]}")"

case "${METHOD}" in
  ewc) TRACE_METHOD="EWC" ;;
  lwf) TRACE_METHOD="LwF" ;;
  gem) TRACE_METHOD="GEM" ;;
  olora) TRACE_METHOD="O-LoRA" ;;
  *)
    echo "[ERROR] Unknown method: ${METHOD}" >&2
    exit 2
    ;;
esac

case "${MODEL_KEY}" in
  llama31)
    MODEL_PATH="${SLORA_LLAMA31_PATH:-${SCAFFOLD}/models/Llama-3.1-8B-Instruct}"
    ;;
  qwen25_7b)
    MODEL_PATH="${SLORA_QWEN25_7B_PATH:-${SCAFFOLD}/models/Qwen2.5-7B-Instruct}"
    ;;
  *)
    echo "[ERROR] Unknown model key: ${MODEL_KEY}" >&2
    exit 2
    ;;
esac

command=(
  python3
  inference/infer_single.py
  --data_path "${DATA_ROOT}"
  --data_output_path "${RUN_DIR}/eval_data_cache"
  --model_name_or_path "${MODEL_PATH}"
  --inference_model_path "${CHECKPOINT_DIR}"
  --inference_tasks "${TASK_CSV}"
  --inference_output_path "${EVAL_DIR}"
  --inference_batch "${INFERENCE_BATCH}"
  --max_prompt_len 1024
  --max_ans_len 512
  --temperature 0.1
  --seed 2025
  --CL_method "${TRACE_METHOD}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf "%q " "${command[@]}"
  printf "\n"
  exit 0
fi

[[ -d "${CHECKPOINT_DIR}/7" ]] || {
  echo "[ERROR] Final TRACE checkpoint is missing: ${CHECKPOINT_DIR}/7" >&2
  exit 3
}

mkdir -p "${EVAL_DIR}"
cd "${ROOT}"
"${command[@]}" 2>&1 | tee "${EVAL_DIR}/eval.log"
