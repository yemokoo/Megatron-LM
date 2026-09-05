#!/usr/bin/env bash
set -euo pipefail

# Do not allow host user packages to shadow the project virtualenv.
unset PYTHONPATH

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}"
SCAFFOLD="$(cd "${ROOT}/../.." && pwd)"
PYTHON_BIN="${SLORA_EVAL_PYTHON:-${TRACE_PYTHON:-${SCAFFOLD}/.venv-runtime/bin/python}}"
[[ -x "${PYTHON_BIN}" ]] || {
  echo "[ERROR] SLoRA evaluation Python is not executable: ${PYTHON_BIN}" >&2
  exit 4
}

METHOD="${1:?usage: eval_trace.sh <seq|pre|post> <llama31|qwen25_7b>}"
MODEL_KEY="${2:?usage: eval_trace.sh <seq|pre|post> <llama31|qwen25_7b>}"

DATA_ROOT="${TRACE_DATA_ROOT:-${SCAFFOLD}/data/trace}"
OUTPUT_ROOT="${SLORA_OUTPUT_ROOT:-${SCAFFOLD}/results/full_runs}"
RESULT_RUN_DIR="${OUTPUT_ROOT}/${MODEL_KEY}/${METHOD}"
CHECKPOINT_METHOD="${METHOD}"
if [[ "${METHOD}" == "post" ]]; then
  CHECKPOINT_METHOD="seq"
fi
CHECKPOINT_RUN_DIR="${OUTPUT_ROOT}/${MODEL_KEY}/${CHECKPOINT_METHOD}"
EVAL_DIR="${RESULT_RUN_DIR}/evaluation"
EVAL_ALL_ROUNDS="${EVAL_ALL_ROUNDS:-1}"
EVAL_SPARSE_15="${EVAL_SPARSE_15:-0}"
EVAL_BATCH="${SLORA_EVAL_BATCH:-4}"
EVAL_SHARD_COUNT="${EVAL_SHARD_COUNT:-1}"
EVAL_SHARD_INDEX="${EVAL_SHARD_INDEX:-0}"
EVAL_CONTINUE_ON_CELL_ERROR="${EVAL_CONTINUE_ON_CELL_ERROR:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DRY_RUN="${DRY_RUN:-0}"
export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

[[ "${EVAL_BATCH}" =~ ^[1-9][0-9]*$ ]] || {
  echo "[ERROR] SLORA_EVAL_BATCH must be a positive integer: ${EVAL_BATCH}" >&2
  exit 2
}
[[ "${EVAL_SPARSE_15}" == "0" || "${EVAL_SPARSE_15}" == "1" ]] || {
  echo "[ERROR] EVAL_SPARSE_15 must be 0 or 1: ${EVAL_SPARSE_15}" >&2
  exit 2
}
[[ "${EVAL_SHARD_COUNT}" =~ ^[1-9][0-9]*$ ]] || {
  echo "[ERROR] EVAL_SHARD_COUNT must be positive: ${EVAL_SHARD_COUNT}" >&2
  exit 2
}
[[ "${EVAL_SHARD_INDEX}" =~ ^[0-9]+$ && "${EVAL_SHARD_INDEX}" -lt "${EVAL_SHARD_COUNT}" ]] || {
  echo "[ERROR] EVAL_SHARD_INDEX must be in [0, EVAL_SHARD_COUNT): ${EVAL_SHARD_INDEX}" >&2
  exit 2
}
[[ "${EVAL_CONTINUE_ON_CELL_ERROR}" == "0" || "${EVAL_CONTINUE_ON_CELL_ERROR}" == "1" ]] || {
  echo "[ERROR] EVAL_CONTINUE_ON_CELL_ERROR must be 0 or 1" >&2
  exit 2
}

TASKS=(C-STANCE FOMC MeetingBank Py150 ScienceQA NumGLUE-cm NumGLUE-ds 20Minuten)

case "${METHOD}" in
  seq) DENOISED_MODE="seq_lora" ;;
  pre) DENOISED_MODE="slora_pre" ;;
  post) DENOISED_MODE="slora_post" ;;
  *)
    echo "[ERROR] Unknown method: ${METHOD}" >&2
    exit 2
    ;;
esac

case "${MODEL_KEY}" in
  llama31)
    MODEL_PATH="${SLORA_LLAMA31_PATH:-${SCAFFOLD}/models/Llama-3.1-8B-Instruct}"
    CONV_MODE="llama3"
    ;;
  qwen25_7b)
    MODEL_PATH="${SLORA_QWEN25_7B_PATH:-${SCAFFOLD}/models/Qwen2.5-7B-Instruct}"
    CONV_MODE="qwen"
    ;;
  *)
    echo "[ERROR] Unknown model key: ${MODEL_KEY}" >&2
    exit 2
    ;;
esac

if [[ "${DRY_RUN}" != "1" && ! -f "${CHECKPOINT_RUN_DIR}/order8/adapter_config.json" ]]; then
  echo "[ERROR] Final SLoRA checkpoint is missing: ${CHECKPOINT_RUN_DIR}/order8" >&2
  exit 3
fi

if [[ "${EVAL_ALL_ROUNDS}" == "1" ]]; then
  START_ROUND=1
else
  START_ROUND=8
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${EVAL_DIR}"
  FAILURE_FILE="${EVAL_DIR}/failed_cells.shard${EVAL_SHARD_INDEX}.tsv"
  printf 'round\ttask\tstage\tlog\n' > "${FAILURE_FILE}"
  {
    echo "method=${METHOD}"
    echo "checkpoint_method=${CHECKPOINT_METHOD}"
    echo "checkpoint_run_dir=${CHECKPOINT_RUN_DIR}"
    echo "denoised_mode=${DENOISED_MODE}"
    echo "eval_all_rounds=${EVAL_ALL_ROUNDS}"
    echo "eval_sparse_15=${EVAL_SPARSE_15}"
    echo "eval_batch=${EVAL_BATCH}"
    echo "temperature=0"
    echo "top_p=1.0"
    echo "num_beams=1"
    echo "max_new_tokens=1024"
    echo "padding_side=left"
  } > "${EVAL_DIR}/eval.env"
fi

record_cell_failure() {
  local round="$1" task="$2" stage="$3" log="$4"
  printf '%s\t%s\t%s\t%s\n' "${round}" "${task}" "${stage}" "${log}" \
    >> "${FAILURE_FILE}"
  echo "[SKIP CELL] order${round}.${task} failed at ${stage}; log=${log}" >&2
  if [[ "${EVAL_CONTINUE_ON_CELL_ERROR}" != "1" ]]; then
    exit 5
  fi
}

cell_index=0
for ((round=START_ROUND; round<=8; round++)); do
  for ((task_index=0; task_index<round; task_index++)); do
    if [[ "${EVAL_SPARSE_15}" == "1" && "${round}" -lt 8 && "${task_index}" -ne $((round - 1)) ]]; then
      continue
    fi
    assigned_shard=$((cell_index % EVAL_SHARD_COUNT))
    cell_index=$((cell_index + 1))
    if [[ "${assigned_shard}" -ne "${EVAL_SHARD_INDEX}" ]]; then
      continue
    fi
    task="${TASKS[$task_index]}"
    task_dir="${EVAL_DIR}/order${round}/${task}"
    infer_file="${task_dir}/infer.jsonl"
    mkdir -p "${task_dir}"

    infer_command=(
      "${PYTHON_BIN}" -u -m src.eval.model_diverse_gen_batch
      --model-path "${CHECKPOINT_RUN_DIR}"
      --model-base "${MODEL_PATH}"
      --question-file "${DATA_ROOT}/${task}/test.json"
      --answers-file "${infer_file}"
      --temperature 0
      --conv-mode "${CONV_MODE}"
      --denoised_mode "${DENOISED_MODE}"
      --test_order "${round}"
      --batch_size "${EVAL_BATCH}"
      --resume
    )
    eval_command=(
      "${PYTHON_BIN}" -u -m src.eval.eval_trace
      --input_file "${infer_file}"
      --output_file "${task_dir}/wrong.jsonl"
    )

    if [[ "${DRY_RUN}" == "1" ]]; then
      printf "%q " "${infer_command[@]}"
      printf "\n"
      printf "%q " "${eval_command[@]}"
      printf "\n"
      continue
    fi

    cd "${ROOT}"
    if ! "${infer_command[@]}" 2>&1 | tee "${task_dir}/infer.log"; then
      record_cell_failure "${round}" "${task}" infer "${task_dir}/infer.log"
      continue
    fi
    if ! "${PYTHON_BIN}" - "${DATA_ROOT}/${task}/test.json" "${infer_file}" <<'PY'
import json
import sys
from pathlib import Path

questions = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
lines = Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()
if len(lines) != len(questions):
    raise SystemExit(f"[ERROR] incomplete inference output: {len(lines)} != {len(questions)}")
for line_no, line in enumerate(lines, 1):
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[ERROR] malformed JSONL line {line_no}: {exc}") from exc
    missing = {"prompt", "text", "solution"} - record.keys()
    if missing:
        raise SystemExit(f"[ERROR] JSONL line {line_no} missing keys: {sorted(missing)}")
print(f"[OK] validated {len(lines)} inference records")
PY
    then
      record_cell_failure "${round}" "${task}" validate "${task_dir}/infer.log"
      continue
    fi
    if ! "${eval_command[@]}" 2>&1 | tee "${task_dir}/eval.log"; then
      record_cell_failure "${round}" "${task}" score "${task_dir}/eval.log"
      continue
    fi
  done
done

if [[ "${DRY_RUN}" != "1" ]]; then
  failure_count=$(( $(wc -l < "${FAILURE_FILE}") - 1 ))
  echo "[EVAL COMPLETE] shard=${EVAL_SHARD_INDEX}/${EVAL_SHARD_COUNT} failed_cells=${failure_count}"
fi
