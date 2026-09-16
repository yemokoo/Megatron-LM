#!/usr/bin/env bash
set -euo pipefail

# Do not allow host user packages to shadow the project virtualenv.
unset PYTHONPATH

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}"
SCAFFOLD="$(cd "${ROOT}/../.." && pwd)"

# Resolve python from the project virtualenv rather than whatever PATH the
# caller happens to carry: a chain launched from a plain login shell picks up
# the conda base interpreter, which has no torch.
VENV_BIN="${SCAFFOLD}/.venv-runtime/bin"
if [[ -x "${VENV_BIN}/python3" ]]; then
  export PATH="${VENV_BIN}:${PATH}"
fi
export PYTHONNOUSERSITE=1

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
EVAL_BATCH="${SLORA_EVAL_BATCH:-32}"
# Per-task generation batch. Decoding is bounded by the LONGEST row in a batch:
# one rare 1024-token continuation keeps every finished row in the batch alive,
# so the tasks that emit long text are faster at a small batch while the
# short-answer tasks are far faster at a large one. The Ours sparse-15 runner
# tuned this table (run_ours_sparse15_optimized.py:446-459) against its own
# evaluator; the two slow-generation tasks are left near the value this port
# has actually been measured at (4) rather than copied over, because that
# tuning does not transfer across evaluators unverified.
BATCH_C_STANCE="${SLORA_BATCH_C_STANCE:-32}"
BATCH_FOMC="${SLORA_BATCH_FOMC:-32}"
BATCH_MEETINGBANK="${SLORA_BATCH_MEETINGBANK:-4}"
BATCH_PY150="${SLORA_BATCH_PY150:-8}"
BATCH_SCIENCEQA="${SLORA_BATCH_SCIENCEQA:-128}"
BATCH_NUMGLUE_CM="${SLORA_BATCH_NUMGLUE_CM:-32}"
BATCH_NUMGLUE_DS="${SLORA_BATCH_NUMGLUE_DS:-32}"
BATCH_20MINUTEN="${SLORA_BATCH_20MINUTEN:-8}"

batch_for_task() {
  case "$1" in
    C-STANCE)    echo "${BATCH_C_STANCE}" ;;
    FOMC)        echo "${BATCH_FOMC}" ;;
    MeetingBank) echo "${BATCH_MEETINGBANK}" ;;
    Py150)       echo "${BATCH_PY150}" ;;
    ScienceQA)   echo "${BATCH_SCIENCEQA}" ;;
    NumGLUE-cm)  echo "${BATCH_NUMGLUE_CM}" ;;
    NumGLUE-ds)  echo "${BATCH_NUMGLUE_DS}" ;;
    20Minuten)   echo "${BATCH_20MINUTEN}" ;;
    *)           echo "${EVAL_BATCH}" ;;
  esac
}
EVAL_SHARD_COUNT="${EVAL_SHARD_COUNT:-1}"
EVAL_SHARD_INDEX="${EVAL_SHARD_INDEX:-0}"
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
  {
    echo "method=${METHOD}"
    echo "checkpoint_method=${CHECKPOINT_METHOD}"
    echo "checkpoint_run_dir=${CHECKPOINT_RUN_DIR}"
    echo "denoised_mode=${DENOISED_MODE}"
    echo "eval_all_rounds=${EVAL_ALL_ROUNDS}"
    echo "eval_sparse_15=${EVAL_SPARSE_15}"
    echo "eval_batch_default=${EVAL_BATCH}"
    echo "eval_batch_per_task=C-STANCE:${BATCH_C_STANCE},FOMC:${BATCH_FOMC},MeetingBank:${BATCH_MEETINGBANK},Py150:${BATCH_PY150},ScienceQA:${BATCH_SCIENCEQA},NumGLUE-cm:${BATCH_NUMGLUE_CM},NumGLUE-ds:${BATCH_NUMGLUE_DS},20Minuten:${BATCH_20MINUTEN}"
    echo "temperature=0"
    echo "top_p=1.0"
    echo "num_beams=1"
    echo "max_new_tokens=1024"
    echo "padding_side=left"
  } > "${EVAL_DIR}/eval.env"
fi

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
    task_batch="$(batch_for_task "${task}")"
    task_dir="${EVAL_DIR}/order${round}/${task}"
    infer_file="${task_dir}/infer.jsonl"
    mkdir -p "${task_dir}"

    infer_command=(
      python3 -u -m src.eval.model_diverse_gen_batch
      --model-path "${CHECKPOINT_RUN_DIR}"
      --model-base "${MODEL_PATH}"
      --question-file "${DATA_ROOT}/${task}/test.json"
      --answers-file "${infer_file}"
      --temperature 0
      --conv-mode "${CONV_MODE}"
      --denoised_mode "${DENOISED_MODE}"
      --test_order "${round}"
      --batch_size "${task_batch}"
      --resume
    )
    eval_command=(
      python3 -u -m src.eval.eval_trace
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
    "${infer_command[@]}" 2>&1 | tee "${task_dir}/infer.log"
    python3 - "${DATA_ROOT}/${task}/test.json" "${infer_file}" <<'PY'
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
    "${eval_command[@]}" 2>&1 | tee "${task_dir}/eval.log"
  done
done
