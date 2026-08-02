#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD="${1:-slora_pre_released}"
ROUND="${2:-8}"
TASK="${3:-Py150}"
case "${METHOD}" in
  slora_pre_released)
    IMPL="${ROOT}/implementations/SLoRA-upstream-port"
    CHECKPOINT_RUN="${ROOT}/results/full_runs_upstream_code/llama31/pre"
    RESULT_RUN="${CHECKPOINT_RUN}"
    DENOISED_MODE="slora_pre" ;;
  slora_post)
    IMPL="${ROOT}/implementations/SLoRA-repro"
    CHECKPOINT_RUN="${ROOT}/results/full_runs/llama31/seq"
    RESULT_RUN="${ROOT}/results/full_runs/llama31/post"
    DENOISED_MODE="slora_post" ;;
  seq_lora)
    IMPL="${ROOT}/implementations/SLoRA-repro"
    CHECKPOINT_RUN="${ROOT}/results/full_runs/llama31/seq"
    RESULT_RUN="${CHECKPOINT_RUN}"
    DENOISED_MODE="seq_lora" ;;
  *) echo "[ERROR] unsupported method: ${METHOD}" >&2; exit 2 ;;
esac
TASK_DIR="${RESULT_RUN}/evaluation/order${ROUND}/${TASK}"
DATA="${TRACE_DATA_ROOT:-${ROOT}/data/trace}/${TASK}/test.json"
MODEL="${ROOT}/models/Llama-3.1-8B-Instruct"
PYTHON="${ROOT}/.venv-runtime/bin/python"
if [[ "${TASK}" == "MeetingBank" ]]; then DEFAULT_TASK_BATCH=4; else DEFAULT_TASK_BATCH=8; fi
TASK_BATCH="${4:-${DEFAULT_TASK_BATCH}}"
GLOBAL="${TASK_DIR}/infer.jsonl"
REMAINING_DATA="${TASK_DIR}/remaining.test.json"

export PYTHONPATH="${IMPL}"
export PATH="${ROOT}/.venv-runtime/bin:${PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${PYTHON}" "${ROOT}/scripts/py150_remaining_parts.py" prepare \
  --global-file "${GLOBAL}" --task-dir "${TASK_DIR}" \
  --source "${DATA}" --remaining "${REMAINING_DATA}"

pids=()
for gpu in 0 1 2 3; do
  (
    cd "${IMPL}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m src.eval.model_diverse_gen_batch \
      --model-path "${CHECKPOINT_RUN}" \
      --model-base "${MODEL}" \
      --question-file "${REMAINING_DATA}" \
      --answers-file "${TASK_DIR}/infer.shard${gpu}.jsonl" \
      --temperature 0 \
      --conv-mode llama3 \
      --denoised_mode "${DENOISED_MODE}" \
      --test_order "${ROUND}" \
      --batch_size "${TASK_BATCH}" \
      --num_chunks 4 \
      --chunk_idx "${gpu}" \
      --resume \
      > "${TASK_DIR}/infer.shard${gpu}.log" 2>&1
  ) &
  pids+=("$!")
  echo "[${TASK} SHARD START] gpu=${gpu} pid=$!"
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
[[ "${status}" -eq 0 ]] || {
  echo "[ERROR] one or more Py150 shards failed" >&2
  exit 1
}

"${PYTHON}" "${ROOT}/scripts/py150_remaining_parts.py" merge \
  --global-file "${GLOBAL}" --task-dir "${TASK_DIR}" \
  --source "${DATA}" --remaining "${REMAINING_DATA}"

cd "${IMPL}"
"${PYTHON}" -u -m src.eval.eval_trace \
  --input_file "${GLOBAL}" \
  --output_file "${TASK_DIR}/wrong.jsonl" \
  2>&1 | tee "${TASK_DIR}/eval.log"
