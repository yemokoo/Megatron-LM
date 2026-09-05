#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${TRACE_PYTHON:-${ROOT}/.venv-runtime/bin/python}"
MODEL="${INSTRUCT_MODEL_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}"
CACHE="${INSTRUCT_TOKEN_CACHE:-/data2/seonghyeonnoh/LLM-continual-learning-caches/llama31_8b_instruct/slora_chat_full_len1024}"
RUN_ROOT="${TRACE_RUN_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace}"
TAG="${TRACE_HIDDEN_MSE_TAG:-instruct_hidden_mse_postkd_20260813}"
GPUS="${TRACE_CHAIN_GPUS:-0,1,2,3}"
UPSTREAM_UNIT="${TRACE_UPSTREAM_UNIT:-trace-hidden-mse-long-chain-gpu0123-20260813.service}"
UPSTREAM_STATUS="${TRACE_UPSTREAM_STATUS:-${RUN_ROOT}/logs/${TAG}/status.log}"
SLORA_ROOT="${TRACE_SLORA_POST_ROOT:-${RUN_ROOT}/slora/${TAG}}"
LOG_ROOT="${TRACE_SLORA_POST_LOG_ROOT:-${RUN_ROOT}/logs/${TAG}/slora_post_followup}"
STATUS_LOG="${LOG_ROOT}/status.log"

mkdir -p "${LOG_ROOT}" "${SLORA_ROOT}"

status() {
  echo "$*" | tee -a "${STATUS_LOG}"
}

status "[WAIT START] $(date --iso-8601=seconds) upstream=${UPSTREAM_UNIT} gpus=${GPUS}"
while systemctl --user is-active --quiet "${UPSTREAM_UNIT}"; do
  status "[WAIT] $(date --iso-8601=seconds) upstream chain still active"
  sleep 60
done

if ! grep -q '^\[CHAIN COMPLETE\]' "${UPSTREAM_STATUS}"; then
  status "[ERROR] $(date --iso-8601=seconds) upstream stopped without CHAIN COMPLETE: ${UPSTREAM_STATUS}"
  exit 3
fi

# SLoRA-Post has no independent training trajectory. It post-denoises freshly
# trained Seq-LoRA raw adapters, so train Seq-LoRA first and then register the
# post run as reusing those exact checkpoints.
SEQ_FINAL="${SLORA_ROOT}/llama31/seq/order8/adapter_model.safetensors"
if [[ ! -s "${SEQ_FINAL}" ]]; then
  status "[1/3] $(date --iso-8601=seconds) Seq-LoRA raw adapters retrain for SLoRA-Post"
  CUDA_VISIBLE_DEVICES="${GPUS}" \
  WORLD_SIZE=4 MICRO_BATCH=8 GRAD_ACCUM=2 \
  SLORA_OUTPUT_ROOT="${SLORA_ROOT}" \
  SLORA_LLAMA31_PATH="${MODEL}" \
  SLORA_LLAMA31_TOKEN_CACHE="${CACHE}" \
  PYTHONNOUSERSITE=1 WANDB_MODE=offline \
    bash "${ROOT}/scripts/baselines/llama31/seq_lora.sh" train \
      2>&1 | tee -a "${LOG_ROOT}/01_seq_for_post_train.log"
else
  status "[1/3 SKIP] $(date --iso-8601=seconds) completed Seq-LoRA raw adapters: ${SEQ_FINAL}"
fi

status "[2/3] $(date --iso-8601=seconds) register SLoRA-Post checkpoint reuse"
CUDA_VISIBLE_DEVICES="${GPUS}" \
WORLD_SIZE=4 MICRO_BATCH=8 GRAD_ACCUM=2 \
SLORA_OUTPUT_ROOT="${SLORA_ROOT}" \
SLORA_LLAMA31_PATH="${MODEL}" \
SLORA_LLAMA31_TOKEN_CACHE="${CACHE}" \
PYTHONNOUSERSITE=1 WANDB_MODE=offline \
  bash "${ROOT}/scripts/baselines/llama31/slora_post.sh" train \
    2>&1 | tee -a "${LOG_ROOT}/02_slora_post_prepare.log"

status "[3/3] $(date --iso-8601=seconds) SLoRA-Post sparse-15 evaluation (4 GPU queues)"
pids=()
IFS=',' read -r -a gpu_list <<< "${GPUS}"
for index in 0 1 2 3; do
  gpu="${gpu_list[$index]}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  SLORA_OUTPUT_ROOT="${SLORA_ROOT}" \
  SLORA_LLAMA31_PATH="${MODEL}" \
  EVAL_ALL_ROUNDS=1 EVAL_SPARSE_15=1 \
  EVAL_SHARD_COUNT=4 EVAL_SHARD_INDEX="${index}" \
  SLORA_EVAL_BATCH=4 PYTHONNOUSERSITE=1 \
    bash "${ROOT}/implementations/SLoRA-repro/scripts/repro/eval_trace.sh" \
      post llama31 >"${LOG_ROOT}/03_slora_post_eval_shard${index}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for index in 0 1 2 3; do
  if ! wait "${pids[$index]}"; then
    status "[ERROR] SLoRA-Post eval shard ${index} failed: ${LOG_ROOT}/03_slora_post_eval_shard${index}.log"
    failed=1
  fi
done
[[ "${failed}" == 0 ]] || exit 4

"${PYTHON}" "${ROOT}/scripts/collect_results.py" \
  --method slora_post --model llama31 --sparse-15 \
  --run-dir "${SLORA_ROOT}/llama31/post" --family slora \
  --output "${SLORA_ROOT}/llama31/post/sparse15_summary.json" \
  2>&1 | tee -a "${LOG_ROOT}/03_slora_post_eval.log"

status "[SLoRA-POST CHAIN COMPLETE] $(date --iso-8601=seconds)"
