#!/usr/bin/env bash
set -euo pipefail

# Do not allow host user packages to shadow the project virtualenv.
unset PYTHONPATH

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}"
SCAFFOLD="$(cd "${ROOT}/../.." && pwd)"
METHOD="${1:?usage: train_trace.sh <seq|pre|post> <llama31|qwen25_7b>}"
MODEL_KEY="${2:?usage: train_trace.sh <seq|pre|post> <llama31|qwen25_7b>}"
DATA_ROOT="${TRACE_DATA_ROOT:-${SCAFFOLD}/data/trace}"
OUTPUT_ROOT="${SLORA_OUTPUT_ROOT:-${SCAFFOLD}/results/full_runs}"
DRY_RUN="${DRY_RUN:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"
MICRO_BATCH="${MICRO_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
START_ORDER="${START_ORDER:-1}"
END_ORDER="${END_ORDER:-8}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
export CUDA_VISIBLE_DEVICES

case "${MODEL_KEY}" in
  llama31)
    MODEL_PATH="${SLORA_LLAMA31_PATH:-${SCAFFOLD}/models/Llama-3.1-8B-Instruct}"
    MODEL_FAMILY="llama3"
    PREFLIGHT_KEY="llama31_8b_instruct"
    ;;
  qwen25_7b)
    MODEL_PATH="${SLORA_QWEN25_7B_PATH:-${SCAFFOLD}/models/Qwen2.5-7B-Instruct}"
    MODEL_FAMILY="qwen"
    PREFLIGHT_KEY="qwen25_7b_instruct"
    ;;
  *)
    echo "Unknown model key: ${MODEL_KEY}" >&2
    exit 2
    ;;
esac

case "${METHOD}" in
  seq|post)
    TRAIN_MODULE="src/train/cl_train.py"
    ;;
  pre)
    TRAIN_MODULE="src/train/cl_train_slora.py"
    ;;
  *)
    echo "Unknown method: ${METHOD}" >&2
    exit 2
    ;;
esac

if [[ "${DRY_RUN}" != "1" ]]; then
  python3 "${ROOT}/../../scripts/preflight.py" --mode full --models "${PREFLIGHT_KEY}"
fi

TASKS=(C-STANCE FOMC MeetingBank Py150 ScienceQA NumGLUE-cm NumGLUE-ds 20Minuten)
EPOCHS=(5 3 7 5 3 5 5 7)
RUN_DIR="${OUTPUT_ROOT}/${MODEL_KEY}/${METHOD}"
mkdir -p "${RUN_DIR}"

# SLoRA-Post uses the same raw continual adapters as Seq-LoRA. Only the
# evaluation-time denoising differs, so never repeat the eight-task training.
if [[ "${METHOD}" == "post" ]]; then
  SEQ_RUN_DIR="${OUTPUT_ROOT}/${MODEL_KEY}/seq"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[REUSE] SLoRA-Post training reuses Seq-LoRA checkpoints: ${SEQ_RUN_DIR}"
    exit 0
  fi
  if [[ ! -f "${SEQ_RUN_DIR}/order8/adapter_config.json" ]]; then
    echo "[ERROR] Train Seq-LoRA first; missing: ${SEQ_RUN_DIR}/order8" >&2
    exit 3
  fi
  printf "checkpoint_source=%s\nreason=post_uses_seq_raw_adapters\n" "${SEQ_RUN_DIR}" > "${RUN_DIR}/reuse.env"
  echo "[REUSE] SLoRA-Post training reuses completed Seq-LoRA checkpoints: ${SEQ_RUN_DIR}"
  exit 0
fi

cd "${ROOT}"

for index in "${!TASKS[@]}"; do
  task="${TASKS[$index]}"
  order="$((index + 1))"
  epochs="${EPOCHS[$index]}"
  train_json="${DATA_ROOT}/${task}/train.json"
  task_output="${RUN_DIR}/order${order}"

  if (( order < START_ORDER || order > END_ORDER )); then
    continue
  fi
  completion_file="${task_output}/adapter_config.json"
  if [[ "${METHOD}" == "pre" ]]; then
    completion_file="${task_output}/max.safetensors"
  fi
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${task_output}/adapter_config.json" && -f "${completion_file}" ]]; then
    echo "[SKIP] completed task order${order}: ${task_output}"
    continue
  fi

  python "${ROOT}/../../scripts/run_contract.py" \
    --train-json "${train_json}" --task "${task}" --expected-samples 5000 \
    --micro-batch "${MICRO_BATCH}" --world-size "${WORLD_SIZE}" \
    --gradient-accumulation "${GRAD_ACCUM}" --epochs "${epochs}" --logging-steps 1 \
    | tee "${RUN_DIR}/order${order}.contract.json"

  command=(
    torchrun
    "--nproc_per_node=${WORLD_SIZE}"
    "${TRAIN_MODULE}"
    --bf16 True
    --use_peft True
    --lora_r 64
    --lora_alpha 128
    --deepspeed "${ROOT}/scripts/zero2.json"
    --model_name_or_path "${MODEL_PATH}"
    --model "${MODEL_FAMILY}"
    --dataset_name "${task}"
    --train_data_path "${train_json}"
    --output_dir "${task_output}"
    --num_train_epochs "${epochs}"
    --per_device_train_batch_size "${MICRO_BATCH}"
    --per_device_eval_batch_size 4
    --gradient_accumulation_steps "${GRAD_ACCUM}"
    --eval_strategy no
    --save_strategy steps
    --save_steps 100
    --save_total_limit 1
    --learning_rate 2e-4
    --weight_decay 0
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
    --logging_steps 1
    --gradient_checkpointing True
    --seed 2025
    --task_id "${order}"
  )
  if [[ "${METHOD}" == "pre" ]]; then
    command+=(--mode max)
  fi
  printf '%q ' "${command[@]}" | tee "${RUN_DIR}/order${order}.command.txt"
  printf '\n' | tee -a "${RUN_DIR}/order${order}.command.txt"
  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi
  "${command[@]}" 2>&1 | tee "${RUN_DIR}/order${order}.train.log"
done
