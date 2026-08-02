#!/usr/bin/env bash
set -euo pipefail

# Do not allow host user packages to shadow the project virtualenv.
unset PYTHONPATH

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}"
SCAFFOLD="$(cd "${ROOT}/../.." && pwd)"

METHOD="${1:?usage: train_trace.sh <ewc|lwf|gem|olora> <upstream|corrected> <llama31|qwen25_7b>}"
VARIANT="${2:?usage: train_trace.sh <ewc|lwf|gem|olora> <upstream|corrected> <llama31|qwen25_7b>}"
MODEL_KEY="${3:?usage: train_trace.sh <ewc|lwf|gem|olora> <upstream|corrected> <llama31|qwen25_7b>}"

DATA_ROOT="${TRACE_DATA_ROOT:-${SCAFFOLD}/data/trace}"
OUTPUT_ROOT="${TRACE_OUTPUT_ROOT:-${SCAFFOLD}/results/full_runs}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
WORLD_SIZE="${WORLD_SIZE:-4}"
DRY_RUN="${DRY_RUN:-0}"
RUN_NAME="${RUN_NAME:-${MODEL_KEY}/${METHOD}_${VARIANT}}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

TASKS=(C-STANCE FOMC MeetingBank Py150 ScienceQA NumGLUE-cm NumGLUE-ds 20Minuten)
EPOCHS=(5 3 7 5 3 5 5 7)
TASK_CSV="$(IFS=,; echo "${TASKS[*]}")"
EPOCH_CSV="$(IFS=,; echo "${EPOCHS[*]}")"

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

case "${VARIANT}" in
  upstream|corrected) ;;
  *)
    echo "[ERROR] Unknown variant: ${VARIANT}" >&2
    exit 2
    ;;
esac

if [[ "${VARIANT}" == "corrected" && "${METHOD}" != "gem" && "${METHOD}" != "olora" ]]; then
  echo "[ERROR] Corrected variants are defined only for GEM and O-LoRA." >&2
  exit 2
fi

case "${MODEL_KEY}" in
  llama31)
    MODEL_PATH="${SLORA_LLAMA31_PATH:-${SCAFFOLD}/models/Llama-3.1-8B-Instruct}"
    PREFLIGHT_KEY="llama31_8b_instruct"
    ;;
  qwen25_7b)
    MODEL_PATH="${SLORA_QWEN25_7B_PATH:-${SCAFFOLD}/models/Qwen2.5-7B-Instruct}"
    PREFLIGHT_KEY="qwen25_7b_instruct"
    ;;
  *)
    echo "[ERROR] Unknown model key: ${MODEL_KEY}" >&2
    exit 2
    ;;
esac

mkdir -p "${RUN_DIR}/contracts" "${RUN_DIR}/data_cache"

for index in "${!TASKS[@]}"; do
  task="${TASKS[$index]}"
  python3 "${SCAFFOLD}/scripts/run_contract.py" \
    --train-json "${DATA_ROOT}/${task}/train.json" \
    --task "${task}" \
    --expected-samples 5000 \
    --micro-batch 2 \
    --world-size "${WORLD_SIZE}" \
    --gradient-accumulation 8 \
    --epochs "${EPOCHS[$index]}" \
    --logging-steps 1 \
    > "${RUN_DIR}/contracts/$((index + 1))-${task}.json"
done

command=(
  deepspeed
  "--include=localhost:${GPU_LIST}"
  training/main.py
  --data_path "${DATA_ROOT}"
  --data_output_path "${RUN_DIR}/data_cache"
  --dataset_name "${TASK_CSV}"
  --model_name_or_path "${MODEL_PATH}"
  --per_device_train_batch_size 2
  --per_device_eval_batch_size 16
  --max_prompt_len 1024
  --max_ans_len 512
  --learning_rate 2e-4
  --weight_decay 0
  --num_train_epochs "${EPOCH_CSV}"
  --gradient_accumulation_steps 8
  --lr_scheduler_type cosine
  --warmup_ratio 0.03
  --seed 2025
  --zero_stage 2
  --deepspeed
  --gradient_checkpointing
  --disable_dropout
  --print_loss
  --CL_method "${TRACE_METHOD}"
  --implementation_variant "${VARIANT}"
  --output_dir "${RUN_DIR}/checkpoints"
)

{
  echo "method=${METHOD}"
  echo "trace_method=${TRACE_METHOD}"
  echo "variant=${VARIANT}"
  echo "model_key=${MODEL_KEY}"
  echo "model_path=${MODEL_PATH}"
  echo "data_root=${DATA_ROOT}"
  echo "world_size=${WORLD_SIZE}"
  echo "gpu_list=${GPU_LIST}"
  echo "seed=2025"
  echo "task_order=${TASK_CSV}"
  echo "epochs=${EPOCH_CSV}"
  printf "command="
  printf "%q " "${command[@]}"
  printf "\n"
} > "${RUN_DIR}/run.env"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf "%q " "${command[@]}"
  printf "\n"
  exit 0
fi

python3 "${SCAFFOLD}/scripts/preflight.py" \
  --mode full \
  --models "${PREFLIGHT_KEY}"

cd "${ROOT}"
"${command[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
