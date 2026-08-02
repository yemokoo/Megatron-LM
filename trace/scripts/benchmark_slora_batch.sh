#!/usr/bin/env bash
set -euo pipefail

unset PYTHONPATH
unset BNB_CUDA_VERSION

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="${ROOT}/.venv-runtime/bin"
[[ -x "${VENV_BIN}/torchrun" ]] || { echo "[ERROR] Runtime is missing: ${VENV_BIN}" >&2; exit 4; }
export PATH="${VENV_BIN}:${PATH}"
SLORA_ROOT="${ROOT}/implementations/SLoRA-repro"
MODEL_KEY="${1:?usage: benchmark_slora_batch.sh <llama31|qwen25_7b> <micro_batch> <grad_accum> [max_steps]}"
MICRO_BATCH="${2:?usage: benchmark_slora_batch.sh <llama31|qwen25_7b> <micro_batch> <grad_accum> [max_steps]}"
GRAD_ACCUM="${3:?usage: benchmark_slora_batch.sh <llama31|qwen25_7b> <micro_batch> <grad_accum> [max_steps]}"
MAX_STEPS="${4:-3}"
WORLD_SIZE="${WORLD_SIZE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${SLORA_ROOT}"

case "${MODEL_KEY}" in
  llama31)
    MODEL_PATH="${ROOT}/models/Llama-3.1-8B-Instruct"
    MODEL_FAMILY="llama3"
    PREFLIGHT_KEY="llama31_8b_instruct"
    ;;
  qwen25_7b)
    MODEL_PATH="${ROOT}/models/Qwen2.5-7B-Instruct"
    MODEL_FAMILY="qwen"
    PREFLIGHT_KEY="qwen25_7b_instruct"
    ;;
  *)
    echo "[ERROR] Unknown model key: ${MODEL_KEY}" >&2
    exit 2
    ;;
esac

DATA_ROOT="${TRACE_DATA_ROOT:-${ROOT}/data/trace}"
TRAIN_JSON="${DATA_ROOT}/MeetingBank/train.json"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${ROOT}/results/memory_smoke/${MODEL_KEY}/mb${MICRO_BATCH}-ga${GRAD_ACCUM}-${RUN_STAMP}"
GPU_LOG="${RUN_DIR}/gpu-memory.csv"
mkdir -p "${RUN_DIR}"

effective_batch=$((MICRO_BATCH * WORLD_SIZE * GRAD_ACCUM))
echo "[SMOKE] task=MeetingBank model=${MODEL_KEY} micro_batch=${MICRO_BATCH} grad_accum=${GRAD_ACCUM} world_size=${WORLD_SIZE} effective_batch=${effective_batch} max_steps=${MAX_STEPS}"

"${ROOT}/.venv-runtime/bin/python" "${ROOT}/scripts/verify_model_files.py" "${MODEL_PATH}"
"${ROOT}/.venv-runtime/bin/python" "${ROOT}/scripts/preflight.py" \
  --mode full --models "${PREFLIGHT_KEY}"

nvidia-smi \
  --query-gpu=index,memory.used \
  --format=csv,noheader,nounits \
  --loop-ms=200 \
  --filename="${GPU_LOG}" &
monitor_pid=$!
cleanup_monitor() {
  if kill -0 "${monitor_pid}" 2>/dev/null; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
}
trap cleanup_monitor EXIT

cd "${SLORA_ROOT}"
set +e
torchrun \
  "--nproc_per_node=${WORLD_SIZE}" \
  src/train/cl_train.py \
  --bf16 True \
  --use_peft True \
  --lora_r 64 \
  --lora_alpha 128 \
  --deepspeed scripts/zero2.json \
  --model_name_or_path "${MODEL_PATH}" \
  --model "${MODEL_FAMILY}" \
  --dataset_name MeetingBank \
  --train_data_path "${TRAIN_JSON}" \
  --output_dir "${RUN_DIR}/checkpoint" \
  --num_train_epochs 1 \
  --max_steps "${MAX_STEPS}" \
  --max_length 1024 \
  --per_device_train_batch_size "${MICRO_BATCH}" \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --eval_strategy no \
  --save_strategy no \
  --learning_rate 2e-4 \
  --weight_decay 0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --gradient_checkpointing True \
  --seed 2025 \
  --task_id 1 \
  --report_to none \
  2>&1 | tee "${RUN_DIR}/train.log"
train_status=${PIPESTATUS[0]}
set -e

cleanup_monitor
trap - EXIT

awk -F, '
  {
    gpu=$1
    used=$2
    gsub(/^[ \t]+|[ \t]+$/, "", gpu)
    gsub(/^[ \t]+|[ \t]+$/, "", used)
    if (used + 0 > peak[gpu]) peak[gpu] = used + 0
  }
  END {
    for (gpu in peak) printf "gpu=%s peak_used_mib=%d\n", gpu, peak[gpu]
  }
' "${GPU_LOG}" | sort | tee "${RUN_DIR}/peak-memory.txt"

echo "run_dir=${RUN_DIR}"
exit "${train_status}"
