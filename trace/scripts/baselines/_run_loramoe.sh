#!/usr/bin/env bash
set -euo pipefail

unset PYTHONPATH

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LORAMOE_ROOT="${LORAMOE_ROOT:-${ROOT}/implementations/llmcl_benchmark}"
ACTION="${1:?usage: _run_loramoe.sh <validate|train|eval|all> <llama31|qwen25_7b>}"
MODEL="${2:?usage: _run_loramoe.sh <validate|train|eval|all> <llama31|qwen25_7b>}"
PYTHON_BIN="${LORAMOE_PYTHON:-${TRACE_PYTHON:-${ROOT}/.venv-runtime/bin/python}}"
DATA_ROOT="${TRACE_DATA_ROOT:-${ROOT}/data/trace}"
RUN_DIR="${LORAMOE_OUTPUT_ROOT:-${ROOT}/results/full_runs}/${MODEL}/loramoe"
GPUS="${LORAMOE_GPUS:-0,1,2,3}"
EPOCHS="${LORAMOE_EPOCHS:-5,3,7,5,3,5,5,7}"
MICRO_BATCH="${LORAMOE_MICRO_BATCH:-2}"
GRAD_ACCUM="${LORAMOE_GRAD_ACCUM:-8}"
RANK="${LORAMOE_RANK:-64}"
ALPHA="${LORAMOE_ALPHA:-128}"
NUM_EXPERTS="${LORAMOE_NUM_EXPERTS:-8}"
TOP_K="${LORAMOE_TOP_K:-1}"
PORT="${LORAMOE_PORT:-29631}"
EVAL_BATCH="${LORAMOE_EVAL_BATCH:-8}"
DRY_RUN="${DRY_RUN:-0}"
TASKS=(C-STANCE FOMC MeetingBank Py150 ScienceQA NumGLUE-cm NumGLUE-ds 20Minuten)
TASK_CSV="$(IFS=,; echo "${TASKS[*]}")"

case "${MODEL}" in
  llama31)
    MODEL_PATH="${SLORA_LLAMA31_PATH:-${ROOT}/models/Llama-3.1-8B-Instruct}"
    PREFLIGHT_KEY="llama31_8b_instruct"
    ;;
  qwen25_7b)
    MODEL_PATH="${SLORA_QWEN25_7B_PATH:-${ROOT}/models/Qwen2.5-7B-Instruct}"
    PREFLIGHT_KEY="qwen25_7b_instruct"
    ;;
  *)
    echo "[ERROR] Unknown model: ${MODEL}" >&2
    exit 2
    ;;
esac

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
WORLD_SIZE="${#GPU_ARRAY[@]}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_COMMAND=(
  "${PYTHON_BIN}" -m torch.distributed.run
  "--nproc_per_node=${WORLD_SIZE}"
  "--master_port=${PORT}"
  training/main_paper_baselines.py
  --method loramoe
  --model_name_or_path "${MODEL_PATH}"
  --data_path "${DATA_ROOT}"
  --dataset_name all
  --data_output_path "${RUN_DIR}/data_cache"
  --output_dir "${RUN_DIR}"
  --num_train_epochs "${EPOCHS}"
  --per_device_train_batch_size "${MICRO_BATCH}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --per_device_eval_batch_size 8
  --max_prompt_len 1024
  --max_ans_len 512
  --learning_rate 2e-4
  --weight_decay 0
  --lr_scheduler_type cosine
  --num_warmup_steps 0
  --gradient_checkpointing_tasks MeetingBank,Py150,ScienceQA,20Minuten
  --lora_rank "${RANK}"
  --lora_alpha "${ALPHA}"
  --lora_dropout 0
  --loramoe_num_experts "${NUM_EXPERTS}"
  --top_k "${TOP_K}"
  --routing_weight_mode full_softmax
  --moe_aux_loss_coeff 0
  --moe_z_loss_coeff 0
  --seed 2025
)

if [[ -n "${LORAMOE_RESUME_CHECKPOINT:-}" ]]; then
  TRAIN_COMMAND+=(--resume_checkpoint "${LORAMOE_RESUME_CHECKPOINT}")
fi

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

preflight() {
  [[ -f "${LORAMOE_ROOT}/training/main_paper_baselines.py" ]] || {
    echo "[ERROR] LoRAMoE implementation missing: ${LORAMOE_ROOT}" >&2
    exit 3
  }
  [[ -x "${PYTHON_BIN}" ]] || {
    echo "[ERROR] LoRAMoE venv missing: ${PYTHON_BIN}" >&2
    exit 4
  }
  python3 "${ROOT}/scripts/preflight.py" --mode full --models "${PREFLIGHT_KEY}"
}

run_train() {
  mkdir -p "${RUN_DIR}"
  {
    echo "classification=local-compatible-port-not-slora-author-code"
    echo "model=${MODEL}"
    echo "model_path=${MODEL_PATH}"
    echo "data_root=${DATA_ROOT}"
    echo "task_order=${TASK_CSV}"
    echo "epochs=${EPOCHS}"
    echo "world_size=${WORLD_SIZE}"
    echo "micro_batch=${MICRO_BATCH}"
    echo "gradient_accumulation=${GRAD_ACCUM}"
    echo "rank=${RANK}"
    echo "alpha=${ALPHA}"
    echo "num_experts=${NUM_EXPERTS}"
    echo "top_k=${TOP_K}"
  } > "${RUN_DIR}/run.env"
  print_command "${TRAIN_COMMAND[@]}" > "${RUN_DIR}/train.command.txt"
  if [[ "${DRY_RUN}" == "1" ]]; then
    print_command "${TRAIN_COMMAND[@]}"
    return
  fi
  preflight
  cd "${LORAMOE_ROOT}"
  "${TRAIN_COMMAND[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
}

run_eval() {
  local round task_csv eval_dir
  if [[ "${DRY_RUN}" != "1" ]]; then
    preflight
  fi
  for ((round=1; round<=8; round++)); do
    task_csv="$(IFS=,; echo "${TASKS[*]:0:${round}}")"
    eval_dir="${RUN_DIR}/evaluation/order${round}"
    EVAL_COMMAND=(
      "${PYTHON_BIN}" evaluate_Ours_LoRA_MoE.py
      --checkpoint_dir "${RUN_DIR}/$((round - 1))"
      --base_model_name_or_path "${MODEL_PATH}"
      --data_path "${DATA_ROOT}"
      --inference_tasks "${task_csv}"
      --inference_output_path "${eval_dir}"
      --max_prompt_len 1024
      --max_ans_len 512
      --per_device_eval_batch_size "${EVAL_BATCH}"
      --temperature 0
    )
    if [[ "${DRY_RUN}" == "1" ]]; then
      print_command "${EVAL_COMMAND[@]}"
      continue
    fi
    [[ -f "${RUN_DIR}/$((round - 1))/paper_baseline_meta.json" ]] || {
      echo "[ERROR] Missing LoRAMoE checkpoint: ${RUN_DIR}/$((round - 1))" >&2
      exit 3
    }
    mkdir -p "${eval_dir}"
    cd "${LORAMOE_ROOT}"
    "${EVAL_COMMAND[@]}" 2>&1 | tee "${eval_dir}/eval.log"
  done
}

run_collect() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    print_command python3 "${ROOT}/scripts/collect_results.py" --method loramoe --model "${MODEL}"
  else
    python3 "${ROOT}/scripts/collect_results.py" --method loramoe --model "${MODEL}"
  fi
}

case "${ACTION}" in
  validate)
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
