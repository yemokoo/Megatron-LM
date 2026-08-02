#!/usr/bin/env bash
set -euo pipefail

unset PYTHONPATH

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OURS_ROOT="${OURS_LORAMOE_ROOT:-${ROOT}/implementations/llmcl_benchmark}"
ACTION="${1:?usage: _run_ours_lora_moe.sh <validate|train|eval|all> <llama31|qwen25_7b> <v1|v2|v2_5>}"
MODEL="${2:?usage: _run_ours_lora_moe.sh <validate|train|eval|all> <llama31|qwen25_7b> <v1|v2|v2_5>}"
VERSION="${3:?usage: _run_ours_lora_moe.sh <validate|train|eval|all> <llama31|qwen25_7b> <v1|v2|v2_5>}"
METHOD="ours_lora_moe_${VERSION}"
REPLAY_MANIFEST="${OURS_REPLAY_MANIFEST:-${ROOT}/manifests/replay/trace_seed2025_random50_per_task.json}"
LLAMA31_TOKEN_CACHE="${OURS_LLAMA31_TOKEN_CACHE:-${ROOT}/cache/tokenized/llama31_8b/slora_chat_full_len1024}"

PYTHON_BIN="${OURS_LORAMOE_PYTHON:-${TRACE_PYTHON:-${ROOT}/.venv-runtime/bin/python}}"
DATA_ROOT="${TRACE_DATA_ROOT:-${ROOT}/data/trace}"
RUN_DIR="${OURS_LORAMOE_OUTPUT_ROOT:-${ROOT}/results/full_runs/${MODEL}/${METHOD}}"
GPUS="${OURS_LORAMOE_GPUS:-0,1,2,3}"
EPOCHS="${OURS_LORAMOE_EPOCHS:-5,3,7,5,3,5,5,7}"
if [[ "${VERSION}" == "v2" || "${VERSION}" == "v2_5" ]]; then
  MICRO_BATCH="${OURS_LORAMOE_MICRO_BATCH:-16}"
  GRAD_ACCUM="${OURS_LORAMOE_GRAD_ACCUM:-1}"
else
  MICRO_BATCH="${OURS_LORAMOE_MICRO_BATCH:-8}"
  GRAD_ACCUM="${OURS_LORAMOE_GRAD_ACCUM:-2}"
fi
RANK="${OURS_LORAMOE_RANK:-64}"
ALPHA="${OURS_LORAMOE_ALPHA:-128}"
EXPERTS_PER_TASK="${OURS_LORAMOE_EXPERTS_PER_TASK:-1}"
TOP_K="${OURS_LORAMOE_TOP_K:-1}"
LEARNING_RATE="${OURS_LORAMOE_LR:-2e-4}"
PORT="${OURS_LORAMOE_PORT:-29641}"
EVAL_BATCH="${OURS_LORAMOE_EVAL_BATCH:-4}"
EVAL_SPARSE_15="${OURS_EVAL_SPARSE_15:-${EVAL_SPARSE_15:-0}}"
EVAL_SHARD_COUNT="${EVAL_SHARD_COUNT:-1}"
EVAL_SHARD_INDEX="${EVAL_SHARD_INDEX:-0}"
DRY_RUN="${DRY_RUN:-0}"
TASKS=(C-STANCE FOMC MeetingBank Py150 ScienceQA NumGLUE-cm NumGLUE-ds 20Minuten)
TASK_CSV="$(IFS=,; echo "${TASKS[*]}")"

case "${VERSION}" in v1|v2|v2_5) ;; *) echo "[ERROR] version must be v1, v2, or v2_5" >&2; exit 2 ;; esac
if [[ "${VERSION}" == "v2_5" ]]; then
  AUX_LOSS_COEFF=0
  Z_LOSS_COEFF=0
else
  AUX_LOSS_COEFF="${OURS_LORAMOE_AUX_COEFF:-0.01}"
  Z_LOSS_COEFF="${OURS_LORAMOE_Z_COEFF:-0.001}"
fi
[[ "${EVAL_SPARSE_15}" == "0" || "${EVAL_SPARSE_15}" == "1" ]] || {
  echo "[ERROR] OURS_EVAL_SPARSE_15 must be 0 or 1: ${EVAL_SPARSE_15}" >&2
  exit 2
}
[[ "${EVAL_SHARD_COUNT}" =~ ^[1-9][0-9]*$ ]] || {
  echo "[ERROR] EVAL_SHARD_COUNT must be positive: ${EVAL_SHARD_COUNT}" >&2; exit 2; }
[[ "${EVAL_SHARD_INDEX}" =~ ^[0-9]+$ && "${EVAL_SHARD_INDEX}" -lt "${EVAL_SHARD_COUNT}" ]] || {
  echo "[ERROR] EVAL_SHARD_INDEX must be in [0, EVAL_SHARD_COUNT): ${EVAL_SHARD_INDEX}" >&2; exit 2; }
case "${MODEL}" in
  llama31)
    MODEL_PATH="${SLORA_LLAMA31_PATH:-${ROOT}/models/Llama-3.1-8B-Instruct}"
    PREFLIGHT_KEY="llama31_8b_instruct"
    CONV_MODE="llama3"
    ;;
  qwen25_7b)
    MODEL_PATH="${SLORA_QWEN25_7B_PATH:-${ROOT}/models/Qwen2.5-7B-Instruct}"
    PREFLIGHT_KEY="qwen25_7b_instruct"
    CONV_MODE="qwen"
    ;;
  *) echo "[ERROR] Unknown model: ${MODEL}" >&2; exit 2 ;;
esac

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
WORLD_SIZE="${#GPU_ARRAY[@]}"
IFS="," read -r -a MICRO_BATCH_VALUES <<< "${MICRO_BATCH}"
IFS="," read -r -a GRAD_ACCUM_VALUES <<< "${GRAD_ACCUM}"
if [[ "${#MICRO_BATCH_VALUES[@]}" -eq 1 ]]; then
  MICRO_BATCH_VALUES=("${MICRO_BATCH_VALUES[0]}" "${MICRO_BATCH_VALUES[0]}" "${MICRO_BATCH_VALUES[0]}" "${MICRO_BATCH_VALUES[0]}" "${MICRO_BATCH_VALUES[0]}" "${MICRO_BATCH_VALUES[0]}" "${MICRO_BATCH_VALUES[0]}" "${MICRO_BATCH_VALUES[0]}")
fi
if [[ "${#GRAD_ACCUM_VALUES[@]}" -eq 1 ]]; then
  GRAD_ACCUM_VALUES=("${GRAD_ACCUM_VALUES[0]}" "${GRAD_ACCUM_VALUES[0]}" "${GRAD_ACCUM_VALUES[0]}" "${GRAD_ACCUM_VALUES[0]}" "${GRAD_ACCUM_VALUES[0]}" "${GRAD_ACCUM_VALUES[0]}" "${GRAD_ACCUM_VALUES[0]}" "${GRAD_ACCUM_VALUES[0]}")
fi
[[ "${#MICRO_BATCH_VALUES[@]}" -eq 8 && "${#GRAD_ACCUM_VALUES[@]}" -eq 8 ]] || { echo "[ERROR] batch and accumulation profiles must contain 1 or 8 values" >&2; exit 2; }
EFFECTIVE_GLOBAL_BATCH=$((MICRO_BATCH_VALUES[0] * WORLD_SIZE * GRAD_ACCUM_VALUES[0]))
for index in "${!TASKS[@]}"; do
  task_global=$((MICRO_BATCH_VALUES[index] * WORLD_SIZE * GRAD_ACCUM_VALUES[index]))
  [[ "${task_global}" -eq "${EFFECTIVE_GLOBAL_BATCH}" ]] || { echo "[ERROR] ${TASKS[index]} effective global batch ${task_global} != ${EFFECTIVE_GLOBAL_BATCH}" >&2; exit 2; }
done
export CUDA_VISIBLE_DEVICES="${GPUS}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_COMMAND=(
  "${PYTHON_BIN}" -m torch.distributed.run
  "--nproc_per_node=${WORLD_SIZE}"
  "--master_port=${PORT}"
  training/main_Ours_LoRA_MoE.py
  --training_version "${VERSION}"
  --model_name_or_path "${MODEL_PATH}"
  --data_path "${DATA_ROOT}"
  --dataset_name all
  --data_output_path "${RUN_DIR}/data_cache"
  --output_dir "${RUN_DIR}"
  --num_train_epochs "${EPOCHS}"
  --per_device_train_batch_size "${MICRO_BATCH}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --per_device_eval_batch_size "${EVAL_BATCH}"
  --max_prompt_len 1024
  --max_ans_len 512
  --max_train_len 1024
  --learning_rate "${LEARNING_RATE}"
  --weight_decay 0
  --adam_beta1 0.9
  --adam_beta2 0.999
  --adam_epsilon 1e-8
  --train_format slora_chat_full
  --lr_scheduler_type cosine
  --num_warmup_steps 0
  --warmup_ratio 0.03
  --gradient_checkpointing
  --experts_per_task "${EXPERTS_PER_TASK}"
  --lora_moe_rank "${RANK}"
  --lora_moe_alpha "${ALPHA}"
  --lora_moe_dropout "${OURS_LORAMOE_DROPOUT:-0.05}"
  --top_k "${TOP_K}"
  --routing_weight_mode full_softmax
  --moe_aux_loss_coeff "${AUX_LOSS_COEFF}"
  --moe_z_loss_coeff "${Z_LOSS_COEFF}"
  --seed "${OURS_LORAMOE_SEED:-2025}"
  --replay_subset_ratio "${OURS_REPLAY_SUBSET_RATIO:-0.01}"
  --replay_distribution "${OURS_REPLAY_DISTRIBUTION:-equal_task}"
  --replay_subset_seed "${OURS_REPLAY_SUBSET_SEED:--1}"
  --router_replay_exposure_samples "${OURS_ROUTER_REPLAY_EXPOSURE_SAMPLES:-1000}"
  --replay_manifest_path "${REPLAY_MANIFEST}"
)

if [[ "${VERSION}" == "v1" ]]; then
  TRAIN_COMMAND+=(
    --router_retune_epochs "${OURS_V1_ROUTER_RETUNE_EPOCHS:-1}"
  )
else
  TRAIN_COMMAND+=(
    --router_retune_epochs 0
    --v2_memory_batch_size "${OURS_V2_MEMORY_BATCH_SIZE:-0}"
    --v2_max_replay_batches_per_step "${OURS_V2_MAX_REPLAY_BATCHES_PER_STEP:-0}"
    --v2_joint_replay_loss_coeff "${OURS_V2_REPLAY_LOSS_COEFF:-1.0}"
    --v2_kd_loss_coeff "${OURS_V2_KD_LOSS_COEFF:-1.0}"
    --v2_kd_temperature "${OURS_V2_KD_TEMPERATURE:-1.0}"
    --v2_kd_learning_rate "${OURS_V2_KD_LR:-0}"
    --v2_kd_chunk_tokens "${OURS_V2_KD_CHUNK_TOKENS:-256}"
    --v2_kd_token_scope "${OURS_V2_KD_TOKEN_SCOPE:-nonpad}"
  )
fi
if [[ "${MODEL}" == "llama31" ]]; then
  TRAIN_COMMAND+=(--tokenized_train_cache_dir "${LLAMA31_TOKEN_CACHE}")
fi


if [[ "${OURS_DISABLE_TRAINING_FLOP_COUNTER:-1}" == "1" ]]; then
  TRAIN_COMMAND+=(--disable_training_flop_counter)
fi

if [[ -n "${OURS_LORAMOE_RESUME_CHECKPOINT:-}" ]]; then
  TRAIN_COMMAND+=(--resume_checkpoint "${OURS_LORAMOE_RESUME_CHECKPOINT}")
fi

print_command() { printf '%q ' "$@"; printf '\n'; }

preflight() {
  [[ -f "${OURS_ROOT}/training/main_Ours_LoRA_MoE.py" ]] || {
    echo "[ERROR] Ours implementation missing: ${OURS_ROOT}" >&2; exit 3; }
  [[ -x "${PYTHON_BIN}" ]] || {
    echo "[ERROR] training venv missing: ${PYTHON_BIN}" >&2; exit 4; }
  [[ -f "${REPLAY_MANIFEST}" ]] || {
    echo "[ERROR] replay manifest missing: ${REPLAY_MANIFEST}" >&2; exit 5; }
  if [[ "${MODEL}" == "llama31" ]]; then
    [[ -f "${LLAMA31_TOKEN_CACHE}/manifest.json" ]] || {
      echo "[ERROR] Llama token cache missing: ${LLAMA31_TOKEN_CACHE}" >&2; exit 6; }
  fi
  python3 "${ROOT}/scripts/preflight.py" --mode full --models "${PREFLIGHT_KEY}"
}

write_manifest() {
  mkdir -p "${RUN_DIR}"
  {
    echo "classification=local-ours-lora-moe-${VERSION}"
    echo "training_version=${VERSION}"
    echo "model=${MODEL}"
    echo "model_path=${MODEL_PATH}"
    echo "data_root=${DATA_ROOT}"
    echo "task_order=${TASK_CSV}"
    echo "epochs=${EPOCHS}"
    echo "world_size=${WORLD_SIZE}"
    echo "micro_batch=${MICRO_BATCH}"
    echo "gradient_accumulation=${GRAD_ACCUM}"
    echo "effective_global_batch=${EFFECTIVE_GLOBAL_BATCH}"
    echo "micro_batch_by_task=${MICRO_BATCH}"
    echo "gradient_accumulation_by_task=${GRAD_ACCUM}"
    echo "max_train_len=1024"
    echo "train_format=slora_chat_full"
    echo "label_scope=all_nonpadding_tokens"
    echo "padding_side=right"
    echo "replay_manifest=${REPLAY_MANIFEST}"
    [[ "${MODEL}" == "llama31" ]] && echo "tokenized_train_cache=${LLAMA31_TOKEN_CACHE}"
    echo "adam_betas=0.9,0.999"
    echo "adam_epsilon=1e-8"
    echo "warmup_ratio=0.03"
    echo "lora_dropout=${OURS_LORAMOE_DROPOUT:-0.05}"
    echo "rank=${RANK}"
    echo "alpha=${ALPHA}"
    echo "experts_per_task=${EXPERTS_PER_TASK}"
    echo "replay_subset_ratio=${OURS_REPLAY_SUBSET_RATIO:-0.01}"
    echo "replay_distribution=${OURS_REPLAY_DISTRIBUTION:-equal_task}"
    echo "router_replay_exposure_samples=${OURS_ROUTER_REPLAY_EXPOSURE_SAMPLES:-1000}"
    echo "training_workload_file=${RUN_DIR}/training_workload.json"
    echo "training_flop_counter_disabled=${OURS_DISABLE_TRAINING_FLOP_COUNTER:-1}"
    echo "moe_aux_loss_coeff=${AUX_LOSS_COEFF}"
    echo "moe_z_loss_coeff=${Z_LOSS_COEFF}"
    if [[ "${VERSION}" == "v2" || "${VERSION}" == "v2_5" ]]; then
      echo "gradient_pair=new-data(new-expert+router)+past-data(router-only)"
    fi
  } > "${RUN_DIR}/run.env"
  print_command "${TRAIN_COMMAND[@]}" > "${RUN_DIR}/train.command.txt"
}

run_train() {
  write_manifest
  if [[ "${DRY_RUN}" == "1" ]]; then print_command "${TRAIN_COMMAND[@]}"; return; fi
  preflight
  cd "${OURS_ROOT}"
  "${TRAIN_COMMAND[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
}

run_eval() {
  local round task_index task task_csv eval_dir cell_index=0 assigned_shard
  if [[ "${DRY_RUN}" != "1" ]]; then preflight; fi
  for ((round=1; round<=8; round++)); do
    for ((task_index=0; task_index<round; task_index++)); do
      if [[ "${EVAL_SPARSE_15}" == "1" && "${round}" -lt 8 && "${task_index}" -ne $((round - 1)) ]]; then
        continue
      fi
      assigned_shard=$((cell_index % EVAL_SHARD_COUNT))
      cell_index=$((cell_index + 1))
      [[ "${assigned_shard}" -eq "${EVAL_SHARD_INDEX}" ]] || continue
      task="${TASKS[$task_index]}"
      task_csv="${task}"
      eval_dir="${RUN_DIR}/evaluation/order${round}"
      EVAL_COMMAND=(
        "${PYTHON_BIN}" evaluate_Ours_LoRA_MoE.py
        --checkpoint_dir "${RUN_DIR}/$((round - 1))"
        --base_model_name_or_path "${MODEL_PATH}"
        --data_path "${DATA_ROOT}"
        --inference_tasks "${task_csv}"
        --inference_output_path "${eval_dir}"
        --summary_filename "${task}.summary.json"
        --max_prompt_len 0
        --max_ans_len 1024
        --no-task_generation_limits
        --slora_conv_mode "${CONV_MODE}"
        --per_device_eval_batch_size "${EVAL_BATCH}"
        --temperature 0
      )
      if [[ "${DRY_RUN}" == "1" ]]; then print_command "${EVAL_COMMAND[@]}"; continue; fi
      [[ -f "${RUN_DIR}/$((round - 1))/lora_moe_meta.json" ]] || {
        echo "[ERROR] Missing Ours checkpoint: ${RUN_DIR}/$((round - 1))" >&2; exit 3; }
      mkdir -p "${eval_dir}"
      {
        echo "eval_sparse_15=${EVAL_SPARSE_15}"
        echo "eval_batch=${EVAL_BATCH}"
        echo "inference_tasks=${task_csv}"
        echo "temperature=0"
        echo "max_new_tokens=1024"
      } > "${eval_dir}/${task}.eval.env"
      cd "${OURS_ROOT}"
      "${EVAL_COMMAND[@]}" 2>&1 | tee "${eval_dir}/${task}.eval.log"
    done
  done
}

run_collect() {
  local command=(python3 "${ROOT}/scripts/collect_results.py" --method "${METHOD}" --model "${MODEL}")
  if [[ "${EVAL_SPARSE_15}" == "1" ]]; then command+=(--sparse-15); fi
  if [[ "${DRY_RUN}" == "1" ]]; then print_command "${command[@]}"; else "${command[@]}"; fi
}

case "${ACTION}" in
  validate) DRY_RUN=1 run_train; DRY_RUN=1 run_eval ;;
  train) run_train ;;
  eval) run_eval; [[ "${EVAL_SKIP_COLLECT:-0}" == "1" ]] || run_collect ;;
  all) run_train; run_eval; run_collect ;;
  *) echo "[ERROR] Unknown action: ${ACTION}" >&2; exit 2 ;;
esac
