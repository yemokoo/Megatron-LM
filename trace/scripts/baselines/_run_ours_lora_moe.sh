#!/usr/bin/env bash
set -euo pipefail

unset PYTHONPATH

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OURS_ROOT="${OURS_LORAMOE_ROOT:-${ROOT}/implementations/llmcl_benchmark}"
ACTION="${1:?usage: _run_ours_lora_moe.sh <validate|train|eval|all> <llama31|qwen25_7b> <v1|v1_expert_first|v2|v2_new|v2_new_top4|v2_5|v3|v3_new|v3_new_top4|v3_new_replay40|v3_new_hidden_mse_full|v3_new_replay1to1|v3_new_hidden_mse_1to1|v3_new_replay1to1_recency|v3_new_replay1to1_p5k|v3_new_hidden_mse_1to1_p5k|v3_new_kd35k|v3_new_recency_kd175k|v3_new_p5k_kd175k|v3_new_r20_kd100|v3_new_kd200|v3_new_recency_p2|v3_new_hmse_kd200>}"
MODEL="${2:?usage: _run_ours_lora_moe.sh <validate|train|eval|all> <llama31|qwen25_7b> <v1|v1_expert_first|v2|v2_new|v2_new_top4|v2_5|v3|v3_new|v3_new_top4|v3_new_replay40|v3_new_hidden_mse_full|v3_new_replay1to1|v3_new_hidden_mse_1to1|v3_new_replay1to1_recency|v3_new_replay1to1_p5k|v3_new_hidden_mse_1to1_p5k|v3_new_kd35k|v3_new_recency_kd175k|v3_new_p5k_kd175k|v3_new_r20_kd100|v3_new_kd200|v3_new_recency_p2|v3_new_hmse_kd200>}"
VERSION="${3:?usage: _run_ours_lora_moe.sh <validate|train|eval|all> <llama31|qwen25_7b> <v1|v1_expert_first|v2|v2_new|v2_new_top4|v2_5|v3|v3_new|v3_new_top4|v3_new_replay40|v3_new_hidden_mse_full|v3_new_replay1to1|v3_new_hidden_mse_1to1|v3_new_replay1to1_recency|v3_new_replay1to1_p5k|v3_new_hidden_mse_1to1_p5k|v3_new_kd35k|v3_new_recency_kd175k|v3_new_p5k_kd175k|v3_new_r20_kd100|v3_new_kd200|v3_new_recency_p2|v3_new_hmse_kd200>}"
METHOD="ours_lora_moe_${VERSION}"
if [[ "${VERSION}" == "v1_expert_first" || "${VERSION}" == "v2_new" || "${VERSION}" == "v2_new_top4" || "${VERSION}" == "v3_new" || "${VERSION}" == "v3_new_top4" || "${VERSION}" == "v3_new_replay40" || "${VERSION}" == "v3_new_hidden_mse_full" || "${VERSION}" == "v3_new_replay1to1" || "${VERSION}" == "v3_new_hidden_mse_1to1" || "${VERSION}" == "v3_new_replay1to1_recency" || "${VERSION}" == "v3_new_replay1to1_p5k" || "${VERSION}" == "v3_new_hidden_mse_1to1_p5k" || "${VERSION}" == "v3_new_p5k_kd175k" || "${VERSION}" == "v3_new_r20_kd100" || "${VERSION}" == "v3_new_kd200" || "${VERSION}" == "v3_new_recency_p2" || "${VERSION}" == "v3_new_hmse_kd200" || "${VERSION}" == "v3_new_recency_kd175k" || "${VERSION}" == "v3_new_kd35k" ]]; then
  # V2-new intentionally samples and persists its deterministic 500 records
  # inside the run. Supplying OURS_REPLAY_MANIFEST remains an explicit override.
  REPLAY_MANIFEST="${OURS_REPLAY_MANIFEST:-}"
  REPLAY_SUBSET_RATIO="${OURS_REPLAY_SUBSET_RATIO:-0.1}"
else
  REPLAY_MANIFEST="${OURS_REPLAY_MANIFEST:-${ROOT}/manifests/replay/trace_seed2025_random50_per_task.json}"
  REPLAY_SUBSET_RATIO="${OURS_REPLAY_SUBSET_RATIO:-0.01}"
fi
# The old default pointed at a cache built with the BASE tokenizer, whose
# eos is <|end_of_text|>; the collator appends it because <|eot_id|> is not
# the base eos, so every record carried one extra supervised token and all
# 5,000 rows per task differed from what every published baseline trained
# on.  Default to the Instruct cache the baselines actually used.
LLAMA31_TOKEN_CACHE="${OURS_LLAMA31_TOKEN_CACHE:-/data2/seonghyeonnoh/LLM-continual-learning-caches/llama31_8b_instruct/slora_chat_full_len1024}"
if [[ ! -d "${LLAMA31_TOKEN_CACHE}" ]]; then
  echo "[ERROR] tokenized cache not found: ${LLAMA31_TOKEN_CACHE}" >&2
  exit 2
fi

PYTHON_BIN="${OURS_LORAMOE_PYTHON:-${TRACE_PYTHON:-${ROOT}/.venv-runtime/bin/python}}"
DATA_ROOT="${TRACE_DATA_ROOT:-${ROOT}/data/trace}"
case "${VERSION}" in
  v1*) VERSION_GROUP="v1" ;;
  v2*) VERSION_GROUP="v2" ;;
  v3*) VERSION_GROUP="v3" ;;
  *) echo "[ERROR] cannot classify version directory: ${VERSION}" >&2; exit 2 ;;
esac
TRACE_RUN_ROOT="${TRACE_RUN_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace}"
RUN_DIR="${OURS_LORAMOE_OUTPUT_ROOT:-${TRACE_RUN_ROOT}/${VERSION_GROUP}/${MODEL}/${METHOD}}"
GPUS="${OURS_LORAMOE_GPUS:-0,1,2,3}"
EPOCHS="${OURS_LORAMOE_EPOCHS:-5,3,7,5,3,5,5,7}"
if [[ "${VERSION}" == "v1_expert_first" || "${VERSION}" == "v2" || "${VERSION}" == "v2_new" || "${VERSION}" == "v2_new_top4" || "${VERSION}" == "v2_5" ]]; then
  MICRO_BATCH="${OURS_LORAMOE_MICRO_BATCH:-16}"
  GRAD_ACCUM="${OURS_LORAMOE_GRAD_ACCUM:-1}"
else
  MICRO_BATCH="${OURS_LORAMOE_MICRO_BATCH:-8}"
  GRAD_ACCUM="${OURS_LORAMOE_GRAD_ACCUM:-2}"
fi
if [[ "${VERSION}" == "v2_new_top4" || "${VERSION}" == "v3_new_top4" ]]; then
  # Four active rank-16 experts have the same LoRA parameter capacity as one
  # active rank-64 expert. Keep alpha=128: normalized top-4 weights are about
  # 1/4 each, so 1/4 * (128/16) preserves top-1's 128/64 coefficient.
  RANK="${OURS_LORAMOE_RANK:-16}"
  ALPHA="${OURS_LORAMOE_ALPHA:-128}"
  EXPERTS_PER_TASK="${OURS_LORAMOE_EXPERTS_PER_TASK:-4}"
  TOP_K="${OURS_LORAMOE_TOP_K:-4}"
  KD_PASS_MULTIPLIER="${OURS_V2_KD_PASS_MULTIPLIER:-2}"
  KD_MEMORY_BATCH_SIZE="${OURS_V2_KD_MEMORY_BATCH_SIZE:-4}"
else
  RANK="${OURS_LORAMOE_RANK:-64}"
  ALPHA="${OURS_LORAMOE_ALPHA:-128}"
  EXPERTS_PER_TASK="${OURS_LORAMOE_EXPERTS_PER_TASK:-1}"
  TOP_K="${OURS_LORAMOE_TOP_K:-1}"
  KD_PASS_MULTIPLIER="${OURS_V2_KD_PASS_MULTIPLIER:-1}"
  KD_MEMORY_BATCH_SIZE="${OURS_V2_KD_MEMORY_BATCH_SIZE:-0}"
fi
LEARNING_RATE="${OURS_LORAMOE_LR:-2e-4}"
PORT="${OURS_LORAMOE_PORT:-29641}"
EVAL_BATCH="${OURS_LORAMOE_EVAL_BATCH:-4}"
if [[ "${VERSION}" == "v1_expert_first" || "${VERSION}" == "v2_new" || "${VERSION}" == "v2_new_top4" || "${VERSION}" == "v3_new" || "${VERSION}" == "v3_new_top4" || "${VERSION}" == "v3_new_replay40" || "${VERSION}" == "v3_new_hidden_mse_full" || "${VERSION}" == "v3_new_replay1to1" || "${VERSION}" == "v3_new_hidden_mse_1to1" || "${VERSION}" == "v3_new_replay1to1_recency" || "${VERSION}" == "v3_new_replay1to1_p5k" || "${VERSION}" == "v3_new_hidden_mse_1to1_p5k" || "${VERSION}" == "v3_new_kd35k" || "${VERSION}" == "v3_new_recency_kd175k" || "${VERSION}" == "v3_new_p5k_kd175k" || "${VERSION}" == "v3_new_r20_kd100" || "${VERSION}" == "v3_new_kd200" || "${VERSION}" == "v3_new_recency_p2" || "${VERSION}" == "v3_new_hmse_kd200" ]]; then
  ROUTING_WEIGHT_MODE="${OURS_LORAMOE_ROUTING_WEIGHT_MODE:-straight_through_topk}"
else
  ROUTING_WEIGHT_MODE="${OURS_LORAMOE_ROUTING_WEIGHT_MODE:-full_softmax}"
fi
RUN_SEED="${OURS_LORAMOE_SEED:-2025}"
REPLAY_SUBSET_SEED="${OURS_REPLAY_SUBSET_SEED:--1}"
if [[ ("${VERSION}" == "v1_expert_first" || "${VERSION}" == "v2_new" || "${VERSION}" == "v2_new_top4" || "${VERSION}" == "v3_new" || "${VERSION}" == "v3_new_top4" || "${VERSION}" == "v3_new_replay40" || "${VERSION}" == "v3_new_hidden_mse_full" || "${VERSION}" == "v3_new_replay1to1" || "${VERSION}" == "v3_new_hidden_mse_1to1" || "${VERSION}" == "v3_new_replay1to1_recency" || "${VERSION}" == "v3_new_replay1to1_p5k" || "${VERSION}" == "v3_new_hidden_mse_1to1_p5k" || "${VERSION}" == "v3_new_kd35k" || "${VERSION}" == "v3_new_recency_kd175k" || "${VERSION}" == "v3_new_p5k_kd175k" || "${VERSION}" == "v3_new_r20_kd100" || "${VERSION}" == "v3_new_kd200" || "${VERSION}" == "v3_new_recency_p2" || "${VERSION}" == "v3_new_hmse_kd200") && "${REPLAY_SUBSET_SEED}" -lt 0 ]]; then
  EFFECTIVE_REPLAY_SUBSET_SEED="${RUN_SEED}"
else
  EFFECTIVE_REPLAY_SUBSET_SEED="${REPLAY_SUBSET_SEED}"
fi
EVAL_SPARSE_15="${OURS_EVAL_SPARSE_15:-${EVAL_SPARSE_15:-0}}"
EVAL_SHARD_COUNT="${EVAL_SHARD_COUNT:-1}"
EVAL_SHARD_INDEX="${EVAL_SHARD_INDEX:-0}"
DRY_RUN="${DRY_RUN:-0}"
TASKS=(C-STANCE FOMC MeetingBank Py150 ScienceQA NumGLUE-cm NumGLUE-ds 20Minuten)
TASK_CSV="$(IFS=,; echo "${TASKS[*]}")"

case "${VERSION}" in v1|v1_expert_first|v2|v2_new|v2_new_top4|v2_5|v3|v3_new|v3_new_top4|v3_new_replay40|v3_new_hidden_mse_full|v3_new_replay1to1|v3_new_hidden_mse_1to1|v3_new_replay1to1_recency|v3_new_replay1to1_p5k|v3_new_hidden_mse_1to1_p5k|v3_new_kd35k|v3_new_recency_kd175k|v3_new_p5k_kd175k|v3_new_r20_kd100|v3_new_kd200|v3_new_recency_p2|v3_new_hmse_kd200) ;; *) echo "[ERROR] unsupported version: ${VERSION}" >&2; exit 2 ;; esac
if [[ "${VERSION}" == "v2_new_top4" || "${VERSION}" == "v3_new_top4" ]]; then
  [[ "${RANK}" == "16" && "${ALPHA}" == "128" && \
     "${EXPERTS_PER_TASK}" == "4" && "${TOP_K}" == "4" && \
     "${KD_PASS_MULTIPLIER}" == "2" ]] || {
    echo "[ERROR] ${VERSION} requires rank=16 alpha=128 experts_per_task=4 top_k=4 kd_pass_multiplier=2" >&2
    exit 2
  }
fi
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
    # No default.  The silent fallback used to resolve to the BASE
    # Llama-3.1-8B, which has no chat_template, so a run that forgot to
    # export this trained on a different model than every published
    # baseline and only showed up as bad numbers hours later.  Fail loudly.
    MODEL_PATH="${SLORA_LLAMA31_PATH:?export SLORA_LLAMA31_PATH (Llama-3.1-8B-Instruct); there is no default}"
    PREFLIGHT_KEY="llama31_8b"
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
  --routing_weight_mode "${ROUTING_WEIGHT_MODE}"
  --moe_aux_loss_coeff "${AUX_LOSS_COEFF}"
  --moe_z_loss_coeff "${Z_LOSS_COEFF}"
  --seed "${RUN_SEED}"
  --replay_subset_ratio "${REPLAY_SUBSET_RATIO}"
  --replay_distribution "${OURS_REPLAY_DISTRIBUTION:-equal_task}"
  --replay_recency_power "${OURS_REPLAY_RECENCY_POWER:-1.0}"
  --replay_subset_seed "${REPLAY_SUBSET_SEED}"
  --replay_selection_mode "${OURS_REPLAY_SELECTION_MODE:-random}"
  --router_replay_exposure_samples "${OURS_ROUTER_REPLAY_EXPOSURE_SAMPLES:-1000}"
)

if [[ -n "${REPLAY_MANIFEST}" ]]; then
  TRAIN_COMMAND+=(--replay_manifest_path "${REPLAY_MANIFEST}")
fi

if [[ "${VERSION}" == "v1" ]]; then
  TRAIN_COMMAND+=(
    --router_retune_epochs "${OURS_V1_ROUTER_RETUNE_EPOCHS:-1}"
  )
else
  if [[ "${VERSION}" == "v1_expert_first" ]]; then
    ROUTER_RETUNE_EPOCHS="${OURS_V1_ROUTER_RETUNE_EPOCHS:-1}"
    JOINT_NEW_TO_REPLAY_RATIO="${OURS_V2_JOINT_NEW_TO_REPLAY_RATIO:-0}"
  else
    ROUTER_RETUNE_EPOCHS=0
    JOINT_NEW_TO_REPLAY_RATIO="${OURS_V2_JOINT_NEW_TO_REPLAY_RATIO:-5}"
  fi
  TRAIN_COMMAND+=(
    --router_retune_epochs "${ROUTER_RETUNE_EPOCHS}"
    --v2_memory_batch_size "${OURS_V2_MEMORY_BATCH_SIZE:-0}"
    --v2_replay_forward_batch_size "${OURS_V2_REPLAY_FORWARD_BATCH_SIZE:-8}"
    --v2_kd_memory_batch_size "${KD_MEMORY_BATCH_SIZE}"
    --v2_max_replay_batches_per_step "${OURS_V2_MAX_REPLAY_BATCHES_PER_STEP:-0}"
    --v2_joint_replay_loss_coeff "${OURS_V2_REPLAY_LOSS_COEFF:-1.0}"
    --v2_joint_replay_objective \
      "${OURS_V2_JOINT_REPLAY_OBJECTIVE:-lm}"
    --v2_hidden_mse_loss_coeff \
      "${OURS_V2_HIDDEN_MSE_LOSS_COEFF:-1.0}"
    --v2_joint_new_to_replay_ratio "${JOINT_NEW_TO_REPLAY_RATIO}"
    --v2_kd_loss_coeff "${OURS_V2_KD_LOSS_COEFF:-1.0}"
    --v2_kd_pass_multiplier "${KD_PASS_MULTIPLIER}"
    --v2_kd_temperature "${OURS_V2_KD_TEMPERATURE:-1.0}"
    --v2_kd_learning_rate "${OURS_V2_KD_LR:-0}"
    --v2_kd_chunk_tokens "${OURS_V2_KD_CHUNK_TOKENS:-256}"
  --v2_kd_token_scope "${OURS_V2_KD_TOKEN_SCOPE:-nonpad}"
  )
  if [[ "${OURS_V2_ALLOW_MEMORY_BATCH_RESUME_OVERRIDE:-0}" == "1" ]]; then
    TRAIN_COMMAND+=(--allow_v2_memory_batch_resume_override)
  fi
fi
if [[ "${VERSION}" == "v1_expert_first" || "${VERSION}" == "v2_new" || "${VERSION}" == "v2_new_top4" || "${VERSION}" == "v3_new" || "${VERSION}" == "v3_new_top4" || "${VERSION}" == "v3_new_replay40" || "${VERSION}" == "v3_new_hidden_mse_full" || "${VERSION}" == "v3_new_replay1to1" || "${VERSION}" == "v3_new_hidden_mse_1to1" || "${VERSION}" == "v3_new_replay1to1_recency" || "${VERSION}" == "v3_new_replay1to1_p5k" || "${VERSION}" == "v3_new_hidden_mse_1to1_p5k" || "${VERSION}" == "v3_new_kd35k" || "${VERSION}" == "v3_new_recency_kd175k" || "${VERSION}" == "v3_new_p5k_kd175k" || "${VERSION}" == "v3_new_r20_kd100" || "${VERSION}" == "v3_new_kd200" || "${VERSION}" == "v3_new_recency_p2" || "${VERSION}" == "v3_new_hmse_kd200" ]]; then
  TRAIN_COMMAND+=(
    --v2_new_active_memory_cap "${OURS_V2_NEW_ACTIVE_MEMORY_CAP:-1000}"
    --v2_new_persistent_samples_per_task \
      "${OURS_V2_NEW_PERSISTENT_SAMPLES_PER_TASK:-500}"
  )
  if [[ -n "${OURS_V2_NEW_ACTIVE_UNIQUE_CAP:-}" ]]; then
    TRAIN_COMMAND+=(
      --v2_new_active_unique_cap "${OURS_V2_NEW_ACTIVE_UNIQUE_CAP}")
  fi
  if [[ -n "${OURS_V2_KD_EXPOSURE_SAMPLES:-}" ]]; then
    TRAIN_COMMAND+=(
      --v2_kd_exposure_samples "${OURS_V2_KD_EXPOSURE_SAMPLES}")
  fi
  if [[ -n "${OURS_V2_KD_EPOCHS:-}" ]]; then
    TRAIN_COMMAND+=(--v2_kd_epochs "${OURS_V2_KD_EPOCHS}")
  fi
  if [[ -n "${OURS_V2_NEW_EXPERT_QUOTA_SCHEDULE:-}" ]]; then
    TRAIN_COMMAND+=(
      --v2_new_expert_quota_schedule
      "${OURS_V2_NEW_EXPERT_QUOTA_SCHEDULE}")
  fi
  if [[ "${OURS_V2_NEW_EXPERT_AUX_MIX:-0}" != "0" ]]; then
    TRAIN_COMMAND+=(
      --v2_new_expert_aux_mix
      "${OURS_V2_NEW_EXPERT_AUX_MIX}"
      --v2_new_expert_aux_loss_coeff
      "${OURS_V2_NEW_EXPERT_AUX_LOSS_COEFF:-1.0}")
  fi
fi
if [[ "${VERSION}" == "v3" || "${VERSION}" == "v3_new" || "${VERSION}" == "v3_new_top4" || "${VERSION}" == "v3_new_replay40" || "${VERSION}" == "v3_new_hidden_mse_full" || "${VERSION}" == "v3_new_replay1to1" || "${VERSION}" == "v3_new_hidden_mse_1to1" || "${VERSION}" == "v3_new_replay1to1_recency" || "${VERSION}" == "v3_new_replay1to1_p5k" || "${VERSION}" == "v3_new_hidden_mse_1to1_p5k" || "${VERSION}" == "v3_new_kd35k" || "${VERSION}" == "v3_new_recency_kd175k" || "${VERSION}" == "v3_new_p5k_kd175k" || "${VERSION}" == "v3_new_r20_kd100" || "${VERSION}" == "v3_new_kd200" || "${VERSION}" == "v3_new_recency_p2" || "${VERSION}" == "v3_new_hmse_kd200" ]]; then
  TRAIN_COMMAND+=(
    --v3_epoch_probe_samples "${OURS_V3_EPOCH_PROBE_SAMPLES:-64}"
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
if [[ -n "${OURS_LORAMOE_STOP_AFTER_TASK:-}" ]]; then
  TRAIN_COMMAND+=(--stop_after_task "${OURS_LORAMOE_STOP_AFTER_TASK}")
fi
if [[ "${OURS_V2_ACQUISITION_DIAGNOSTIC_INTERVAL:-0}" != "0" ]]; then
  TRAIN_COMMAND+=(
    --v2_acquisition_diagnostic_interval
    "${OURS_V2_ACQUISITION_DIAGNOSTIC_INTERVAL}")
fi

print_command() { printf '%q ' "$@"; printf '\n'; }

set_train_command_arg() {
  local flag="$1" value="$2" index
  for index in "${!TRAIN_COMMAND[@]}"; do
    if [[ "${TRAIN_COMMAND[$index]}" == "${flag}" ]]; then
      TRAIN_COMMAND[$((index + 1))]="${value}"
      return 0
    elif [[ "${TRAIN_COMMAND[$index]}" == "${flag}="* ]]; then
      TRAIN_COMMAND[$index]="${flag}=${value}"
      return 0
    fi
  done
  TRAIN_COMMAND+=("${flag}" "${value}")
}

add_train_command_flag_once() {
  local flag="$1" value
  for value in "${TRAIN_COMMAND[@]}"; do
    [[ "${value}" == "${flag}" ]] && return 0
  done
  TRAIN_COMMAND+=("${flag}")
}

hidden_mse_oom_signature() {
  local log="$1" start_byte="${2:-0}"
  local pattern='CUDA out of memory|torch\.OutOfMemoryError|OutOfMemoryError|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED'
  if (( start_byte > 0 )); then
    grep -Eiq "${pattern}" < <(tail -c "+$((start_byte + 1))" "${log}")
  else
    grep -Eiq "${pattern}" "${log}"
  fi
}

latest_complete_round_checkpoint() {
  local candidate round best_round=-1 best_path=""
  shopt -s nullglob
  for candidate in "${RUN_DIR}"/[0-9]*; do
    [[ -d "${candidate}" ]] || continue
    round="${candidate##*/}"
    [[ "${round}" =~ ^[0-9]+$ ]] || continue
    [[ -s "${candidate}/pytorch_model.bin" && \
       -s "${candidate}/lora_moe_meta.json" ]] || continue
    if (( 10#${round} > best_round )); then
      best_round=$((10#${round}))
      best_path="${candidate}"
    fi
  done
  shopt -u nullglob
  printf '%s' "${best_path}"
}

hidden_mse_fallback_batch_profiles() {
  local index micro accum
  local -a fallback_micro=() fallback_accum=()
  for index in "${!MICRO_BATCH_VALUES[@]}"; do
    micro="${MICRO_BATCH_VALUES[$index]}"
    accum="${GRAD_ACCUM_VALUES[$index]}"
    if (( micro < 2 || micro % 2 != 0 )); then
      echo "[ERROR] cannot halve hidden-MSE micro batch ${micro} while preserving global batch" >&2
      return 1
    fi
    fallback_micro+=("$((micro / 2))")
    fallback_accum+=("$((accum * 2))")
  done
  local IFS=,
  printf '%s;%s' "${fallback_micro[*]}" "${fallback_accum[*]}"
}

preflight() {
  [[ -f "${OURS_ROOT}/training/main_Ours_LoRA_MoE.py" ]] || {
    echo "[ERROR] Ours implementation missing: ${OURS_ROOT}" >&2; exit 3; }
  [[ -x "${PYTHON_BIN}" ]] || {
    echo "[ERROR] training venv missing: ${PYTHON_BIN}" >&2; exit 4; }
  if [[ -n "${REPLAY_MANIFEST}" ]]; then
    [[ -f "${REPLAY_MANIFEST}" ]] || {
      echo "[ERROR] replay manifest missing: ${REPLAY_MANIFEST}" >&2; exit 5; }
  fi
  if [[ "${MODEL}" == "llama31" ]]; then
    [[ -f "${LLAMA31_TOKEN_CACHE}/manifest.json" ]] || {
      echo "[ERROR] Llama token cache missing: ${LLAMA31_TOKEN_CACHE}" >&2; exit 6; }
  fi
  python3 "${ROOT}/scripts/preflight.py" --mode full --models "${PREFLIGHT_KEY}"
}

write_manifest() {
  local manifest_env="${RUN_DIR}/run.env"
  local manifest_command="${RUN_DIR}/train.command.txt"
  if [[ -n "${OURS_LORAMOE_RESUME_CHECKPOINT:-}" ]]; then
    local resume_round="${OURS_LORAMOE_RESUME_CHECKPOINT%/}"
    resume_round="${resume_round##*/}"
    manifest_env="${RUN_DIR}/resume_from_${resume_round}.env"
    manifest_command="${RUN_DIR}/resume_from_${resume_round}.command.txt"
  fi
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
    echo "replay_manifest=${REPLAY_MANIFEST:-internal_deterministic_subset}"
    [[ "${MODEL}" == "llama31" ]] && echo "tokenized_train_cache=${LLAMA31_TOKEN_CACHE}"
    echo "adam_betas=0.9,0.999"
    echo "adam_epsilon=1e-8"
    echo "warmup_ratio=0.03"
    echo "lora_dropout=${OURS_LORAMOE_DROPOUT:-0.05}"
    echo "rank=${RANK}"
    echo "alpha=${ALPHA}"
    echo "experts_per_task=${EXPERTS_PER_TASK}"
    echo "top_k=${TOP_K}"
    echo "routing_weight_mode=${ROUTING_WEIGHT_MODE}"
    echo "replay_subset_ratio=${REPLAY_SUBSET_RATIO}"
    echo "replay_distribution=${OURS_REPLAY_DISTRIBUTION:-equal_task}"
    echo "replay_recency_power=${OURS_REPLAY_RECENCY_POWER:-1.0}"
    echo "replay_selection_mode=${OURS_REPLAY_SELECTION_MODE:-random}"
    echo "router_replay_exposure_samples=${OURS_ROUTER_REPLAY_EXPOSURE_SAMPLES:-1000}"
    echo "training_workload_file=${RUN_DIR}/training_workload.json"
    echo "training_flop_counter_disabled=${OURS_DISABLE_TRAINING_FLOP_COUNTER:-1}"
    echo "moe_aux_loss_coeff=${AUX_LOSS_COEFF}"
    echo "moe_z_loss_coeff=${Z_LOSS_COEFF}"
    if [[ "${VERSION}" == "v2" || "${VERSION}" == "v2_new" || "${VERSION}" == "v2_new_top4" || "${VERSION}" == "v2_5" || "${VERSION}" == "v3" || "${VERSION}" == "v3_new" || "${VERSION}" == "v3_new_top4" || "${VERSION}" == "v3_new_replay40" || "${VERSION}" == "v3_new_hidden_mse_full" || "${VERSION}" == "v3_new_replay1to1" || "${VERSION}" == "v3_new_hidden_mse_1to1" || "${VERSION}" == "v3_new_replay1to1_recency" || "${VERSION}" == "v3_new_replay1to1_p5k" || "${VERSION}" == "v3_new_hidden_mse_1to1_p5k" || "${VERSION}" == "v3_new_kd35k" || "${VERSION}" == "v3_new_recency_kd175k" || "${VERSION}" == "v3_new_p5k_kd175k" || "${VERSION}" == "v3_new_r20_kd100" || "${VERSION}" == "v3_new_kd200" || "${VERSION}" == "v3_new_recency_p2" || "${VERSION}" == "v3_new_hmse_kd200" ]]; then
      echo "gradient_pair=new-data(new-expert+router)+past-data(router-only)"
      if [[ "${VERSION}" == "v2_new" || "${VERSION}" == "v2_new_top4" || "${VERSION}" == "v3_new" || "${VERSION}" == "v3_new_top4" || "${VERSION}" == "v3_new_replay40" || "${VERSION}" == "v3_new_hidden_mse_full" || "${VERSION}" == "v3_new_replay1to1" || "${VERSION}" == "v3_new_hidden_mse_1to1" || "${VERSION}" == "v3_new_replay1to1_recency" || "${VERSION}" == "v3_new_replay1to1_p5k" || "${VERSION}" == "v3_new_hidden_mse_1to1_p5k" || "${VERSION}" == "v3_new_kd35k" || "${VERSION}" == "v3_new_recency_kd175k" || "${VERSION}" == "v3_new_p5k_kd175k" || "${VERSION}" == "v3_new_r20_kd100" || "${VERSION}" == "v3_new_kd200" || "${VERSION}" == "v3_new_recency_p2" || "${VERSION}" == "v3_new_hmse_kd200" ]]; then
        echo "joint_replay_schedule=every_optimizer_update_active_stream_per_primary_epoch"
        echo "active_memory_cap=${OURS_V2_NEW_ACTIVE_MEMORY_CAP:-1000}"
        echo "persistent_samples_per_task=${OURS_V2_NEW_PERSISTENT_SAMPLES_PER_TASK:-500}"
        echo "active_memory_selection=stable_nested_task_prefix"
        echo "kd_base_stream_epochs=match_primary_epochs"
        echo "kd_pass_multiplier=${KD_PASS_MULTIPLIER}"
        echo "kd_unique_active_stream_samples=${OURS_V2_NEW_ACTIVE_MEMORY_CAP:-1000}"
        echo "kd_total_passes=primary_epochs_times_${KD_PASS_MULTIPLIER}"
        echo "replay_subset_seed_effective=${EFFECTIVE_REPLAY_SUBSET_SEED}"
        echo "new_expert_aux_mix=${OURS_V2_NEW_EXPERT_AUX_MIX:-0}"
        echo "new_expert_aux_loss_coeff=${OURS_V2_NEW_EXPERT_AUX_LOSS_COEFF:-1.0}"
        echo "new_expert_aux_route=detached_router_hard_top1_interpolation"
        if [[ "${VERSION}" == "v2_new_top4" || "${VERSION}" == "v3_new_top4" ]]; then
          echo "profile=${VERSION}"
          echo "active_rank_capacity=$((TOP_K * RANK))"
          echo "lora_scaling=8.0"
          echo "uniform_top4_effective_scaling=2.0"
        fi
      else
        echo "joint_replay_schedule=every_optimizer_update_fixed_total_no_epoch_multiplier"
      fi
      echo "joint_replay_reduction=active_sample_mean"
      echo "joint_new_to_replay_sample_ratio=${OURS_V2_JOINT_NEW_TO_REPLAY_RATIO:-5}:1"
      echo "kd_memory_batch_size=${KD_MEMORY_BATCH_SIZE}"
      echo "replay_memory_batch_size=${OURS_V2_MEMORY_BATCH_SIZE:-0}"
      echo "replay_forward_batch_size=${OURS_V2_REPLAY_FORWARD_BATCH_SIZE:-8}"
    fi
    if [[ "${VERSION}" == "v3" || "${VERSION}" == "v3_new" || "${VERSION}" == "v3_new_top4" || "${VERSION}" == "v3_new_replay40" || "${VERSION}" == "v3_new_hidden_mse_full" || "${VERSION}" == "v3_new_replay1to1" || "${VERSION}" == "v3_new_hidden_mse_1to1" || "${VERSION}" == "v3_new_replay1to1_recency" || "${VERSION}" == "v3_new_replay1to1_p5k" || "${VERSION}" == "v3_new_hidden_mse_1to1_p5k" || "${VERSION}" == "v3_new_kd35k" || "${VERSION}" == "v3_new_recency_kd175k" || "${VERSION}" == "v3_new_p5k_kd175k" || "${VERSION}" == "v3_new_r20_kd100" || "${VERSION}" == "v3_new_kd200" || "${VERSION}" == "v3_new_recency_p2" || "${VERSION}" == "v3_new_hmse_kd200" ]]; then
      echo "architecture=shared_router_qkvo_ffn"
      echo "router_position=post_input_layernorm_pre_self_attention"
      echo "attention_targets=q,k,v,o"
      echo "attention_rank=${RANK}"
      echo "epoch_probe_global_samples=${OURS_V3_EPOCH_PROBE_SAMPLES:-64}"
      echo "epoch_probe_output=${RUN_DIR}/epoch_probe.jsonl"
    fi
    echo "joint_replay_objective=${OURS_V2_JOINT_REPLAY_OBJECTIVE:-lm}"
    echo "hidden_mse_loss_coeff=${OURS_V2_HIDDEN_MSE_LOSS_COEFF:-1.0}"
    if [[ "${OURS_V2_JOINT_REPLAY_OBJECTIVE:-lm}" == "hidden_mse" ]]; then
      echo "hidden_mse_teacher=expanded_post_kd_init"
      echo "hidden_mse_targets=all_decoder_layer_outputs"
      echo "hidden_mse_reduction=active_sample_mean_equal_layer_mean"
      echo "hidden_mse_oom_retry=${OURS_HIDDEN_MSE_OOM_RETRY:-1}"
      echo "hidden_mse_oom_retry_policy=last_complete_round_micro_half_accum_double"
      echo "hidden_mse_oom_retry_kd_memory_batch_size=${OURS_HIDDEN_MSE_OOM_RETRY_KD_BATCH_SIZE:-2}"
      echo "hidden_mse_oom_retry_replay_forward_batch_size=${OURS_HIDDEN_MSE_OOM_RETRY_REPLAY_FORWARD_BATCH_SIZE:-2}"
    fi
  } > "${manifest_env}"
  print_command "${TRAIN_COMMAND[@]}" > "${manifest_command}"
}

run_train() {
  write_manifest
  if [[ "${DRY_RUN}" == "1" ]]; then print_command "${TRAIN_COMMAND[@]}"; return; fi
  preflight
  cd "${OURS_ROOT}"
  local initial_log_mode="" initial_log_bytes=0
  [[ -n "${OURS_LORAMOE_RESUME_CHECKPOINT:-}" ]] && initial_log_mode="-a"
  if [[ -n "${initial_log_mode}" && -f "${RUN_DIR}/train.log" ]]; then
    initial_log_bytes="$(stat -c %s "${RUN_DIR}/train.log")"
  fi
  set +e
  if [[ -n "${initial_log_mode}" ]]; then
    "${TRAIN_COMMAND[@]}" 2>&1 | tee -a "${RUN_DIR}/train.log"
  else
    "${TRAIN_COMMAND[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
  fi
  local train_rc="${PIPESTATUS[0]}"
  set -e
  [[ "${train_rc}" -eq 0 ]] && return 0

  if [[ "${OURS_V2_JOINT_REPLAY_OBJECTIVE:-lm}" != "hidden_mse" || \
        "${OURS_HIDDEN_MSE_OOM_RETRY:-1}" != "1" ]]; then
    return "${train_rc}"
  fi
  if ! hidden_mse_oom_signature "${RUN_DIR}/train.log" "${initial_log_bytes}"; then
    return "${train_rc}"
  fi

  local profiles fallback_micro fallback_accum resume_checkpoint retry_port
  profiles="$(hidden_mse_fallback_batch_profiles)" || return "${train_rc}"
  fallback_micro="${profiles%%;*}"
  fallback_accum="${profiles#*;}"
  resume_checkpoint="$(latest_complete_round_checkpoint)"
  retry_port="${OURS_HIDDEN_MSE_OOM_RETRY_PORT:-$((PORT + 100))}"

  set_train_command_arg --per_device_train_batch_size "${fallback_micro}"
  set_train_command_arg --gradient_accumulation_steps "${fallback_accum}"
  set_train_command_arg --v2_kd_memory_batch_size \
    "${OURS_HIDDEN_MSE_OOM_RETRY_KD_BATCH_SIZE:-2}"
  set_train_command_arg --v2_replay_forward_batch_size \
    "${OURS_HIDDEN_MSE_OOM_RETRY_REPLAY_FORWARD_BATCH_SIZE:-2}"
  set_train_command_arg --master_port "${retry_port}"
  if [[ -n "${resume_checkpoint}" ]]; then
    set_train_command_arg --resume_checkpoint "${resume_checkpoint}"
    add_train_command_flag_once --allow_v2_memory_batch_resume_override
  fi

  {
    echo "trigger=cuda_oom"
    echo "initial_exit_code=${train_rc}"
    echo "initial_micro_batch=${MICRO_BATCH}"
    echo "initial_gradient_accumulation=${GRAD_ACCUM}"
    echo "fallback_micro_batch=${fallback_micro}"
    echo "fallback_gradient_accumulation=${fallback_accum}"
    echo "fallback_effective_global_batch=${EFFECTIVE_GLOBAL_BATCH}"
    echo "fallback_kd_memory_batch_size=${OURS_HIDDEN_MSE_OOM_RETRY_KD_BATCH_SIZE:-2}"
    echo "fallback_replay_forward_batch_size=${OURS_HIDDEN_MSE_OOM_RETRY_REPLAY_FORWARD_BATCH_SIZE:-2}"
    echo "resume_checkpoint=${resume_checkpoint:-none_restart_from_task0}"
    echo "retry_port=${retry_port}"
  } > "${RUN_DIR}/oom_retry.env"
  print_command "${TRAIN_COMMAND[@]}" > "${RUN_DIR}/oom_retry.command.txt"
  {
    echo "[OOM RETRY] $(date --iso-8601=seconds) CUDA OOM detected"
    echo "[OOM RETRY] micro=${MICRO_BATCH} grad_accum=${GRAD_ACCUM} -> micro=${fallback_micro} grad_accum=${fallback_accum}"
    echo "[OOM RETRY] kd_batch=${KD_MEMORY_BATCH_SIZE} -> ${OURS_HIDDEN_MSE_OOM_RETRY_KD_BATCH_SIZE:-2}; replay_forward_batch=${OURS_V2_REPLAY_FORWARD_BATCH_SIZE:-8} -> ${OURS_HIDDEN_MSE_OOM_RETRY_REPLAY_FORWARD_BATCH_SIZE:-2}"
    echo "[OOM RETRY] resume=${resume_checkpoint:-task0_restart} port=${retry_port}"
  } | tee -a "${RUN_DIR}/train.log"
  sleep 10
  "${TRAIN_COMMAND[@]}" 2>&1 | tee -a "${RUN_DIR}/train.log"
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
  local command=(python3 "${ROOT}/scripts/collect_results.py" --method "${METHOD}" --model "${MODEL}" --run-dir "${RUN_DIR}" --family paper_baseline)
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
