#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${TRACE_PYTHON:-${ROOT}/.venv-runtime/bin/python}"
MODEL="${INSTRUCT_MODEL_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}"
CACHE="${INSTRUCT_TOKEN_CACHE:-/data2/seonghyeonnoh/LLM-continual-learning-caches/llama31_8b_instruct/slora_chat_full_len1024}"
RUN_ROOT="${TRACE_RUN_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace}"
TAG="${TRACE_HIDDEN_MSE_C10_TAG:-instruct_hidden_mse_postkd_c10_20260814}"
V2_OUT="${RUN_ROOT}/v2/${TAG}/v2_new_hidden_mse_c10"
V3_OUT="${RUN_ROOT}/v3/${TAG}/v3_new_hidden_mse_c10"
LOG_ROOT="${RUN_ROOT}/logs/${TAG}/train_then_parallel_eval_20260815"
TOP4_OUTPUT="${RUN_ROOT}/v3/instruct_priority_fourway_20260812/v3_new_top4"
GRPO_QUEUE="${GRPO_QUEUE_SCRIPT:-/home/seonghyeonnoh/yemokoo/androidflux/rl/train/queue_parallel_grpo.sh}"
mkdir -p "${LOG_ROOT}"
status() { echo "$*" | tee -a "${LOG_ROOT}/status.log"; }
complete() { [[ -s "$1/7/pytorch_model.bin" && -s "$1/7/lora_moe_meta.json" ]]; }

train_one() {
  local version="$1" output="$2" port="$3" micro="$4" accum="$5" log="$6"
  if complete "${output}"; then status "[TRAIN SKIP] ${version} complete"; return; fi
  local resume=()
  if [[ "${version}" == v2_new && -s "${output}/1/pytorch_model.bin" && -s "${output}/1/lora_moe_meta.json" ]]; then
    resume=(OURS_LORAMOE_RESUME_CHECKPOINT="${output}/1" OURS_V2_ALLOW_MEMORY_BATCH_RESUME_OVERRIDE=1)
    status "[TRAIN RESUME] ${version} from ${output}/1; MeetingBank restarts with micro=8, GA=2; KD micro=8"
  elif [[ -d "${output}" ]] && find "${output}" -mindepth 1 -print -quit | grep -q .; then
    status "[ERROR] unsupported partial output: ${output}"; exit 2
  fi
  status "[TRAIN START] $(date --iso-8601=seconds) ${version} micro=${micro} ga=${accum} KD=8"
  env "${resume[@]}" \
    SLORA_LLAMA31_PATH="${MODEL}" OURS_LLAMA31_TOKEN_CACHE="${CACHE}" \
    OURS_LORAMOE_OUTPUT_ROOT="${output}" OURS_LORAMOE_GPUS=0,1,2,3 \
    OURS_LORAMOE_PORT="${port}" OURS_LORAMOE_MICRO_BATCH="${micro}" \
    OURS_LORAMOE_GRAD_ACCUM="${accum}" OURS_V2_KD_MEMORY_BATCH_SIZE=8 \
    OURS_V2_REPLAY_FORWARD_BATCH_SIZE=8 \
    OURS_V2_JOINT_REPLAY_OBJECTIVE=hidden_mse \
    OURS_V2_HIDDEN_MSE_LOSS_COEFF=10.0 OURS_V2_REPLAY_LOSS_COEFF=1.0 \
    OURS_V2_JOINT_NEW_TO_REPLAY_RATIO=5 OURS_V2_NEW_ACTIVE_MEMORY_CAP=1000 \
    OURS_V2_NEW_PERSISTENT_SAMPLES_PER_TASK=500 \
    OURS_HIDDEN_MSE_OOM_RETRY=1 OURS_HIDDEN_MSE_OOM_RETRY_KD_BATCH_SIZE=4 \
    OURS_HIDDEN_MSE_OOM_RETRY_REPLAY_FORWARD_BATCH_SIZE=2 \
    OURS_LORAMOE_SEED=2025 OURS_REPLAY_SUBSET_SEED=2025 \
    OURS_REPLAY_SELECTION_MODE=random OURS_V3_EPOCH_PROBE_SAMPLES=64 \
    PYTHONNOUSERSITE=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
      bash "${ROOT}/scripts/baselines/_run_ours_lora_moe.sh" train llama31 "${version}" \
      2>&1 | tee -a "${log}"
  complete "${output}" || { status "[ERROR] incomplete ${version}"; exit 3; }
}

eval_one() {
  local output="$1" method="$2" gpus="$3" log="$4"
  SLORA_LLAMA31_PATH="${MODEL}" OURS_LORAMOE_OUTPUT_ROOT="${output}" \
  SPARSE15_METHODS="${method}" SPARSE15_GPUS="${gpus}" \
  SPARSE15_EXACT_STOP_MARKERS=1 TRACE_PYTHON="${PYTHON}" \
  PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false \
    "${PYTHON}" -u "${ROOT}/scripts/run_ours_sparse15_optimized.py" \
      --matrix-mode sparse15 >"${log}" 2>&1
}

: > "${LOG_ROOT}/status.log"
status "[CHAIN START] $(date --iso-8601=seconds) train serial 4GPU; eval parallel 2+2GPU"
train_one v2_new "${V2_OUT}" 30051 \
  16,16,8,8,8,16,16,8 1,1,2,2,2,1,1,2 "${LOG_ROOT}/01_v2_train.log"
train_one v3_new "${V3_OUT}" 30052 \
  16,16,8,8,8,16,16,8 1,1,2,2,2,1,1,2 "${LOG_ROOT}/02_v3_train.log"

status "[EVAL START] $(date --iso-8601=seconds) V2 GPUs=0,1; V3 GPUs=2,3"
eval_one "${V2_OUT}" ours_lora_moe_v2_new_hidden_mse 0,1 "${LOG_ROOT}/03_v2_eval.log" & p2=$!
eval_one "${V3_OUT}" ours_lora_moe_v3_new_hidden_mse 2,3 "${LOG_ROOT}/04_v3_eval.log" & p3=$!
failed=0
wait "${p2}" || { status "[ERROR] V2 evaluation failed"; failed=1; }
wait "${p3}" || { status "[ERROR] V3 evaluation failed"; failed=1; }
[[ "${failed}" == 0 ]] || exit 4
status "[EVAL COMPLETE] $(date --iso-8601=seconds)"

status "[TOP4 LOWER] $(date --iso-8601=seconds)"
bash "${ROOT}/scripts/run_ours_lower_triangle_optimized.sh" \
  "${TOP4_OUTPUT}" ours_lora_moe_v3_new_top4 0,1,2,3 "${LOG_ROOT}/05_top4_lower.log"
if [[ "${TRACE_SKIP_FINAL_GRPO:-0}" == "1" ]]; then
  status "[GRPO SKIP] already completed by the parent chain"
else
  status "[GRPO] $(date --iso-8601=seconds)"
  GRPO_QUEUE_SKIP_SOURCE_WAIT=1 GRPO_QUEUE_TARGET_GPUS=0,1,2,3 \
    bash "${GRPO_QUEUE}" 2>&1 | tee -a "${LOG_ROOT}/06_grpo.log"
fi
status "[CHAIN COMPLETE] $(date --iso-8601=seconds)"
