#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${TRACE_PYTHON:-${ROOT}/.venv-runtime/bin/python}"
MODEL="${INSTRUCT_MODEL_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}"
CACHE="${INSTRUCT_TOKEN_CACHE:-/data2/seonghyeonnoh/LLM-continual-learning-caches/llama31_8b_instruct/slora_chat_full_len1024}"
RUN_ROOT="${TRACE_RUN_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace}"
TAG="${TRACE_HIDDEN_MSE_C10_TAG:-instruct_hidden_mse_postkd_c10_20260814}"
GPUS="${TRACE_CHAIN_GPUS:-0,1,2,3}"
LOG_ROOT="${RUN_ROOT}/logs/${TAG}"
V2_OUT="${RUN_ROOT}/v2/${TAG}/v2_new_hidden_mse_c10"
V3_OUT="${RUN_ROOT}/v3/${TAG}/v3_new_hidden_mse_c10"
mkdir -p "${LOG_ROOT}" "$(dirname "${V2_OUT}")" "$(dirname "${V3_OUT}")"

status() { echo "$*" | tee -a "${LOG_ROOT}/status.log"; }
complete() { [[ -s "$1/7/pytorch_model.bin" && -s "$1/7/lora_moe_meta.json" ]]; }

train_one() {
  local version="$1" output="$2" port="$3" log="$4"
  if complete "${output}"; then
    status "[TRAIN SKIP COMPLETE] $(date --iso-8601=seconds) ${version} ${output}"
    return
  fi
  if [[ -d "${output}" ]] && find "${output}" -mindepth 1 -print -quit | grep -q .; then
    status "[ERROR] partial output requires explicit resume: ${output}"
    exit 2
  fi
  status "[TRAIN START] $(date --iso-8601=seconds) ${version} hidden-MSE c=10"
  SLORA_LLAMA31_PATH="${MODEL}" \
  OURS_LLAMA31_TOKEN_CACHE="${CACHE}" \
  OURS_LORAMOE_OUTPUT_ROOT="${output}" \
  OURS_LORAMOE_GPUS="${GPUS}" \
  OURS_LORAMOE_PORT="${port}" \
  OURS_LORAMOE_MICRO_BATCH=16 \
  OURS_LORAMOE_GRAD_ACCUM=1 \
  OURS_V2_KD_MEMORY_BATCH_SIZE=4 \
  OURS_V2_REPLAY_FORWARD_BATCH_SIZE=8 \
  OURS_V2_JOINT_REPLAY_OBJECTIVE=hidden_mse \
  OURS_V2_HIDDEN_MSE_LOSS_COEFF=10.0 \
  OURS_V2_REPLAY_LOSS_COEFF=1.0 \
  OURS_V2_JOINT_NEW_TO_REPLAY_RATIO=5 \
  OURS_V2_NEW_ACTIVE_MEMORY_CAP=1000 \
  OURS_V2_NEW_PERSISTENT_SAMPLES_PER_TASK=500 \
  OURS_HIDDEN_MSE_OOM_RETRY=1 \
  OURS_HIDDEN_MSE_OOM_RETRY_KD_BATCH_SIZE=2 \
  OURS_HIDDEN_MSE_OOM_RETRY_REPLAY_FORWARD_BATCH_SIZE=2 \
  OURS_LORAMOE_SEED=2025 \
  OURS_REPLAY_SUBSET_SEED=2025 \
  OURS_REPLAY_SELECTION_MODE=random \
  OURS_V3_EPOCH_PROBE_SAMPLES=64 \
  PYTHONNOUSERSITE=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    bash "${ROOT}/scripts/baselines/_run_ours_lora_moe.sh" \
      train llama31 "${version}" 2>&1 | tee -a "${log}"
  complete "${output}" || { status "[ERROR] incomplete output: ${output}"; exit 3; }
  "${PYTHON}" - "${output}/7/lora_moe_meta.json" <<'PY'
import json, sys
from pathlib import Path
m = json.loads(Path(sys.argv[1]).read_text())["v2"]
assert m["joint_replay_objective"] == "hidden_mse", m
assert m["hidden_mse_teacher"] == "expanded_post_kd_init", m
assert m["hidden_mse_targets"] == "all_decoder_layer_outputs", m
assert float(m["hidden_mse_loss_coeff"]) == 10.0, m
PY
}

eval_one() {
  local output="$1" method="$2" log="$3"
  status "[EVAL START] $(date --iso-8601=seconds) ${method}"
  SLORA_LLAMA31_PATH="${MODEL}" \
  OURS_LORAMOE_OUTPUT_ROOT="${output}" \
  SPARSE15_METHODS="${method}" SPARSE15_GPUS="${GPUS}" \
  SPARSE15_EXACT_STOP_MARKERS=1 TRACE_PYTHON="${PYTHON}" \
  PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false \
    "${PYTHON}" -u "${ROOT}/scripts/run_ours_sparse15_optimized.py" \
      --matrix-mode sparse15 2>&1 | tee -a "${log}"
}

: > "${LOG_ROOT}/status.log"
status "[C10 CHAIN START] $(date --iso-8601=seconds) GPUs=${GPUS} layerwise-MSE=all32 reduction=mean coeff=10"
train_one v2_new "${V2_OUT}" 30041 "${LOG_ROOT}/01_v2_train.log"
eval_one "${V2_OUT}" ours_lora_moe_v2_new_hidden_mse "${LOG_ROOT}/02_v2_eval.log"
train_one v3_new "${V3_OUT}" 30042 "${LOG_ROOT}/03_v3_train.log"
eval_one "${V3_OUT}" ours_lora_moe_v3_new_hidden_mse "${LOG_ROOT}/04_v3_eval.log"
status "[C10 CHAIN COMPLETE] $(date --iso-8601=seconds)"
