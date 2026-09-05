#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${TRACE_PYTHON:-${ROOT}/.venv-runtime/bin/python}"
MODEL="${INSTRUCT_MODEL_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}"
CACHE="${INSTRUCT_TOKEN_CACHE:-/data2/seonghyeonnoh/LLM-continual-learning-caches/llama31_8b_instruct/slora_chat_full_len1024}"
RUN_ROOT="${TRACE_RUN_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace}"
TAG="${TRACE_HIDDEN_MSE_TAG:-instruct_hidden_mse_postkd_20260813}"
GPUS="${TRACE_CHAIN_GPUS:-0,1,2,3}"
V3_REFERENCE="${TRACE_V3_REFERENCE:-${RUN_ROOT}/v3/instruct_priority_fourway_20260812/v3_new}"
V2_MSE="${TRACE_V2_HIDDEN_MSE_OUT:-${RUN_ROOT}/v2/${TAG}/v2_new_hidden_mse}"
V3_MSE="${TRACE_V3_HIDDEN_MSE_OUT:-${RUN_ROOT}/v3/${TAG}/v3_new_hidden_mse}"
SLORA_ROOT="${TRACE_SLORA_PRE_ROOT:-${RUN_ROOT}/slora/${TAG}}"
LOG_ROOT="${TRACE_HIDDEN_MSE_LOG_ROOT:-${RUN_ROOT}/logs/${TAG}}"
mkdir -p "${LOG_ROOT}" "$(dirname "${V2_MSE}")" \
  "$(dirname "${V3_MSE}")" "${SLORA_ROOT}"

training_complete() {
  local output="$1"
  [[ -s "${output}/7/pytorch_model.bin" && \
     -s "${output}/7/lora_moe_meta.json" ]]
}

smoke_complete() {
  local output="$1"
  [[ -s "${output}/1/pytorch_model.bin" && \
     -s "${output}/1/lora_moe_meta.json" && \
     -s "${output}/SMOKE_COMPLETE" ]]
}

refuse_partial_training() {
  local output="$1"
  if [[ -d "${output}" ]] && find "${output}" -mindepth 1 -print -quit | grep -q .; then
    echo "[ERROR] partial/nonempty output needs explicit resume handling: ${output}" >&2
    exit 2
  fi
}

run_lower_triangle() {
  local output="$1" method="$2" log="$3"
  bash "${ROOT}/scripts/run_ours_lower_triangle_optimized.sh" \
    "${output}" "${method}" "${GPUS}" "${log}"
}

run_ours_hidden_mse_train() {
  local version="$1" output="$2" port="$3" log="$4"
  if training_complete "${output}"; then
    echo "[TRAIN SKIP COMPLETE] $(date --iso-8601=seconds) ${version} ${output}" | tee -a "${log}"
    return
  fi
  refuse_partial_training "${output}"
  echo "[TRAIN START] $(date --iso-8601=seconds) ${version} hidden-MSE ${output}" | tee -a "${log}"
  SLORA_LLAMA31_PATH="${MODEL}" \
  OURS_LLAMA31_TOKEN_CACHE="${CACHE}" \
  OURS_LORAMOE_OUTPUT_ROOT="${output}" \
  OURS_LORAMOE_GPUS="${GPUS}" \
  OURS_LORAMOE_PORT="${port}" \
  OURS_LORAMOE_MICRO_BATCH=8 \
  OURS_LORAMOE_GRAD_ACCUM=2 \
  OURS_V2_KD_MEMORY_BATCH_SIZE=4 \
  OURS_V2_REPLAY_FORWARD_BATCH_SIZE=8 \
  OURS_V2_JOINT_REPLAY_OBJECTIVE=hidden_mse \
  OURS_V2_HIDDEN_MSE_LOSS_COEFF=1.0 \
  OURS_V2_REPLAY_LOSS_COEFF=1.0 \
  OURS_V2_JOINT_NEW_TO_REPLAY_RATIO=5 \
  OURS_V2_NEW_ACTIVE_MEMORY_CAP=1000 \
  OURS_V2_NEW_PERSISTENT_SAMPLES_PER_TASK=500 \
  OURS_LORAMOE_SEED=2025 \
  OURS_REPLAY_SUBSET_SEED=2025 \
  OURS_REPLAY_SELECTION_MODE=random \
  OURS_V3_EPOCH_PROBE_SAMPLES=64 \
  PYTHONNOUSERSITE=1 \
  WANDB_MODE=offline \
  TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 \
  OPENBLAS_NUM_THREADS=8 \
    bash "${ROOT}/scripts/baselines/_run_ours_lora_moe.sh" \
      train llama31 "${version}" 2>&1 | tee -a "${log}"
  training_complete "${output}" || {
    echo "[ERROR] incomplete ${version} hidden-MSE output: ${output}" >&2
    exit 3
  }
}

run_ours_hidden_mse_smoke() {
  local version="$1" port="$2" log="$3"
  local output="${RUN_ROOT}/${version%%_*}/${TAG}/smoke_${version}_hidden_mse_through_fomc"
  if smoke_complete "${output}"; then
    echo "[SMOKE SKIP COMPLETE] ${version} ${output}" | tee -a "${log}"
    return
  fi
  refuse_partial_training "${output}"
  echo "[SMOKE START] $(date --iso-8601=seconds) ${version}: " \
       "C-STANCE 1 epoch -> expansion KD -> FOMC hidden-MSE joint 1 epoch" \
       | tee -a "${log}"
  SLORA_LLAMA31_PATH="${MODEL}" \
  OURS_LLAMA31_TOKEN_CACHE="${CACHE}" \
  OURS_LORAMOE_OUTPUT_ROOT="${output}" \
  OURS_LORAMOE_GPUS="${GPUS}" \
  OURS_LORAMOE_PORT="${port}" \
  OURS_LORAMOE_EPOCHS=1,1,1,1,1,1,1,1 \
  OURS_LORAMOE_STOP_AFTER_TASK=FOMC \
  OURS_LORAMOE_MICRO_BATCH=8 \
  OURS_LORAMOE_GRAD_ACCUM=2 \
  OURS_V2_KD_MEMORY_BATCH_SIZE=4 \
  OURS_V2_REPLAY_FORWARD_BATCH_SIZE=8 \
  OURS_V2_JOINT_REPLAY_OBJECTIVE=hidden_mse \
  OURS_V2_HIDDEN_MSE_LOSS_COEFF=1.0 \
  OURS_V2_JOINT_NEW_TO_REPLAY_RATIO=5 \
  OURS_V2_NEW_ACTIVE_MEMORY_CAP=1000 \
  OURS_V2_NEW_PERSISTENT_SAMPLES_PER_TASK=500 \
  OURS_LORAMOE_SEED=2025 \
  OURS_REPLAY_SUBSET_SEED=2025 \
  OURS_REPLAY_SELECTION_MODE=random \
  OURS_V3_EPOCH_PROBE_SAMPLES=64 \
  PYTHONNOUSERSITE=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    bash "${ROOT}/scripts/baselines/_run_ours_lora_moe.sh" \
      train llama31 "${version}" 2>&1 | tee -a "${log}"
  [[ -s "${output}/1/pytorch_model.bin" && \
     -s "${output}/1/lora_moe_meta.json" ]] || {
    echo "[ERROR] ${version} hidden-MSE smoke did not save FOMC checkpoint" >&2
    exit 3
  }
  grep -q "frozen teacher = expanded post-KD-init" "${output}/train.log" || {
    echo "[ERROR] ${version} smoke never constructed the post-KD teacher" >&2
    exit 3
  }
  grep -q "joint every-update router hidden_mse replay" "${output}/train.log" || {
    echo "[ERROR] ${version} smoke never entered hidden-MSE replay" >&2
    exit 3
  }
  "${PYTHON}" - "${output}/1/lora_moe_meta.json" <<'PY'
import json
import math
import sys
from pathlib import Path
meta = json.loads(Path(sys.argv[1]).read_text())
v2 = meta["v2"]
assert v2["joint_replay_objective"] == "hidden_mse", v2
assert v2["hidden_mse_teacher"] == "expanded_post_kd_init", v2
assert v2["hidden_mse_targets"] == "all_decoder_layer_outputs", v2
assert math.isfinite(float(v2["hidden_mse_loss_coeff"])), v2
PY
  touch "${output}/SMOKE_COMPLETE"
  echo "[SMOKE PASS] $(date --iso-8601=seconds) ${version} ${output}" \
    | tee -a "${log}"
}

run_ours_sparse15() {
  local output="$1" method="$2" log="$3"
  echo "[EVAL START] $(date --iso-8601=seconds) ${method}" | tee -a "${log}"
  SLORA_LLAMA31_PATH="${MODEL}" \
  OURS_LORAMOE_OUTPUT_ROOT="${output}" \
  SPARSE15_METHODS="${method}" \
  SPARSE15_GPUS="${GPUS}" \
  SPARSE15_EXACT_STOP_MARKERS=1 \
  TRACE_PYTHON="${PYTHON}" \
  PYTHONNOUSERSITE=1 \
  TOKENIZERS_PARALLELISM=false \
    "${PYTHON}" -u "${ROOT}/scripts/run_ours_sparse15_optimized.py" \
      --matrix-mode sparse15 2>&1 | tee -a "${log}"
}

run_slora_pre() {
  local run_dir="${SLORA_ROOT}/llama31/pre"
  local train_log="${LOG_ROOT}/05_slora_pre_train.log"
  if [[ ! -s "${run_dir}/order8/adapter_model.safetensors" ]]; then
    echo "[SLORA PRE TRAIN START] $(date --iso-8601=seconds)" | tee -a "${train_log}"
    CUDA_VISIBLE_DEVICES="${GPUS}" \
    WORLD_SIZE=4 MICRO_BATCH=8 GRAD_ACCUM=2 \
    SLORA_OUTPUT_ROOT="${SLORA_ROOT}" \
    SLORA_LLAMA31_PATH="${MODEL}" \
    SLORA_LLAMA31_TOKEN_CACHE="${CACHE}" \
    PYTHONNOUSERSITE=1 WANDB_MODE=offline \
      bash "${ROOT}/scripts/baselines/llama31/slora_pre.sh" train \
      2>&1 | tee -a "${train_log}"
  else
    echo "[SLORA PRE TRAIN SKIP COMPLETE] ${run_dir}" | tee -a "${train_log}"
  fi

  echo "[SLORA PRE EVAL START] $(date --iso-8601=seconds) 4 parallel cell queues" \
    | tee -a "${LOG_ROOT}/06_slora_pre_eval.log"
  local pids=()
  local index gpu shard_log
  IFS=',' read -r -a gpu_list <<< "${GPUS}"
  for index in 0 1 2 3; do
    gpu="${gpu_list[$index]}"
    shard_log="${LOG_ROOT}/06_slora_pre_eval_shard${index}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    SLORA_OUTPUT_ROOT="${SLORA_ROOT}" \
    SLORA_LLAMA31_PATH="${MODEL}" \
    EVAL_ALL_ROUNDS=1 EVAL_SPARSE_15=1 \
    EVAL_SHARD_COUNT=4 EVAL_SHARD_INDEX="${index}" \
    SLORA_EVAL_BATCH=4 PYTHONNOUSERSITE=1 \
      bash "${ROOT}/implementations/SLoRA-repro/scripts/repro/eval_trace.sh" \
        pre llama31 >"${shard_log}" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for index in 0 1 2 3; do
    if ! wait "${pids[$index]}"; then
      echo "[ERROR] SLoRA eval shard ${index} failed; see ${LOG_ROOT}/06_slora_pre_eval_shard${index}.log" >&2
      failed=1
    fi
  done
  [[ "${failed}" == 0 ]] || exit 4
  "${PYTHON}" "${ROOT}/scripts/collect_results.py" \
    --method slora_pre --model llama31 --sparse-15 \
    --run-dir "${run_dir}" --family slora \
    --output "${run_dir}/sparse15_summary.json" \
    2>&1 | tee -a "${LOG_ROOT}/06_slora_pre_eval.log"
}

echo "[CHAIN START] $(date --iso-8601=seconds) GPUs=${GPUS}" | tee "${LOG_ROOT}/status.log"
echo "[1/8] V2-new actual H100 hidden-MSE smoke" | tee -a "${LOG_ROOT}/status.log"
run_ours_hidden_mse_smoke v2_new 29939 \
  "${LOG_ROOT}/01_v2_new_hidden_mse_smoke.log"
echo "[2/8] V3-new actual H100 hidden-MSE smoke" | tee -a "${LOG_ROOT}/status.log"
run_ours_hidden_mse_smoke v3_new 29940 \
  "${LOG_ROOT}/02_v3_new_hidden_mse_smoke.log"

echo "[3/8] v3_new full lower triangle" | tee -a "${LOG_ROOT}/status.log"
run_lower_triangle "${V3_REFERENCE}" ours_lora_moe_v3_new \
  "${LOG_ROOT}/03_v3_new_lower_triangle.log"

echo "[4/8] v2_new + post-KD all-layer hidden-MSE replay" | tee -a "${LOG_ROOT}/status.log"
run_ours_hidden_mse_train v2_new "${V2_MSE}" 29941 \
  "${LOG_ROOT}/04_v2_new_hidden_mse_train.log"
echo "[5/8] v2_new hidden-MSE sparse-15" | tee -a "${LOG_ROOT}/status.log"
run_ours_sparse15 "${V2_MSE}" ours_lora_moe_v2_new_hidden_mse \
  "${LOG_ROOT}/05_v2_new_hidden_mse_eval.log"

echo "[6/8] v3_new + post-KD all-layer hidden-MSE replay" | tee -a "${LOG_ROOT}/status.log"
run_ours_hidden_mse_train v3_new "${V3_MSE}" 29942 \
  "${LOG_ROOT}/06_v3_new_hidden_mse_train.log"
echo "[7/8] v3_new hidden-MSE sparse-15" | tee -a "${LOG_ROOT}/status.log"
run_ours_sparse15 "${V3_MSE}" ours_lora_moe_v3_new_hidden_mse \
  "${LOG_ROOT}/07_v3_new_hidden_mse_eval.log"

echo "[8/8] SLoRA-Pre retrain + sparse-15" | tee -a "${LOG_ROOT}/status.log"
run_slora_pre
echo "[CHAIN COMPLETE] $(date --iso-8601=seconds)" | tee -a "${LOG_ROOT}/status.log"
