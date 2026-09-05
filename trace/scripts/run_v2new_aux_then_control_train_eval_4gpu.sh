#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace"
RUNS_ROOT="/data2/seonghyeonnoh/LLM-continual-learning-runs"
AUX_RUN="${V2NEW_AUX_RUN_DIR:-${RUNS_ROOT}/v2_new_expert_aux_mix0p1_20260812}"
CONTROL_RUN="${V2NEW_CONTROL_RUN_DIR:-${RUNS_ROOT}/v2_new_control_retrain_20260812}"
CHAIN_LOG="${V2NEW_AUX_CONTROL_CHAIN_LOG:-${RUNS_ROOT}/v2_new_aux_then_control_20260812.log}"

for run_dir in "${AUX_RUN}" "${CONTROL_RUN}"; do
  if [[ -e "${run_dir}/0/lora_moe_meta.json" ]]; then
    echo "[ERROR] refusing to overwrite existing run: ${run_dir}" >&2
    exit 2
  fi
  mkdir -p "${run_dir}/eval_queue_logs"
done

exec > >(tee -a "${CHAIN_LOG}") 2>&1

cd "${ROOT}"
export PYTHONNOUSERSITE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TRACE_DATA_ROOT="${ROOT}/data/trace"

configure_common_train() {
  local run_dir="$1"
  local port="$2"
  export OURS_LORAMOE_OUTPUT_ROOT="${run_dir}"
  export OURS_LORAMOE_GPUS="4,5,6,7"
  export OURS_LORAMOE_PORT="${port}"
  export OURS_LORAMOE_MICRO_BATCH="8"
  export OURS_LORAMOE_GRAD_ACCUM="2"
  export OURS_LORAMOE_EPOCHS="5,3,7,5,3,5,5,7"
  export OURS_LORAMOE_RANK="64"
  export OURS_LORAMOE_ALPHA="128"
  export OURS_LORAMOE_EXPERTS_PER_TASK="1"
  export OURS_LORAMOE_TOP_K="1"
  export OURS_LORAMOE_ROUTING_WEIGHT_MODE="straight_through_topk"
  export OURS_V2_KD_MEMORY_BATCH_SIZE="0"
  export OURS_V2_KD_PASS_MULTIPLIER="1"
  export OURS_V2_KD_LOSS_COEFF="1.0"
  export OURS_V2_KD_TEMPERATURE="1.0"
  export OURS_V2_KD_LR="0"
  export OURS_V2_KD_CHUNK_TOKENS="256"
  export OURS_V2_KD_TOKEN_SCOPE="nonpad"
  export OURS_V2_REPLAY_LOSS_COEFF="1.0"
  export OURS_V2_JOINT_NEW_TO_REPLAY_RATIO="5"
  export OURS_V2_NEW_ACTIVE_MEMORY_CAP="1000"
  export OURS_V2_NEW_PERSISTENT_SAMPLES_PER_TASK="500"
  export OURS_REPLAY_SELECTION_MODE="random"
  export OURS_REPLAY_SUBSET_SEED="2025"
  export OURS_LORAMOE_SEED="2025"
  export OURS_DISABLE_TRAINING_FLOP_COUNTER="1"
  unset OURS_V2_NEW_EXPERT_QUOTA_SCHEDULE
}

run_train() {
  local label="$1"
  local run_dir="$2"
  echo "[${label} TRAIN START] $(date '+%F %T %Z') run_dir=${run_dir}"
  bash "${ROOT}/scripts/baselines/_run_ours_lora_moe.sh" \
    train llama31 v2_new
  if [[ ! -s "${run_dir}/7/lora_moe_meta.json" ]]; then
    echo "[ERROR] ${label} training returned without ${run_dir}/7" >&2
    exit 1
  fi
  echo "[${label} TRAIN COMPLETE] $(date '+%F %T %Z')"
}

run_eval() {
  local label="$1"
  local run_dir="$2"
  export OURS_LORAMOE_OUTPUT_ROOT="${run_dir}"
  export SPARSE15_METHODS="ours_lora_moe_v2_new"
  export SPARSE15_GPUS="4,5,6,7"
  export SPARSE15_LOG_ROOT="${run_dir}/eval_queue_logs"
  export SPARSE15_EVAL_BATCH="32"
  export SPARSE15_SCIENCEQA_BATCH="128"
  export SPARSE15_20MINUTEN_BATCH="32"
  export SPARSE15_MEETINGBANK_BATCH="1"
  export SPARSE15_PY150_BATCH="8"
  export SPARSE15_CPU_THREADS="4"
  echo "[${label} EVAL START] $(date '+%F %T %Z') sparse-15"
  "${ROOT}/.venv-runtime/bin/python" \
    "${ROOT}/scripts/run_ours_sparse15_efficient.py"
  if [[ ! -s "${run_dir}/sparse15_summary.json" ]]; then
    echo "[ERROR] ${label} evaluation returned without summary" >&2
    exit 1
  fi
  echo "[${label} EVAL COMPLETE] $(date '+%F %T %Z')"
}

echo "[CHAIN START] $(date '+%F %T %Z') gpus=4,5,6,7 global_batch=64"

# 1/4: V2-new plus the expert-only auxiliary LM branch.  The 0.1 mixture
# supplies a real but small gradient to the new expert on naturally missed
# tokens; beta=1 leaves that gradient scaled only by the mixture itself.
configure_common_train "${AUX_RUN}" 29871
export OURS_V2_NEW_EXPERT_AUX_MIX="0.1"
export OURS_V2_NEW_EXPERT_AUX_LOSS_COEFF="1.0"
run_train "AUX" "${AUX_RUN}"

# 2/4: sparse-15 acquisition/final evaluation for the treatment.
run_eval "AUX" "${AUX_RUN}"

# 3/4: fresh vanilla V2-new control with every other setting held fixed.
configure_common_train "${CONTROL_RUN}" 29872
unset OURS_V2_NEW_EXPERT_AUX_MIX
unset OURS_V2_NEW_EXPERT_AUX_LOSS_COEFF
run_train "CONTROL" "${CONTROL_RUN}"

# 4/4: sparse-15 evaluation for the fresh control.
run_eval "CONTROL" "${CONTROL_RUN}"

echo "[CHAIN COMPLETE] $(date '+%F %T %Z')"
echo "[AUX SUMMARY] ${AUX_RUN}/sparse15_summary.json"
echo "[CONTROL SUMMARY] ${CONTROL_RUN}/sparse15_summary.json"
