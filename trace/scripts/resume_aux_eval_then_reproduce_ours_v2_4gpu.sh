#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace"
RUNS_ROOT="/data2/seonghyeonnoh/LLM-continual-learning-runs"
AUX_RUN="${V2NEW_AUX_RUN_DIR:-${RUNS_ROOT}/v2_new_expert_aux_mix0p1_20260812}"
V2_RUN="${OURS_V2_REPRO_RUN_DIR:-${RUNS_ROOT}/ours_v2_reproduction_20260812}"
CHAIN_LOG="${OURS_V2_REPRO_CHAIN_LOG:-${RUNS_ROOT}/ours_v2_reproduction_chain_20260812.log}"

mkdir -p "${AUX_RUN}/eval_queue_logs" "${V2_RUN}/eval_queue_logs"
if [[ -e "${V2_RUN}/0/lora_moe_meta.json" ]]; then
  echo "[ERROR] refusing to overwrite existing V2 reproduction: ${V2_RUN}" >&2
  exit 2
fi

exec > >(tee -a "${CHAIN_LOG}") 2>&1

cd "${ROOT}"
export PYTHONNOUSERSITE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TRACE_DATA_ROOT="${ROOT}/data/trace"

run_sparse15() {
  local label="$1"
  local method="$2"
  local run_dir="$3"
  export OURS_LORAMOE_OUTPUT_ROOT="${run_dir}"
  export SPARSE15_METHODS="${method}"
  export SPARSE15_GPUS="4,5,6,7"
  export SPARSE15_LOG_ROOT="${run_dir}/eval_queue_logs"
  export SPARSE15_EVAL_BATCH="32"
  export SPARSE15_SCIENCEQA_BATCH="128"
  export SPARSE15_20MINUTEN_BATCH="32"
  export SPARSE15_MEETINGBANK_BATCH="1"
  export SPARSE15_PY150_BATCH="8"
  export SPARSE15_CPU_THREADS="4"
  echo "[${label} EVAL START] $(date '+%F %T %Z')"
  "${ROOT}/.venv-runtime/bin/python" \
    "${ROOT}/scripts/run_ours_sparse15_efficient.py"
  [[ -s "${run_dir}/sparse15_summary.json" ]] || {
    echo "[ERROR] ${label} evaluation returned without summary" >&2
    exit 1
  }
  echo "[${label} EVAL COMPLETE] $(date '+%F %T %Z')"
}

echo "[CHAIN START] $(date '+%F %T %Z') gpus=4,5,6,7"

# Optionally finish the already-trained V2-new auxiliary model's sparse-15
# evaluation.  Set SKIP_AUX_EVAL=1 when only the Ours v2 reproduction matters.
if [[ "${SKIP_AUX_EVAL:-0}" == "1" ]]; then
  echo "[V2NEW AUX EVAL SKIPPED] $(date '+%F %T %Z')"
else
  run_sparse15 "V2NEW AUX" "ours_lora_moe_v2_new" "${AUX_RUN}"
fi

# Reproduce the Ours v2 algorithm/data contract.  Use the faster H100 profile
# requested for this confirmation run: 16 x 4 GPUs x accumulation 1.  This
# keeps the historical effective global batch of 64 while changing only its
# per-rank packing relative to the July recovery run.
export OURS_LORAMOE_OUTPUT_ROOT="${V2_RUN}"
export OURS_LORAMOE_GPUS="4,5,6,7"
export OURS_LORAMOE_PORT="29873"
export OURS_LORAMOE_MICRO_BATCH="16"
export OURS_LORAMOE_GRAD_ACCUM="1"
export OURS_LORAMOE_EPOCHS="5,3,7,5,3,5,5,7"
export OURS_LORAMOE_RANK="64"
export OURS_LORAMOE_ALPHA="128"
export OURS_LORAMOE_EXPERTS_PER_TASK="1"
export OURS_LORAMOE_TOP_K="1"
export OURS_LORAMOE_ROUTING_WEIGHT_MODE="full_softmax"
export OURS_LORAMOE_DROPOUT="0.05"
export OURS_LORAMOE_LR="2e-4"
export OURS_LORAMOE_SEED="2025"
export OURS_REPLAY_MANIFEST="${ROOT}/manifests/replay/trace_seed2025_random50_per_task.json"
export OURS_REPLAY_SUBSET_RATIO="0.01"
export OURS_REPLAY_SUBSET_SEED="-1"
export OURS_REPLAY_SELECTION_MODE="random"
export OURS_ROUTER_REPLAY_EXPOSURE_SAMPLES="1000"
export OURS_V2_KD_MEMORY_BATCH_SIZE="0"
export OURS_V2_KD_PASS_MULTIPLIER="1"
export OURS_V2_KD_LOSS_COEFF="1.0"
export OURS_V2_KD_TEMPERATURE="1.0"
export OURS_V2_KD_LR="0"
export OURS_V2_KD_CHUNK_TOKENS="256"
export OURS_V2_KD_TOKEN_SCOPE="nonpad"
export OURS_V2_REPLAY_LOSS_COEFF="1.0"
export OURS_V2_JOINT_NEW_TO_REPLAY_RATIO="5"
export OURS_LORAMOE_AUX_COEFF="0.01"
export OURS_LORAMOE_Z_COEFF="0.001"
export OURS_DISABLE_TRAINING_FLOP_COUNTER="1"
unset OURS_V2_NEW_EXPERT_AUX_MIX
unset OURS_V2_NEW_EXPERT_AUX_LOSS_COEFF
unset OURS_V2_NEW_EXPERT_QUOTA_SCHEDULE

echo "[OURS V2 TRAIN START] $(date '+%F %T %Z') run_dir=${V2_RUN}"
bash "${ROOT}/scripts/baselines/_run_ours_lora_moe.sh" train llama31 v2
for round in {0..7}; do
  [[ -s "${V2_RUN}/${round}/lora_moe_meta.json" ]] || {
    echo "[ERROR] missing Ours v2 checkpoint round ${round}" >&2
    exit 1
  }
done
echo "[OURS V2 TRAIN COMPLETE] $(date '+%F %T %Z') checkpoints=0..7"

run_sparse15 "OURS V2" "ours_lora_moe_v2" "${V2_RUN}"

echo "[CHAIN COMPLETE] $(date '+%F %T %Z')"
echo "[V2 CHECKPOINTS] ${V2_RUN}/{0..7}"
echo "[V2 SUMMARY] ${V2_RUN}/sparse15_summary.json"
