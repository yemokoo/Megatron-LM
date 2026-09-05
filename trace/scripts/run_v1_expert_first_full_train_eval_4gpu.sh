#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace"
RUN_DIR="${V1_EXPERT_FIRST_RUN_DIR:-/data2/seonghyeonnoh/LLM-continual-learning-runs/v1_expert_first_full_20260811}"
LOG_DIR="${RUN_DIR}/eval_queue_logs"

if [[ -e "${RUN_DIR}/0/lora_moe_meta.json" ]]; then
  echo "[ERROR] refusing to overwrite existing run: ${RUN_DIR}" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

cd "${ROOT}"
export PYTHONNOUSERSITE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TRACE_DATA_ROOT="${ROOT}/data/trace"

# Match the completed 8-GPU V2-new full run except for using physical GPUs
# 4,5,6,7.  micro=8 remains unchanged; accumulation doubles so the effective
# global batch remains 64: 8 * 4 * 2.
export OURS_LORAMOE_OUTPUT_ROOT="${RUN_DIR}"
export OURS_LORAMOE_GPUS="4,5,6,7"
export OURS_LORAMOE_PORT="${OURS_LORAMOE_PORT:-29841}"
export OURS_LORAMOE_MICRO_BATCH="8"
export OURS_LORAMOE_GRAD_ACCUM="2"
export OURS_LORAMOE_EPOCHS="5,3,7,5,3,5,5,7"
export OURS_LORAMOE_RANK="64"
export OURS_LORAMOE_ALPHA="128"
export OURS_LORAMOE_EXPERTS_PER_TASK="1"
export OURS_LORAMOE_TOP_K="1"
export OURS_LORAMOE_ROUTING_WEIGHT_MODE="straight_through_topk"
export OURS_V1_ROUTER_RETUNE_EPOCHS="1"
export OURS_ROUTER_REPLAY_EXPOSURE_SAMPLES="1000"
export OURS_V2_KD_MEMORY_BATCH_SIZE="0"
export OURS_V2_KD_PASS_MULTIPLIER="1"
export OURS_V2_KD_LOSS_COEFF="1.0"
export OURS_V2_KD_TEMPERATURE="1.0"
export OURS_V2_KD_LR="0"
export OURS_V2_KD_CHUNK_TOKENS="256"
export OURS_V2_KD_TOKEN_SCOPE="nonpad"
export OURS_V2_NEW_ACTIVE_MEMORY_CAP="1000"
export OURS_V2_NEW_PERSISTENT_SAMPLES_PER_TASK="500"
export OURS_REPLAY_SELECTION_MODE="random"
export OURS_REPLAY_SUBSET_SEED="2025"
export OURS_LORAMOE_SEED="2025"
export OURS_DISABLE_TRAINING_FLOP_COUNTER="1"

echo "[CHAIN START] $(date '+%F %T %Z')"
echo "[TRAIN START] version=v1_expert_first gpus=4,5,6,7 micro=8 accum=2 global=64"
bash "${ROOT}/scripts/baselines/_run_ours_lora_moe.sh" \
  train llama31 v1_expert_first

if [[ ! -s "${RUN_DIR}/7/lora_moe_meta.json" ]]; then
  echo "[ERROR] training returned without final checkpoint ${RUN_DIR}/7" >&2
  exit 1
fi
echo "[TRAIN COMPLETE] $(date '+%F %T %Z')"

export SPARSE15_METHODS="ours_lora_moe_v1_expert_first"
export SPARSE15_GPUS="4,5,6,7"
export SPARSE15_LOG_ROOT="${LOG_DIR}"
export SPARSE15_EVAL_BATCH="32"
export SPARSE15_SCIENCEQA_BATCH="128"
export SPARSE15_20MINUTEN_BATCH="32"
export SPARSE15_MEETINGBANK_BATCH="1"
export SPARSE15_PY150_BATCH="8"
export SPARSE15_CPU_THREADS="4"

echo "[EVAL START] $(date '+%F %T %Z') sparse-15 gpus=4,5,6,7"
"${ROOT}/.venv-runtime/bin/python" \
  "${ROOT}/scripts/run_ours_sparse15_efficient.py"

if [[ ! -s "${RUN_DIR}/sparse15_summary.json" ]]; then
  echo "[ERROR] evaluation returned without sparse15_summary.json" >&2
  exit 1
fi
echo "[EVAL COMPLETE] $(date '+%F %T %Z')"
echo "[CHAIN COMPLETE] $(date '+%F %T %Z') run_dir=${RUN_DIR}"
