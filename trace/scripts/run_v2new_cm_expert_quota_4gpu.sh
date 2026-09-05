#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace"
SOURCE_RUN="/data2/seonghyeonnoh/LLM-continual-learning-runs/v2_new_retrain_20260807"
RUN_DIR="/data2/seonghyeonnoh/LLM-continual-learning-runs/v2_new_cm_expert_quota_50_30_20260810"
LOG_FILE="${RUN_DIR}/train.log"

mkdir -p "${RUN_DIR}/fixed_replay_memory"
cp -a "${SOURCE_RUN}/fixed_replay_memory/." "${RUN_DIR}/fixed_replay_memory/"

cd "${ROOT}"
export PYTHONNOUSERSITE=1
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRACE_DATA_ROOT="/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace"
export SLORA_LLAMA31_PATH="/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct"
export OURS_LLAMA31_TOKEN_CACHE="${ROOT}/cache/tokenized/llama31_8b_base/slora_chat_full_len1024"
export OURS_LORAMOE_OUTPUT_ROOT="${RUN_DIR}"
export OURS_LORAMOE_RESUME_CHECKPOINT="${SOURCE_RUN}/4"
export OURS_LORAMOE_STOP_AFTER_TASK="NumGLUE-cm"
export OURS_LORAMOE_GPUS="${OURS_LORAMOE_GPUS:-4,5,6,7}"
export OURS_LORAMOE_MICRO_BATCH="16"
export OURS_LORAMOE_GRAD_ACCUM="1"
export OURS_LORAMOE_EPOCHS="5,3,7,5,3,5,5,7"
export OURS_LORAMOE_SEED="2025"
export OURS_REPLAY_SUBSET_SEED="2025"
export OURS_V2_NEW_EXPERT_QUOTA_SCHEDULE="0.5,0.3,0,0,0"
export OURS_V2_ACQUISITION_DIAGNOSTIC_INTERVAL="10"
export OURS_DISABLE_TRAINING_FLOP_COUNTER="1"
export OURS_LORAMOE_PORT="29741"
export DRY_RUN="0"

mkdir -p "${RUN_DIR}"
echo "[START] $(date '+%F %T %Z') ScienceQA /4 -> identical KD -> CM dual-route quota"
echo "[OUTPUT] ${RUN_DIR}"
echo "[QUOTA] ${OURS_V2_NEW_EXPERT_QUOTA_SCHEDULE}"
exec bash "${ROOT}/scripts/baselines/_run_ours_lora_moe.sh" \
  train llama31 v2_new 2>&1 | tee "${LOG_FILE}"
