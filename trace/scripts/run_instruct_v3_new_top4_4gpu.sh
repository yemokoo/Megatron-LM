#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${INSTRUCT_MODEL_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}"
CACHE_ROOT="${INSTRUCT_TOKEN_CACHE:-/data2/seonghyeonnoh/LLM-continual-learning-caches/llama31_8b_instruct/slora_chat_full_len1024}"
RUN_ROOT="${INSTRUCT_RUN_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace}"
EXPERIMENT_NAME="${INSTRUCT_EXPERIMENT_NAME:-instruct_priority_fourway_20260812}"
OUTPUT="${INSTRUCT_V3_NEW_TOP4_OUT:-${RUN_ROOT}/v3/${EXPERIMENT_NAME}/v3_new_top4}"
GPUS="${INSTRUCT_GPUS:-4,5,6,7}"

if [[ -d "${OUTPUT}" ]] && find "${OUTPUT}" -mindepth 1 -print -quit | grep -q .; then
  echo "[ERROR] refusing to overwrite nonempty output: ${OUTPUT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT}"

export SLORA_LLAMA31_PATH="${MODEL_PATH}"
export OURS_LLAMA31_TOKEN_CACHE="${CACHE_ROOT}"
export OURS_LORAMOE_OUTPUT_ROOT="${OUTPUT}"
export OURS_LORAMOE_GPUS="${GPUS}"
export OURS_LORAMOE_PORT="${OURS_LORAMOE_PORT:-29835}"

# Four H100s: 8 samples/rank x 4 ranks x accumulation 2 = global batch 64.
export OURS_LORAMOE_MICRO_BATCH="${OURS_LORAMOE_MICRO_BATCH:-8}"
export OURS_LORAMOE_GRAD_ACCUM="${OURS_LORAMOE_GRAD_ACCUM:-2}"

# v3_new_top4's fixed capacity-normalized profile is checked by the runner:
# rank=16, alpha=128, four new experts/task, top-k=4, KD pass multiplier=2.
export OURS_V2_KD_MEMORY_BATCH_SIZE="${OURS_V2_KD_MEMORY_BATCH_SIZE:-4}"
export OURS_V2_REPLAY_FORWARD_BATCH_SIZE="${OURS_V2_REPLAY_FORWARD_BATCH_SIZE:-8}"
export OURS_LORAMOE_SEED="${OURS_LORAMOE_SEED:-2025}"
export OURS_REPLAY_SUBSET_SEED="${OURS_REPLAY_SUBSET_SEED:-2025}"
export OURS_REPLAY_SELECTION_MODE="${OURS_REPLAY_SELECTION_MODE:-random}"
export OURS_V3_EPOCH_PROBE_SAMPLES="${OURS_V3_EPOCH_PROBE_SAMPLES:-64}"

export PYTHONNOUSERSITE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

echo "[TRAIN START] $(date --iso-8601=seconds) version=v3_new_top4 gpus=${GPUS} output=${OUTPUT}"
bash "${ROOT}/scripts/baselines/_run_ours_lora_moe.sh" train llama31 v3_new_top4
echo "[TRAIN COMPLETE] $(date --iso-8601=seconds) version=v3_new_top4 output=${OUTPUT}"
