#!/bin/bash
# hidden-MSE at 1:1 replay volume with the full task stored as memory.
#
# v3_new runs 5:1 (1,000 replay per 5,000-sample epoch).  Its forgetting is
# concentrated on the two most recent tasks (NumGLUE-cm 7.41, NumGLUE-ds 9.85
# versus 0.35 for the oldest), which is what more replay should reach.
#
# Architecture, schedule, data and effective batch are the v3_new baseline's;
# only the replay knobs move.  scripts/assert_matches_v3_new_baseline.py
# enforces that before training starts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${OURS_ST_RUN_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/v3_hidden_mse_1to1_p5k}"
OUTPUT="${RUN_ROOT}/v3_new_hidden_mse_1to1_p5k_st_top1"

if [[ -e "${OUTPUT}/0/lora_moe_meta.json" && -z "${OURS_LORAMOE_RESUME_CHECKPOINT:-}" ]]; then
  echo "[ERROR] refusing to overwrite existing run: ${OUTPUT}" >&2
  exit 2
fi
mkdir -p "${OUTPUT}"

# The base checkpoint has no chat_template and silently changes the training
# format, so never let this resolve by accident.
export SLORA_LLAMA31_PATH="${SLORA_LLAMA31_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}"
export TRACE_DATA_ROOT="${TRACE_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace}"
if ! grep -q chat_template "${SLORA_LLAMA31_PATH}/tokenizer_config.json" 2>/dev/null; then
  echo "[ERROR] ${SLORA_LLAMA31_PATH} has no chat_template; that is the BASE model" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

export OURS_LORAMOE_OUTPUT_ROOT="${OUTPUT}"
export OURS_LORAMOE_GPUS="${OURS_LORAMOE_GPUS:-0,1,2,3,4,5,6,7}"
export OURS_LORAMOE_PORT="${OURS_LORAMOE_PORT:-29811}"
# 8 x 8 x 1 = 64, the baseline's effective batch.
export OURS_LORAMOE_MICRO_BATCH="${OURS_LORAMOE_MICRO_BATCH:-8}"
export OURS_LORAMOE_GRAD_ACCUM="${OURS_LORAMOE_GRAD_ACCUM:-1}"

# Architecture and schedule: identical to v3_new.
export OURS_LORAMOE_TOP_K=1
export OURS_LORAMOE_RANK="${OURS_LORAMOE_RANK:-64}"
export OURS_LORAMOE_ALPHA="${OURS_LORAMOE_ALPHA:-128}"
export OURS_LORAMOE_EXPERTS_PER_TASK=1
export OURS_LORAMOE_ROUTING_WEIGHT_MODE=straight_through_topk
export OURS_REPLAY_SUBSET_SEED="${OURS_REPLAY_SUBSET_SEED:-2025}"
export OURS_REPLAY_SELECTION_MODE="${OURS_REPLAY_SELECTION_MODE:-random}"
export OURS_V2_KD_MEMORY_BATCH_SIZE=8
# KD-init is NOT part of the replay increase.  The active-memory cap used to
# size both streams, so raising replay to 5,000 also made KD-init 5,000 x 3/5/7
# epochs -- 33 minutes a round on Py150 for a phase whose job is just to seed
# the new expert.  Pin KD to the published 1,000-record stream and one pass.
export OURS_V2_KD_EXPOSURE_SAMPLES=1000
export OURS_V2_KD_EPOCHS=1

export OURS_V3_EPOCH_PROBE_SAMPLES=64

# Stored memory is raised to the whole task: 5,000 records each instead of the
# baseline's 500.  At 1:1 the 500-record variant replays round 2's single old
# task ten times per epoch, so this run separates "more replay" from "more
# distinct replay".  It costs the published stored-sample figure, which is why
# it is a companion run rather than the headline one.
export OURS_REPLAY_SUBSET_RATIO=1.0
export OURS_V2_NEW_PERSISTENT_SAMPLES_PER_TASK=5000

# 1:1 -- one primary epoch is 5,000 samples, so replay contributes 5,000 too.
# The share per old task therefore shrinks as tasks accumulate: round 2 gives
# one task all 5,000, round 8 splits it across seven.
export OURS_V2_JOINT_NEW_TO_REPLAY_RATIO=1
export OURS_V2_NEW_ACTIVE_MEMORY_CAP=5000
export OURS_ROUTER_REPLAY_EXPOSURE_SAMPLES=5000
export OURS_V2_JOINT_REPLAY_OBJECTIVE=hidden_mse
export OURS_V2_HIDDEN_MSE_LOSS_COEFF="${OURS_V2_HIDDEN_MSE_LOSS_COEFF:-1.0}"

echo "[TRAIN START] v3_new_hidden_mse_1to1_p5k  GPUs=${OURS_LORAMOE_GPUS}"
echo "[TRAIN START] base model = ${SLORA_LLAMA31_PATH}"
echo "[TRAIN START] replay 5,000/epoch (1:1), objective=lm, distribution=${OURS_REPLAY_DISTRIBUTION:-equal_task}"

# OURS_ACTION=validate dry-runs the whole configuration without touching a GPU.
exec bash "${ROOT}/scripts/baselines/_run_ours_lora_moe.sh" \
  "${OURS_ACTION:-train}" llama31 v3_new_hidden_mse_1to1_p5k
