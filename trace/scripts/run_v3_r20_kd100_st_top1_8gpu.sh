#!/bin/bash
# The missing corner of the replay x KD grid: baseline replay, raised KD.
#
# v3_new (replay 20%, KD 20%) -> KD35k (replay 100%, KD 20%) showed replay is
# worth +1.19 AA, and KD35k -> replay1to1 (KD 100%) showed KD is worth -2.04
# forgetting.  What no run covers is replay 20% with KD 100%: if KD alone
# carries most of the gain, the 1:1 replay stream -- five times the replay
# tokens, 23% of total training compute -- can be dropped.
#
# Architecture, schedule, data and effective batch are the v3_new baseline's;
# only the replay knobs move.  scripts/assert_matches_v3_new_baseline.py
# enforces that before training starts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${OURS_ST_RUN_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/v3_r20_kd100}"
OUTPUT="${RUN_ROOT}/v3_new_r20_kd100_st_top1"

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
export OURS_LORAMOE_PORT="${OURS_LORAMOE_PORT:-29851}"
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
export OURS_V3_EPOCH_PROBE_SAMPLES=64

# Stored memory is unchanged from the baseline: 500 records per task at
# ratio 0.1.  Only the amount fed to the router moves, so the method keeps
# its published memory footprint.
export OURS_REPLAY_SUBSET_RATIO=0.1
export OURS_V2_NEW_PERSISTENT_SAMPLES_PER_TASK=500

# Replay stays at the baseline's 1,000 per epoch (the cap drives it), while KD
# is given the 5,000-record stream replay1to1 used.  The two are independent
# only because --v2_kd_exposure_samples exists; the cap alone would move both.
export OURS_V2_JOINT_NEW_TO_REPLAY_RATIO=5
export OURS_V2_NEW_ACTIVE_MEMORY_CAP=1000
export OURS_ROUTER_REPLAY_EXPOSURE_SAMPLES=1000
export OURS_V2_KD_EXPOSURE_SAMPLES=5000
export OURS_V2_JOINT_REPLAY_OBJECTIVE=lm

echo "[TRAIN START] v3_new_r20_kd100  GPUs=${OURS_LORAMOE_GPUS}"
echo "[TRAIN START] base model = ${SLORA_LLAMA31_PATH}"
echo "[TRAIN START] replay 5,000/epoch (1:1), objective=lm, distribution=${OURS_REPLAY_DISTRIBUTION:-equal_task}"

# OURS_ACTION=validate dry-runs the whole configuration without touching a GPU.
exec bash "${ROOT}/scripts/baselines/_run_ours_lora_moe.sh" \
  "${OURS_ACTION:-train}" llama31 v3_new_r20_kd100
