#!/usr/bin/env bash
set -euo pipefail

# Conversation task learning from an already-expanded 24-expert KD-init.
# This stage contains Conversation LM updates only.  Router-only Wiki+Code+
# Conversation retuning is deliberately performed by the following stage.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

FLAME_ENV="${FLAME_ENV:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100}"
[[ -x "$FLAME_ENV/bin/python" ]] || { echo "[ERROR] H100 FLAME environment missing: $FLAME_ENV" >&2; exit 1; }
export PATH="$FLAME_ENV/bin:/usr/bin:/bin"
export PYTHON_BIN="$FLAME_ENV/bin/python"
export PYTHONNOUSERSITE=1
export CUDA_HOME="${CUDA_HOME:-$FLAME_ENV}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

STUDY_ROOT="${STUDY_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/fingerprint_router_geometry_20260809}"
CONV_KD_WEIGHTS="${CONV_KD_WEIGHTS:?set CONV_KD_WEIGHTS to the completed 24E Conversation KD-init}"
CONV_KD_STEPS="${CONV_KD_STEPS:-600}"
CONV_WEIGHTS="${CONV_WEIGHTS:-$STUDY_ROOT/checkpoints/Conversation_only_from_KDinit_mb48_gbs2304_step1800}"
FLAME_DATA_ROOT="${FLAME_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup}"

tracker="$CONV_KD_WEIGHTS/latest_checkpointed_iteration.txt"
if [[ ! -f "$tracker" ]] || [[ "$(tr -d '[:space:]' < "$tracker")" != "$CONV_KD_STEPS" ]]; then
    echo "[ERROR] expected Conversation KD-init step $CONV_KD_STEPS: $CONV_KD_WEIGHTS" >&2
    exit 1
fi

export STUDY_ROOT CONV_KD_WEIGHTS CONV_WEIGHTS FLAME_DATA_ROOT
export SOURCE_TASK=code
export TARGET_TASK=conversation
export SOURCE_WEIGHTS_DIR="$CONV_KD_WEIGHTS"
export SOURCE_REQUIRED_ITERS="$CONV_KD_STEPS"
export SOURCE_NUM_EXPERTS=16
export NUM_EXPERTS=24
export MOE_ROUTER_TOPK=4
export MOE_FFN_HIDDEN_SIZE=352
export NUM_QUERY_GROUPS=16
export MODEL_CONFIG_SCRIPT=scripts/experiment/a100/flame-moe-bf16-no-shared.sh
export LOAD_EXPANDED_SOURCE=1
export FREEZE_SHARED=1
export TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY=1
export TRAIN_ATTENTION_WITH_NEW_EXPERTS=0
export FREEZE_DENSE_ATTENTION_LORA_WITH_NEW_EXPERTS=1

export TRAIN_WEIGHTS="$CONV_WEIGHTS"
export RUN_ID="Conversation-only-from-KDinit-mb48-gbs2304-step1800"
export STAGE_DIR_NAME="fingerprint-router-geometry/conversation-task"
export WANDB_EXP_NAME="Conversation only | from WikiCode KD-init | 1800"
export WANDB_RUN_ID="fingerprint-conversation-only-1800-20260809"
export WANDB_MODE="${WANDB_MODE:-offline}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_PORT="${MASTER_PORT:-33904}"
export MICRO_BATCH_SIZE=48
export GLOBAL_BATCH_SIZE=2304
export TRAIN_ITERS=1800
export LR=3e-4
export MIN_LR=3e-5
export LR_DECAY_STYLE=WSD
export LR_DECAY_ITERS=1800
export LR_WSD_DECAY_ITERS=180
export LR_WARMUP_FRACTION=0.01
export SAVE_INTERVAL="${SAVE_INTERVAL:-300}"
export RECOVERY_SAVE_INTERVAL="${RECOVERY_SAVE_INTERVAL:-50}"
export EVAL_INTERVAL=1800
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export DIRECT_LOCAL_SAVE=1
export RESUME_CONTINUAL_FROM_TRAIN_WEIGHTS=1
export NO_SAVE_OPTIM=0

# Only the normal Conversation LM objective is active in this stage.
export ENABLE_OLD_MODEL_KL=0
export OLD_MODEL_KL_COEFF=0.0
export MOE_EXPANSION_DISTILL_MODE=none
export MOE_JOINT_REPLAY_LM=0
export MOE_JOINT_REPLAY_OLD_DATA_KD=0
export MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=0
export MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=0
export MOE_INTERLEAVE_CODE_STEPS=0
export MOE_INTERLEAVE_ROUTER_STEPS=0
export TRAIN_DATASET_SECONDARY=""
export JOINT_REPLAY_DATASET=""
export JOINT_REPLAY_SECONDARY_DATASET=""
export LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND=0

export RUN_INITIAL_PROBE_EVAL=1
export RUN_INITIAL_VALID_EVAL=1
export PROBE_EVAL_INTERVAL=100
export SECONDARY_PROBE_EVAL_INTERVAL=100
export TERTIARY_PROBE_DATASET="$FLAME_DATA_ROOT/wiki/test"
export TERTIARY_PROBE_NAME=wiki_probe
export TERTIARY_PROBE_EVAL_INTERVAL=100

echo "[CONVERSATION CONFIG] source KD-init: $CONV_KD_WEIGHTS"
echo "[CONVERSATION CONFIG] output:         $CONV_WEIGHTS"
echo "[CONVERSATION CONFIG] train: Conversation LM only, 1800 steps"
echo "[CONVERSATION CONFIG] trainable: experts 16:24 + router rows 16:24 only"
echo "[CONVERSATION CONFIG] disabled: replay, KD/hidden loss, router FT"

exec bash "$SCRIPT_DIR/run_continual_moe_a100_bf16.sh"
