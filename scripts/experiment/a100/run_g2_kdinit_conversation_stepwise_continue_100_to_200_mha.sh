#!/bin/bash
set -euo pipefail

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/../../.." && pwd)"
cd "$R"

: "${WANDB_PROJECT:?WANDB_PROJECT must be set (source ~/.config/wandb/env first)}"

BASE_RUN_ID=g2-kdinit-conv-stepwise-reload-baseline-v7-5400-5500-mb96
CONT_RUN_ID=g2-kdinit-conv-stepwise-continue-v8-5501-5600-mb96

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE=4
export WANDB_MODE="${WANDB_MODE:-online}"

export SOURCE_WEIGHTS_DIR="$R/.local/weights/a100/mha/g2-checkpoints/conversation/stepwise_diagnostic/$BASE_RUN_ID"
export SOURCE_REQUIRED_ITERS=100
export RUN_ID="$CONT_RUN_ID"
export TRAIN_WEIGHTS="$R/.local/weights/a100/mha/g2-checkpoints/conversation/stepwise_diagnostic/$CONT_RUN_ID"
export LOG_DIR="$TRAIN_WEIGHTS/logs"

# Append metrics to the original W&B history while writing model checkpoints to
# a separate continuation directory.
export WANDB_RUN_ID="$BASE_RUN_ID"
export WANDB_EXP_NAME="$BASE_RUN_ID"
export WANDB_RESUME=allow

export TRAIN_ITERS=100
export MICRO_BATCH_SIZE=96
export GLOBAL_BATCH_SIZE=2304

export PROBE_NAME=conversation_probe
export SECONDARY_PROBE_NAME=code_probe
export TERTIARY_PROBE_NAME=wiki_probe
export TERTIARY_PROBE_DATASET="$R/data/wiki/test"
export PROBE_EVAL_INTERVAL=1
export SECONDARY_PROBE_EVAL_INTERVAL=1
export TERTIARY_PROBE_EVAL_INTERVAL=1
export PROBE_EVAL_ITERS=25
export SECONDARY_PROBE_EVAL_ITERS=25
export TERTIARY_PROBE_EVAL_ITERS=25
export PROBE_STEP_OFFSET=5500
export SECONDARY_PROBE_STEP_OFFSET=5500
export TERTIARY_PROBE_STEP_OFFSET=5500
export WANDB_STEP_OFFSET=5500

export RUN_INITIAL_PROBE_EVAL=0
export RUN_INITIAL_VALID_EVAL=0
export LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND=0
export RESUME_CONTINUAL_FROM_TRAIN_WEIGHTS=0

export LR_DECAY_ITERS=1800
export LR_WSD_DECAY_ITERS=180
export LR_WARMUP_FRACTION=0.01
export SAVE_INTERVAL=100
export EVAL_INTERVAL=100
export LOG_INTERVAL=1
export MASTER_PORT="${MASTER_PORT:-29985}"

if [ -e "$TRAIN_WEIGHTS/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: continuation output already contains a checkpoint: $TRAIN_WEIGHTS" >&2
    exit 1
fi

exec bash "$D/run_g2_ffn_only_conversation_wikicode_joint_lm_allrouter_mha.sh"
