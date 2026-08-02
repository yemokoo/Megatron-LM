#!/bin/bash
set -euo pipefail

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/../../.." && pwd)"
cd "$R"

: "${WANDB_PROJECT:?WANDB_PROJECT must be set (source ~/.config/wandb/env first)}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE=4
export WANDB_MODE=online

export RUN_ID="${RUN_ID:-g2-kdinit-conv-newexpert-lr-ramp0to1-probemb32-v4-5400-5600-mb96}"
export WANDB_RUN_ID="$RUN_ID"
export WANDB_EXP_NAME="$RUN_ID"
export WANDB_RESUME=never

# Use the unchanged, already-expanded 16->24 expert checkpoint produced by
# Code training followed by Conversation-expert KD initialization.
export SOURCE_WEIGHTS_DIR="$R/.local/weights/a100/mha/g2-checkpoints/conversation/expansion_distill_init_joint_code/g2-ffn-only-e16to24-conv-init-from-joint-code-logits-wikicode-distill-mb32-600"
export SOURCE_REQUIRED_ITERS=600
export LOAD_EXPANDED_SOURCE=1

export TRAIN_WEIGHTS="$R/.local/weights/a100/mha/g2-checkpoints/conversation/stepwise_diagnostic/$RUN_ID"
export LOG_DIR="$TRAIN_WEIGHTS/logs"

export TRAIN_ITERS=200
export MICRO_BATCH_SIZE=96
export GLOBAL_BATCH_SIZE=2304

# Update 1 uses 0x new-expert LR; update 200 uses the normal scheduled LR.
# Router LR and the base Megatron LR schedule are unchanged.
export MOE_NEW_EXPERT_LR_RAMP_STEPS=200

export PROBE_NAME=conversation_probe
export SECONDARY_PROBE_NAME=code_probe
export TERTIARY_PROBE_NAME=wiki_probe
export TERTIARY_PROBE_DATASET="$R/data/wiki/test"
export PROBE_EVAL_INTERVAL=2
export SECONDARY_PROBE_EVAL_INTERVAL=2
export TERTIARY_PROBE_EVAL_INTERVAL=2
export PROBE_EVAL_ITERS=25
export PROBE_MICRO_BATCH_SIZE=32
export SECONDARY_PROBE_EVAL_ITERS=25
export TERTIARY_PROBE_EVAL_ITERS=25
export PROBE_STEP_OFFSET=5400
export SECONDARY_PROBE_STEP_OFFSET=5400
export TERTIARY_PROBE_STEP_OFFSET=5400
export WANDB_STEP_OFFSET=5400

# Record the untouched KD checkpoint at display step 5400 before update 1.
export RUN_INITIAL_PROBE_EVAL=1
export RUN_INITIAL_VALID_EVAL=0
export LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND=0
export RESUME_CONTINUAL_FROM_TRAIN_WEIGHTS=0

# Preserve the baseline run's base LR schedule; only the new-expert multiplier
# is ramped by this experiment.
export LR_DECAY_ITERS=1800
export LR_WSD_DECAY_ITERS=180
export LR_WARMUP_FRACTION=0.01
export SAVE_INTERVAL=200
export EVAL_INTERVAL=200
export LOG_INTERVAL=2
export TENSORBOARD_LOG_INTERVAL=2
export MASTER_PORT="${MASTER_PORT:-29986}"

if [ -e "$TRAIN_WEIGHTS/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: output already contains a checkpoint: $TRAIN_WEIGHTS" >&2
    exit 1
fi

mkdir -p "$TRAIN_WEIGHTS/wandb"

exec bash "$D/run_g2_ffn_only_conversation_wikicode_joint_lm_allrouter_mha.sh"
