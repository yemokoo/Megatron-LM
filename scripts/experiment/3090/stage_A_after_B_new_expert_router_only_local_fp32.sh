#!/bin/bash
set -euo pipefail

# Task A after Task B, new-expert/router-only variant: continue on Task A while
# freezing every parameter except the newly added experts and the trainable
# portion of the expanded router.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export RUN_ID="${RUN_ID:-stage-a-after-b-new-expert-router-only-local-fp32-$(date -u +%Y%m%d-%H%M%S)}"
export MASTER_PORT="${MASTER_PORT:-29514}"
export TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY="${TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY:-1}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$PROJECT_ROOT/.local/weights/continual-stage-B-to-A-new-only/$RUN_ID}"
export LOG_DIR="${LOG_DIR:-$TRAIN_WEIGHTS/logs}"
export RUN_LOG="${RUN_LOG:-$LOG_DIR/stage_a_after_b_new_only.log}"
export GPU_LOG="${GPU_LOG:-$LOG_DIR/gpu_usage.csv}"
export RUN_METADATA="${RUN_METADATA:-$LOG_DIR/run_metadata.json}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-10}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-10}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-$RUN_ID}"
export WANDB_SAVE_DIR="${WANDB_SAVE_DIR:-$TRAIN_WEIGHTS/wandb}"
export CONTINUAL_PLOT_PREFIX="${CONTINUAL_PLOT_PREFIX:-$LOG_DIR/task_b_probe_continual}"

exec bash "$PROJECT_ROOT/scripts/experiment/3090/stage_A_after_B_local_fp32.sh"
