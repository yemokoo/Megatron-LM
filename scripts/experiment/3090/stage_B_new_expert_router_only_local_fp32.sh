#!/bin/bash
set -euo pipefail

# Task B after Task A, new-expert/router-only variant: continue on Task B while
# freezing every parameter except the newly added experts and the trainable
# portion of the expanded router.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export RUN_ID="${RUN_ID:-stage-b-new-expert-router-only-local-fp32-$(date -u +%Y%m%d-%H%M%S)}"
export TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY="${TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY:-1}"
export STAGE_B_LABEL="${STAGE_B_LABEL:-Stage B new-expert-router-only}"
export STAGE_B_STAGE_NAME="${STAGE_B_STAGE_NAME:-B_new_expert_router_only}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$PROJECT_ROOT/.local/weights/continual-stage-A-to-B-new-only/$RUN_ID}"
export LOG_DIR="${LOG_DIR:-$TRAIN_WEIGHTS/logs}"
export RUN_LOG="${RUN_LOG:-$LOG_DIR/stage_b_new_only.log}"
export GPU_LOG="${GPU_LOG:-$LOG_DIR/gpu_usage.csv}"
export RUN_METADATA="${RUN_METADATA:-$LOG_DIR/run_metadata.json}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-10}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-$RUN_ID}"
export WANDB_SAVE_DIR="${WANDB_SAVE_DIR:-$TRAIN_WEIGHTS/wandb}"
export CONTINUAL_PLOT_PREFIX="${CONTINUAL_PLOT_PREFIX:-$LOG_DIR/task_a_probe_continual}"

exec bash "$PROJECT_ROOT/scripts/experiment/3090/stage_B_local_fp32.sh"
