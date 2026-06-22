#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-72}"
export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export G2_ROOT="${G2_ROOT:-$PROJECT_ROOT/.local/weights/a100/mha/g2-checkpoints}"

export RUN_ID="${RUN_ID:-g2-exp3-aux0-z0-cumulative80-oldfreeze-e8to16-ffn352-r256-wiki-to-code-qkvo-mha-a100-bf16-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"
export STAGE1_WEIGHTS_DIR="${STAGE1_WEIGHTS_DIR:-$G2_ROOT/wiki_aux0/0620-g2-exp3-aux0-z0-wiki-shared-router-qkvo-mha-a100-bf16-mb72-1800}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$G2_ROOT/code/phase1/$RUN_ID}"
export SHARED_ROUTER_HYBRID_PARTIAL_FREEZE_MASK="${SHARED_ROUTER_HYBRID_PARTIAL_FREEZE_MASK:-$PROJECT_ROOT/analysis_outputs/g2_aux0_z0_wiki_router_softmax_importance_1m_cumulative80/router_softmax_cumulative80_selection.json}"

export PARTIAL_FREEZE_SELECTION_LABEL="${PARTIAL_FREEZE_SELECTION_LABEL:-layer-wise cumulative 80% old/wiki experts from aux0,z0 wiki pre-TopK router softmax}"

# Keep the code phase matched to the aux0,z0 wiki pretraining condition.
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.0}"
export MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.0}"

export WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2 - exp3 aux0 z0 cumulative80 old expert freeze - wiki to code}"

exec bash "$SCRIPT_DIR/run_g2_shared_router_partial_old_top4_freeze_code_mha.sh"
