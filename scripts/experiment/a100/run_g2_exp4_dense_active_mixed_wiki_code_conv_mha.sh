#!/bin/bash
set -euo pipefail

# Experiment 4: dense(active) joint/mixed upper-bound.
#
# Same dense(active) architecture as the sequential exp 1
#   layer 1     : dense FFN 5472
#   layers 2-9  : dense FFN 1408  (realized as num_experts=1, topk=1)
# but wiki+code+conversation are randomly blended 1:1:1 and trained jointly in a
# single 5400-step stage (= sequential 3 x 1800). This is the performance ceiling
# (no forgetting) for the dense(active) model.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export RUN_ID="${RUN_ID:-g2-dense-active-matched-e1-ffn1408-top1-mixed-wiki-code-conv-mha-a100-bf16-mb${MICRO_BATCH_SIZE:-72}-5400}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-configs/model/flame-moe-ffn-only-no-shared.sh}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-5472}"
export NUM_EXPERTS="${NUM_EXPERTS:-1}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-1}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-1408}"
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.0}"
export MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.0}"
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-0}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-72}"
export TRAIN_ITERS="${TRAIN_ITERS:-5400}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$PROJECT_ROOT/.local/weights/a100/mha/mixed-3way/dense-active/$RUN_ID}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2 dense(active) mixed wiki+code+conv (upper bound)}"
export MASTER_PORT="${MASTER_PORT:-29561}"

exec bash "$SCRIPT_DIR/pretrain_mixed_3way_local_bf16.sh"
