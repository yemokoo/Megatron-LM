#!/bin/bash
set -euo pipefail

# Experiment 5: fixed24 MoE joint/mixed upper-bound.
#
# Same fixed24 FFN-only MoE architecture as the sequential exp 2
#   layer 1     : dense FFN 5472
#   layers 2-9  : MoE, 24 experts x 352, top-4
# but wiki+code+conversation are randomly blended 1:1:1 and trained jointly in a
# single 5400-step stage (= sequential 3 x 1800). Performance ceiling (no
# forgetting) for the fixed24 MoE model.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export RUN_ID="${RUN_ID:-g2-fixed24-e24-ffn352-top4-mixed-wiki-code-conv-mha-a100-bf16-mb${MICRO_BATCH_SIZE:-48}-5400}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-configs/model/flame-moe-ffn-only-no-shared.sh}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-5472}"
export NUM_EXPERTS="${NUM_EXPERTS:-24}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.01}"
export MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.001}"
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-48}"
export TRAIN_ITERS="${TRAIN_ITERS:-5400}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$PROJECT_ROOT/.local/weights/a100/mha/mixed-3way/fixed24/$RUN_ID}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2 fixed24 MoE mixed wiki+code+conv (upper bound)}"
export MASTER_PORT="${MASTER_PORT:-29562}"

exec bash "$SCRIPT_DIR/pretrain_mixed_3way_local_bf16.sh"
