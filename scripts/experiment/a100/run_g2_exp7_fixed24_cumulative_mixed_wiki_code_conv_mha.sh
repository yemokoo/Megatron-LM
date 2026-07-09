#!/bin/bash
set -euo pipefail

# Experiment 7: fixed24 MoE cumulative-mixed CL upper bound.
#
# Same fixed24 FFN-only MoE architecture as sequential exp 2 / mixed exp 5:
#   layer 1      : dense FFN 5472
#   layers 2-9   : MoE, 24 experts x 352, top-4  (active FFN = 4*352 = 1408)
# but trained as a 3-stage cumulative-replay chain (wiki -> wiki+code ->
# wiki+code+conv), 1800/3600/5400 iters. CL-internal ceiling for the MoE model:
# full replay at every stage, no forgetting, "1800 steps/dataset".
#
# Needs grouped_gemm (or set MOE_GROUPED_GEMM=0 for SequentialMLP). No KD (pure
# replay). 4-GPU DP; top-4 dispatch is heavier than dense so mb defaults to 72.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- fixed24 FFN-only MoE architecture ---
export VARIANT_TAG="${VARIANT_TAG:-fixed24}"
export VARIANT_LABEL="${VARIANT_LABEL:-fixed24 ffn-only e24 moe352 top4}"
export NUM_EXPERTS="${NUM_EXPERTS:-24}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-5472}"
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.01}"
export MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.001}"
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"

# 4-GPU DP (global batch 2304 -> 576 samples/GPU).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-72}"

export STAGE1_MASTER_PORT="${STAGE1_MASTER_PORT:-29911}"
export STAGE2_MASTER_PORT="${STAGE2_MASTER_PORT:-29912}"
export STAGE3_MASTER_PORT="${STAGE3_MASTER_PORT:-29913}"

exec bash "$SCRIPT_DIR/run_g2_cumulative_mixed_chain_mha.sh"
