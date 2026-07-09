#!/bin/bash
set -euo pipefail

# Experiment 6: dense(active) cumulative-mixed CL upper bound.
#
# Same dense(active) architecture as sequential exp 1 / mixed exp 4:
#   layer 1      : dense FFN 5472
#   layers 2-9   : 1-expert top-1 MoE, moe_ffn=1408  (== monolithic dense 1408)
# but trained as a 3-stage cumulative-replay chain (wiki -> wiki+code ->
# wiki+code+conv), 1800/3600/5400 iters. This is the CL-internal ceiling for the
# dense model: full replay at every stage, no forgetting, "1800 steps/dataset".
#
# 1-expert => grouped_gemm brings no benefit; MOE_GROUPED_GEMM=0 (no dependency).
# No KD (pure replay). 4-GPU DP; dense(active) is light so mb defaults to 96.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- dense(active) architecture: 1-expert top-1 in layers 2-9 ---
export VARIANT_TAG="${VARIANT_TAG:-dense-active}"
export VARIANT_LABEL="${VARIANT_LABEL:-dense(active) layer1-5472 layers2to9-dense1408}"
export NUM_EXPERTS="${NUM_EXPERTS:-1}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-1}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-1408}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-5472}"
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.0}"
export MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.0}"
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-0}"

# 4-GPU DP (global batch 2304 -> 576 samples/GPU); dense(active) fits a large mb.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-96}"

export STAGE1_MASTER_PORT="${STAGE1_MASTER_PORT:-29901}"
export STAGE2_MASTER_PORT="${STAGE2_MASTER_PORT:-29902}"
export STAGE3_MASTER_PORT="${STAGE3_MASTER_PORT:-29903}"

exec bash "$SCRIPT_DIR/run_g2_cumulative_mixed_chain_mha.sh"
