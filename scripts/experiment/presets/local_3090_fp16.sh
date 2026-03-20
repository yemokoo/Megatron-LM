#!/bin/bash
set -euo pipefail

# Conservative fp16 preset for RTX 3090-class local runs.
# This keeps the FLAME-MoE structure but shrinks the local model so optimizer
# updates can succeed on fp16-only consumer GPUs.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export PRECISION="${PRECISION:-fp16}"

# Use the older 38M-scale local architecture as the default 3090-safe model.
export NUM_LAYERS="${NUM_LAYERS:-9}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-1368}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-176}"
export MOE_LAYER_FREQ="${MOE_LAYER_FREQ:-[0]*1+[1]*8}"
export NUM_EXPERTS="${NUM_EXPERTS:-4}"
export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-4}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-2}"

export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export SEQ_LENGTH="${SEQ_LENGTH:-512}"

# Keep local fp16 numerically conservative.
export INITIAL_LOSS_SCALE="${INITIAL_LOSS_SCALE:-128}"
export STATIC_LOSS_SCALE="${STATIC_LOSS_SCALE:-1}"
export DISABLE_NAN_CHECKS="${DISABLE_NAN_CHECKS:-1}"
export INIT_METHOD_STD="${INIT_METHOD_STD:-0.01}"
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.001}"
export MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.0001}"

export LR="${LR:-3e-4}"
export MIN_LR="${MIN_LR:-3e-5}"
export LR_DECAY_STYLE="${LR_DECAY_STYLE:-WSD}"
export LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.01}"

export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-configs/model/flame-moe.sh}"
