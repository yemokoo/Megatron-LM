#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Router-memory early-stop variant:
# - Code LM training budget stays unchanged.
# - Wiki memory KL is applied early, then stopped when fixed-probe KL rises.
# - Checkpoints are saved every 300 Code steps for post-hoc analysis.
export CODE_RUN_SUFFIX="${CODE_RUN_SUFFIX:--router-memory-fixed5-kl0p1-earlystop}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-300}"
export ROUTER_MEMORY_KL_COEFF="${ROUTER_MEMORY_KL_COEFF:-0.1}"
export ROUTER_MEMORY_FRACTION="${ROUTER_MEMORY_FRACTION:-0.05}"
export ROUTER_MEMORY_INTERVAL="${ROUTER_MEMORY_INTERVAL:-20}"
export ROUTER_MEMORY_EVAL_INTERVAL="${ROUTER_MEMORY_EVAL_INTERVAL:-0}"
export ROUTER_MEMORY_EVAL_ITERS="${ROUTER_MEMORY_EVAL_ITERS:-1}"
export ROUTER_KL_EARLY_STOP_ENABLED="${ROUTER_KL_EARLY_STOP_ENABLED:-1}"
export ROUTER_KL_EARLY_STOP_METRIC="${ROUTER_KL_EARLY_STOP_METRIC:-fixed_probe_kl}"
export ROUTER_KL_WARMUP_STEPS="${ROUTER_KL_WARMUP_STEPS:-300}"
export ROUTER_KL_PATIENCE="${ROUTER_KL_PATIENCE:-3}"
export ROUTER_KL_MIN_DELTA="${ROUTER_KL_MIN_DELTA:-0.01}"
export ROUTER_KL_SMOOTHING_WINDOW="${ROUTER_KL_SMOOTHING_WINDOW:-3}"
export CODE_MICRO_BATCH_SIZE_G4="${CODE_MICRO_BATCH_SIZE_G4:-32}"
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"
export ATTN_LORA_GROUPED_GEMM="${ATTN_LORA_GROUPED_GEMM:-1}"
export MOE_PERMUTE_FUSION="${MOE_PERMUTE_FUSION:-0}"
export MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}"

exec bash "$SCRIPT_DIR/offline_chain_shared_router_granularity_qkvo_code_only_mha.sh"
