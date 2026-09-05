#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${OURS_ST_RUN_ROOT:-/data3/seonghyeonnoh/LLM-continual-learning-runs/local/trace/results/full_runs/llama31}"
OUTPUT="${RUN_ROOT}/ours_lora_moe_v3_st_top1_fixed1000_epochprobe64"

if [[ -e "${OUTPUT}/0/lora_moe_meta.json" && -z "${OURS_LORAMOE_RESUME_CHECKPOINT:-}" ]]; then
  echo "[ERROR] refusing to overwrite existing run: ${OUTPUT}" >&2
  exit 2
fi
mkdir -p "${OUTPUT}"

export PYTHONNOUSERSITE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export OURS_LORAMOE_OUTPUT_ROOT="${OUTPUT}"
export OURS_LORAMOE_GPUS=0,1,2,3,4,5,6,7
export OURS_LORAMOE_PORT=29731
export OURS_LORAMOE_MICRO_BATCH=8
export OURS_LORAMOE_GRAD_ACCUM=1
export OURS_LORAMOE_TOP_K=1
export OURS_LORAMOE_ROUTING_WEIGHT_MODE=straight_through_topk
export OURS_V2_JOINT_NEW_TO_REPLAY_RATIO=5
export OURS_ROUTER_REPLAY_EXPOSURE_SAMPLES=1000
export OURS_V2_KD_MEMORY_BATCH_SIZE=8
export OURS_V3_EPOCH_PROBE_SAMPLES=64
export OURS_REPLAY_SELECTION_MODE="${OURS_REPLAY_SELECTION_MODE:-random}"

echo "[TRAIN START] V3 GPUs=0-7 micro=8 accum=1 global=64 routing=straight_through_topk"
exec bash "${ROOT}/scripts/baselines/_run_ours_lora_moe.sh" train llama31 v3
