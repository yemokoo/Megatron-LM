#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${OURS_ST_RUN_ROOT:-/data3/seonghyeonnoh/LLM-continual-learning-runs/local/trace/results/full_runs/llama31}"
V2_OUT="${RUN_ROOT}/ours_lora_moe_v2_st_top1_fixed1000"
V3_OUT="${RUN_ROOT}/ours_lora_moe_v3_st_top1_fixed1000_epochprobe64"

for output in "${V2_OUT}" "${V3_OUT}"; do
  if [[ -e "${output}/0/lora_moe_meta.json" ]]; then
    echo "[ERROR] refusing to overwrite existing run: ${output}" >&2
    exit 2
  fi
  mkdir -p "${output}"
done

export PYTHONNOUSERSITE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

echo "[TRAIN START] V2 GPUs=0,1,2,3 micro=16 accum=1 global=64"
(
  export OURS_LORAMOE_OUTPUT_ROOT="${V2_OUT}"
  export OURS_LORAMOE_GPUS=0,1,2,3
  export OURS_LORAMOE_PORT=29721
  export OURS_LORAMOE_MICRO_BATCH=16
  export OURS_LORAMOE_GRAD_ACCUM=1
  export OURS_LORAMOE_ROUTING_WEIGHT_MODE=straight_through_topk
  export OURS_V2_JOINT_NEW_TO_REPLAY_RATIO=5
  export OURS_ROUTER_REPLAY_EXPOSURE_SAMPLES=1000
  bash "${ROOT}/scripts/baselines/_run_ours_lora_moe.sh" train llama31 v2
) >"${V2_OUT}/launcher.log" 2>&1 &
v2_pid=$!

echo "[TRAIN START] V3 GPUs=4,5,6,7 micro=8 accum=2 global=64"
(
  export OURS_LORAMOE_OUTPUT_ROOT="${V3_OUT}"
  export OURS_LORAMOE_GPUS=4,5,6,7
  export OURS_LORAMOE_PORT=29731
  export OURS_LORAMOE_MICRO_BATCH=8
  export OURS_LORAMOE_GRAD_ACCUM=2
  export OURS_LORAMOE_ROUTING_WEIGHT_MODE=straight_through_topk
  export OURS_V2_JOINT_NEW_TO_REPLAY_RATIO=5
  export OURS_ROUTER_REPLAY_EXPOSURE_SAMPLES=1000
  export OURS_V3_EPOCH_PROBE_SAMPLES=64
  bash "${ROOT}/scripts/baselines/_run_ours_lora_moe.sh" train llama31 v3
) >"${V3_OUT}/launcher.log" 2>&1 &
v3_pid=$!

status=0
wait "${v2_pid}" || status=1
wait "${v3_pid}" || status=1
if [[ "${status}" -ne 0 ]]; then
  echo "[ERROR] V2 or V3 training failed" >&2
  exit 1
fi
echo "[TRAIN COMPLETE] V2 and V3"

