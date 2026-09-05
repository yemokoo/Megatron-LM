#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${1:?usage: run_ours_sparse15_parallel.sh <model> <v2|v3> <run_dir> <gpu_csv>}"
VERSION="${2:?usage: run_ours_sparse15_parallel.sh <model> <v2|v3> <run_dir> <gpu_csv>}"
RUN_DIR="${3:?usage: run_ours_sparse15_parallel.sh <model> <v2|v3> <run_dir> <gpu_csv>}"
GPU_CSV="${4:?usage: run_ours_sparse15_parallel.sh <model> <v2|v3> <run_dir> <gpu_csv>}"
IFS=',' read -r -a GPUS <<< "${GPU_CSV}"
SHARDS="${#GPUS[@]}"
PIDS=()

for shard in "${!GPUS[@]}"; do
  gpu="${GPUS[$shard]}"
  (
    export OURS_LORAMOE_OUTPUT_ROOT="${RUN_DIR}"
    export OURS_LORAMOE_GPUS="${gpu}"
    export OURS_EVAL_SPARSE_15=1
    export EVAL_SHARD_COUNT="${SHARDS}"
    export EVAL_SHARD_INDEX="${shard}"
    export EVAL_SKIP_COLLECT=1
    "${ROOT}/scripts/baselines/_run_ours_lora_moe.sh" \
      eval "${MODEL}" "${VERSION}"
  ) > "${RUN_DIR}/eval_sparse15_shard_${shard}.launcher.log" 2>&1 &
  PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || failed=1
done
if [[ "${failed}" -ne 0 ]]; then
  echo "[ERROR] at least one sparse-15 evaluation shard failed" >&2
  exit 1
fi

python3 "${ROOT}/scripts/collect_results.py" \
  --method "ours_lora_moe_${VERSION}" --model "${MODEL}" \
  --run-dir "${RUN_DIR}" --family paper_baseline --sparse-15 \
  --output "${RUN_DIR}/sparse15_summary.json"
