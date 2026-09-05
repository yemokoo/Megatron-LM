#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_UNIT="${V3_TOP4_TRAIN_UNIT:-trace-instruct-v3-new-top4-gpu4567-20260813.service}"
OUTPUT="${V3_TOP4_OUTPUT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/v3/instruct_priority_fourway_20260812/v3_new_top4}"
LOG="${V3_TOP4_LOWER_LOG:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/logs/instruct_priority_fourway_20260812/v3_new_top4_lower_triangle_gpu4567.log}"
mkdir -p "$(dirname "${LOG}")"

echo "[WAIT] $(date --iso-8601=seconds) unit=${TRAIN_UNIT}" | tee -a "${LOG}"
while systemctl --user is-active --quiet "${TRAIN_UNIT}"; do
  checkpoint_count="$(find "${OUTPUT}" -mindepth 2 -maxdepth 2 -name lora_moe_meta.json 2>/dev/null | wc -l)"
  echo "[WAIT] $(date --iso-8601=seconds) completed_checkpoints=${checkpoint_count}/8" | tee -a "${LOG}"
  sleep 60
done

for round in 0 1 2 3 4 5 6 7; do
  [[ -s "${OUTPUT}/${round}/lora_moe_meta.json" && \
     -s "${OUTPUT}/${round}/pytorch_model.bin" ]] || {
    echo "[ERROR] top4 training stopped without complete round ${round}: ${OUTPUT}" | tee -a "${LOG}" >&2
    exit 3
  }
done

echo "[TRAIN COMPLETE] $(date --iso-8601=seconds); starting full lower triangle on GPUs 4,5,6,7" | tee -a "${LOG}"
bash "${ROOT}/scripts/run_ours_lower_triangle_optimized.sh" \
  "${OUTPUT}" ours_lora_moe_v3_new_top4 4,5,6,7 "${LOG}"
echo "[GPU 4-7 RELEASED] $(date --iso-8601=seconds)" | tee -a "${LOG}"
