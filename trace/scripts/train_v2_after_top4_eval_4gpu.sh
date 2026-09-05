#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_UNIT="${TOP4_EVAL_UNIT:-trace-v2-new-top4-eval-resume-gpu0123-20260813.service}"
TOP4_RUN="${INSTRUCT_TOP4_OUT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/instruct_priority_fourway_20260812/v2_new_top4}"
TOP4_SUMMARY="${TOP4_RUN}/sparse15_summary.json"
MODEL_PATH="${INSTRUCT_MODEL_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}"
CACHE_ROOT="${INSTRUCT_TOKEN_CACHE:-/data2/seonghyeonnoh/LLM-continual-learning-caches/llama31_8b_instruct/slora_chat_full_len1024}"
RUN_ROOT="${INSTRUCT_RUN_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace}"
EXPERIMENT_NAME="${INSTRUCT_EXPERIMENT_NAME:-instruct_priority_fourway_20260812}"
OUTPUT="${INSTRUCT_V2_OUT:-${RUN_ROOT}/v2/${EXPERIMENT_NAME}/v2_baseline}"
GPUS="0,1,2,3"
POLL_SECONDS="${POLL_SECONDS:-15}"

echo "[WAIT START] $(date --iso-8601=seconds) unit=${EVAL_UNIT} summary=${TOP4_SUMMARY}"
while [[ "$(systemctl --user is-active "${EVAL_UNIT}" 2>/dev/null || true)" == "active" ]]; do
  sleep "${POLL_SECONDS}"
done

# collect_results writes this only after every sparse-15 cell has completed.
python3 - "${TOP4_SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"top4 evaluation did not complete: missing {path}")
payload = json.loads(path.read_text())
if len(payload.get("tasks", [])) != 8 or "final_average" not in payload:
    raise SystemExit(f"invalid sparse-15 summary: {path}")
print(f"[EVAL VERIFIED] AA={payload['final_average']:.6f} BWT={payload['BWT']:.6f}")
PY

# Do not race a just-exited evaluator's CUDA teardown or another GPU user.
while ! nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | sed -n '1,4p' | awk '$1 >= 1024 {busy=1} END {exit busy ? 1 : 0}'; do
  echo "[GPU WAIT] $(date --iso-8601=seconds) waiting for GPUs 0-3 to become free"
  sleep "${POLL_SECONDS}"
done

# The classified TRACE path may be a deliberate symlink to the legacy physical
# run root. Materialize that target without replacing the symlink.
if [[ -L "${OUTPUT}" && ! -e "${OUTPUT}" ]]; then
  target="$(readlink "${OUTPUT}")"
  mkdir -p "${target}"
fi
if [[ -d "${OUTPUT}" ]] && find "${OUTPUT}" -mindepth 1 -print -quit | grep -q .; then
  echo "[ERROR] refusing to overwrite nonempty V2 output: ${OUTPUT}" >&2
  exit 2
fi
mkdir -p "${OUTPUT}"

echo "[TRAIN START] $(date --iso-8601=seconds) version=v2 gpus=${GPUS} micro=16 ga=1 gbs=64 kd_batch=8 output=${OUTPUT}"
export SLORA_LLAMA31_PATH="${MODEL_PATH}"
export OURS_LLAMA31_TOKEN_CACHE="${CACHE_ROOT}"
export OURS_LORAMOE_OUTPUT_ROOT="${OUTPUT}"
export OURS_LORAMOE_GPUS="${GPUS}"
export OURS_LORAMOE_PORT="${OURS_LORAMOE_PORT:-29833}"
export OURS_LORAMOE_MICRO_BATCH=16
export OURS_LORAMOE_GRAD_ACCUM=1
export OURS_V2_KD_MEMORY_BATCH_SIZE="${OURS_V2_KD_MEMORY_BATCH_SIZE:-8}"
export OURS_V2_REPLAY_FORWARD_BATCH_SIZE="${OURS_V2_REPLAY_FORWARD_BATCH_SIZE:-8}"
export OURS_LORAMOE_SEED=2025
export OURS_REPLAY_SUBSET_SEED=2025
export OURS_REPLAY_SELECTION_MODE=random
export PYTHONNOUSERSITE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

exec bash "${ROOT}/scripts/baselines/_run_ours_lora_moe.sh" train llama31 v2
