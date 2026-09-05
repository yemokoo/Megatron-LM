#!/usr/bin/env bash
set -euo pipefail

# Continue the Instruct TRACE comparison after the already-running V3 eval.
# This script intentionally hard-pins every remaining training/evaluation job to
# physical GPUs 0,1,2,3 and preserves the original effective global batch 64.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${INSTRUCT_MODEL_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}"
CACHE_ROOT="${INSTRUCT_TOKEN_CACHE:-/data2/seonghyeonnoh/LLM-continual-learning-caches/llama31_8b_instruct/slora_chat_full_len1024}"
DATA_ROOT="${INSTRUCT_DATA_ROOT:-${ROOT}/data/trace}"
RUN_ROOT="${INSTRUCT_RUN_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace}"
EXPERIMENT_NAME="${INSTRUCT_EXPERIMENT_NAME:-instruct_priority_fourway_20260812}"
GPUS="0,1,2,3"
PYTHON_BIN="${TRACE_PYTHON:-${ROOT}/.venv-runtime/bin/python}"
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION="0e9e39f249a16976918f6564b8830bc894c89659"

V3_NEW_OUT="${INSTRUCT_V3_NEW_OUT:-${RUN_ROOT}/v3/${EXPERIMENT_NAME}/v3_new}"
TOP4_OUT="${INSTRUCT_TOP4_OUT:-${RUN_ROOT}/v2/${EXPERIMENT_NAME}/v2_new_top4}"
V2_OUT="${INSTRUCT_V2_OUT:-${RUN_ROOT}/v2/${EXPERIMENT_NAME}/v2_baseline}"
V2_NEW_OUT="${INSTRUCT_V2_NEW_OUT:-${RUN_ROOT}/v2/${EXPERIMENT_NAME}/v2_new}"
LOG_DIR="${INSTRUCT_LOG_DIR:-${RUN_ROOT}/logs/${EXPERIMENT_NAME}}"
COMPARISON_DIR="${INSTRUCT_COMPARISON_DIR:-${RUN_ROOT}/comparisons/${EXPERIMENT_NAME}}"
mkdir -p "${LOG_DIR}" "${COMPARISON_DIR}"

training_output_complete() {
  local version="$1" output="$2"
  "${PYTHON_BIN}" - "${version}" "${output}" <<'PY'
import json
import sys
from pathlib import Path

version, output = sys.argv[1], Path(sys.argv[2])
for round_id in range(8):
    checkpoint = output / str(round_id)
    weights = checkpoint / "pytorch_model.bin"
    if not (checkpoint / "lora_moe_meta.json").is_file():
        raise SystemExit(1)
    if not weights.is_file() or weights.stat().st_size == 0:
        raise SystemExit(1)
metadata = json.loads((output / "7/lora_moe_meta.json").read_text())
if metadata.get("training_version") != version:
    raise SystemExit(1)
PY
}

refuse_nonempty_output() {
  local output="$1"
  if [[ -e "${output}" || -L "${output}" ]]; then
    if [[ -d "${output}" ]] && ! find "${output}" -mindepth 1 -print -quit | grep -q .; then
      return 0
    fi
    echo "[ERROR] refusing nonempty or dangling output: ${output}" >&2
    exit 2
  fi
}

record_provenance() {
  local output="$1" provenance="${output}/run.env"
  touch "${provenance}"
  grep -qxF "model_id=${MODEL_ID}" "${provenance}" ||
    echo "model_id=${MODEL_ID}" >> "${provenance}"
  grep -qxF "model_revision=${MODEL_REVISION}" "${provenance}" ||
    echo "model_revision=${MODEL_REVISION}" >> "${provenance}"
  grep -qxF "continuation_gpu_ids=${GPUS}" "${provenance}" ||
    echo "continuation_gpu_ids=${GPUS}" >> "${provenance}"
  grep -qxF "continuation_effective_global_batch=64" "${provenance}" ||
    echo "continuation_effective_global_batch=64" >> "${provenance}"
}

run_training() {
  local version="$1" output="$2" port="$3" kd_batch="$4"
  if training_output_complete "${version}" "${output}"; then
    record_provenance "${output}"
    echo "[TRAIN SKIP COMPLETE] $(date --iso-8601=seconds) ${version} ${output}"
    return 0
  fi
  refuse_nonempty_output "${output}"
  mkdir -p "${output}"
  echo "[TRAIN START] $(date --iso-8601=seconds) ${version} gpus=${GPUS} micro=16 ga=1 gbs=64 kd_batch=${kd_batch}"
  SLORA_LLAMA31_PATH="${MODEL_PATH}" \
  TRACE_DATA_ROOT="${DATA_ROOT}" \
  OURS_LLAMA31_TOKEN_CACHE="${CACHE_ROOT}" \
  OURS_LORAMOE_OUTPUT_ROOT="${output}" \
  OURS_LORAMOE_GPUS="${GPUS}" \
  OURS_LORAMOE_PORT="${port}" \
  OURS_LORAMOE_MICRO_BATCH=16 \
  OURS_LORAMOE_GRAD_ACCUM=1 \
  OURS_V2_REPLAY_FORWARD_BATCH_SIZE="${INSTRUCT_REPLAY_FORWARD_BATCH_SIZE:-8}" \
  OURS_V2_KD_MEMORY_BATCH_SIZE="${kd_batch}" \
  OURS_LORAMOE_SEED=2025 \
  OURS_REPLAY_SUBSET_SEED=2025 \
  OURS_REPLAY_SELECTION_MODE=random \
  OURS_V3_EPOCH_PROBE_SAMPLES=64 \
  PYTHONNOUSERSITE=1 \
  WANDB_MODE=offline \
  TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 \
  OPENBLAS_NUM_THREADS=8 \
    bash "${ROOT}/scripts/baselines/_run_ours_lora_moe.sh" \
      train llama31 "${version}" 2>&1 | tee -a "${LOG_DIR}/train_${version}.log"
  record_provenance "${output}"
  echo "[TRAIN COMPLETE] $(date --iso-8601=seconds) ${version}"
}

run_sparse15() {
  local version="$1" output="$2"
  if [[ -s "${output}/sparse15_summary.json" ]]; then
    echo "[EVAL SKIP COMPLETE] $(date --iso-8601=seconds) ${version}"
    return 0
  fi
  echo "[EVAL START] $(date --iso-8601=seconds) ${version} gpus=${GPUS} optimized=yes max_new=1024"
  SLORA_LLAMA31_PATH="${MODEL_PATH}" \
  TRACE_DATA_ROOT="${DATA_ROOT}" \
  OURS_LLAMA31_TOKEN_CACHE="${CACHE_ROOT}" \
  OURS_LORAMOE_OUTPUT_ROOT="${output}" \
  SPARSE15_METHODS="ours_lora_moe_${version}" \
  SPARSE15_GPUS="${GPUS}" \
  SPARSE15_EVAL_BATCH="${SPARSE15_EVAL_BATCH:-32}" \
  SPARSE15_SCIENCEQA_BATCH="${SPARSE15_SCIENCEQA_BATCH:-128}" \
  SPARSE15_20MINUTEN_BATCH="${SPARSE15_20MINUTEN_BATCH:-8}" \
  SPARSE15_MEETINGBANK_BATCH="${SPARSE15_MEETINGBANK_BATCH:-1}" \
  SPARSE15_PY150_BATCH="${SPARSE15_PY150_BATCH:-8}" \
  SPARSE15_CPU_THREADS="${SPARSE15_CPU_THREADS:-4}" \
  SPARSE15_STATUS_INTERVAL="${SPARSE15_STATUS_INTERVAL:-30}" \
  SPARSE15_EXACT_STOP_MARKERS=1 \
  SPARSE15_LOG_ROOT="${output}/eval_queue_logs" \
  TRACE_PYTHON="${PYTHON_BIN}" \
  PYTHONNOUSERSITE=1 \
  TOKENIZERS_PARALLELISM=false \
  TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}" \
    "${PYTHON_BIN}" -u "${ROOT}/scripts/run_ours_sparse15_optimized.py" \
      2>&1 | tee -a "${LOG_DIR}/eval_${version}.log"
  [[ -s "${output}/sparse15_summary.json" ]] || {
    echo "[ERROR] missing sparse-15 summary: ${output}" >&2
    exit 3
  }
  echo "[EVAL COMPLETE] $(date --iso-8601=seconds) ${version}"
}

compare_pair() {
  local left_name="$1" left_dir="$2" right_name="$3" right_dir="$4" output="$5"
  "${PYTHON_BIN}" - "${left_name}" "${left_dir}" "${right_name}" "${right_dir}" "${output}" <<'PY'
import json
import sys
from pathlib import Path

left_name, left_dir, right_name, right_dir, output = sys.argv[1:]
left = json.loads((Path(left_dir) / "sparse15_summary.json").read_text())
right = json.loads((Path(right_dir) / "sparse15_summary.json").read_text())
rows = [
    "method\tAA\tBWT\tforgetting",
    f"{left_name}\t{left['final_average']:.6f}\t{left['BWT']:.6f}\t{-left['BWT']:.6f}",
    f"{right_name}\t{right['final_average']:.6f}\t{right['BWT']:.6f}\t{-right['BWT']:.6f}",
    f"delta({left_name}-{right_name})\t{left['final_average']-right['final_average']:.6f}\t{left['BWT']-right['BWT']:.6f}\t{-left['BWT']+right['BWT']:.6f}",
]
Path(output).write_text("\n".join(rows) + "\n")
print("\n".join(rows))
PY
}

echo "[4GPU CONTINUE START] $(date --iso-8601=seconds) gpus=${GPUS}"
[[ -s "${V3_NEW_OUT}/sparse15_summary.json" ]] || {
  echo "[ERROR] V3 sparse-15 summary is not complete: ${V3_NEW_OUT}" >&2
  exit 3
}

# V3 and top4 checkpoints are already trained. Score top4 first, then train and
# score the two reference/replay variants on the same four physical GPUs.
run_sparse15 v2_new_top4 "${TOP4_OUT}"
compare_pair v3_new "${V3_NEW_OUT}" v2_new_top4 "${TOP4_OUT}" \
  "${COMPARISON_DIR}/comparison_v3new_vs_top4.tsv"

run_training v2 "${V2_OUT}" 29833 8
run_training v2_new "${V2_NEW_OUT}" 29834 8
run_sparse15 v2 "${V2_OUT}"
run_sparse15 v2_new "${V2_NEW_OUT}"
compare_pair v2 "${V2_OUT}" v2_new "${V2_NEW_OUT}" \
  "${COMPARISON_DIR}/comparison_v2_vs_v2new.tsv"

echo "[4GPU CONTINUE COMPLETE] $(date --iso-8601=seconds)"
