#!/bin/bash
# One queue unit: train a v3 variant, gate its config, then score it sparse-15.
#
# Train and eval stay in the same unit because the eval is what makes the
# training worth anything, and splitting them would let a lane pick up the next
# training while this one has no numbers yet.
#
# Phases are separable so the series can train every variant first and only
# then score them: comparing six models is much easier when none of them is
# still moving, and a checkpoint that trains late does not have to wait behind
# an earlier model's evaluation.
#
#   run_v3_job.sh <launcher-slug> <version> <gpu-list> [train|eval|both]
# e.g. run_v3_job.sh v3_replay1to1 v3_new_replay1to1 0,1,2,3 train
set -euo pipefail

SLUG="${1:?usage: run_v3_job.sh <slug> <version> <gpus> [phase]}"
VERSION="${2:?usage: run_v3_job.sh <slug> <version> <gpus> [phase]}"
GPUS="${3:?usage: run_v3_job.sh <slug> <version> <gpus> [phase]}"
PHASE="${4:-both}"
case "${PHASE}" in train|eval|both) ;; *) echo "[ERROR] unknown phase: ${PHASE}" >&2; exit 2 ;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace
RUN_ROOT="${RUNS}/${SLUG}"
OUTPUT="${RUN_ROOT}/${VERSION}_st_top1"
LAUNCHER="${ROOT}/scripts/run_${SLUG}_st_top1_8gpu.sh"
[[ -x "${LAUNCHER}" ]] || { echo "[ERROR] missing launcher ${LAUNCHER}" >&2; exit 2; }

export SLORA_LLAMA31_PATH="${SLORA_LLAMA31_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}"
export TRACE_DATA_ROOT="${TRACE_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace}"

# The knobs each run is allowed to move.  Anything else differing from the
# v3_new baseline is a configuration mistake, not an experiment.
ALLOW=(
  --allow=--replay_subset_ratio
  --allow=--router_replay_exposure_samples
  --allow=--v2_joint_new_to_replay_ratio
  --allow=--v2_new_active_memory_cap
  --allow=--v2_new_persistent_samples_per_task
  --allow=--v2_joint_replay_objective
  --allow=--replay_distribution
  --allow=--replay_recency_power
  --allow=--v2_kd_exposure_samples
  --allow=--v2_kd_epochs
)

echo "[JOB ${VERSION}] phase=${PHASE} gpus=${GPUS} $(date '+%F %T')"

if [[ "${PHASE}" != "eval" ]] && [[ ! -e "${OUTPUT}/7/lora_moe_meta.json" ]]; then
  # Gate first: a dry run writes train.command.txt without taking a GPU, so a
  # bad config fails in seconds instead of five hours in.
  VALIDATE_ROOT="${RUNS}/${SLUG}_validate"
  rm -rf "${VALIDATE_ROOT}"
  OURS_ST_RUN_ROOT="${VALIDATE_ROOT}" OURS_ACTION=validate \
    OURS_LORAMOE_PORT=$(( 30000 + RANDOM % 500 )) \
    bash "${LAUNCHER}" > /dev/null 2>&1
  "${ROOT}/.venv-runtime/bin/python" \
    "${ROOT}/scripts/assert_matches_v3_new_baseline.py" \
    "${VALIDATE_ROOT}/${VERSION}_st_top1" "${ALLOW[@]}"
  rm -rf "${VALIDATE_ROOT}"

  # The launcher sizes micro-batch for eight GPUs, but a lane hands it four,
  # which silently halved the effective batch to 32 on every run so far while
  # the gate -- which inspects an eight-GPU dry run -- still passed.  Derive the
  # micro-batch from the lane's actual GPU count so the effective batch is the
  # baseline's 64 regardless of lane width.
  # Cap the per-device batch at 16 -- the largest value that has actually
  # trained here -- and make up the rest of the effective 64 with accumulation.
  # A two-GPU lane would otherwise need micro 32, which doubles activations on
  # a card already sitting near 36GB.
  lane_gpu_count=$(awk -F, '{print NF}' <<< "${GPUS}")
  lane_micro=$(( 64 / lane_gpu_count ))
  (( lane_micro > 16 )) && lane_micro=16
  lane_accum=$(( 64 / (lane_gpu_count * lane_micro) ))
  if (( lane_gpu_count * lane_micro * lane_accum != 64 )); then
    echo "[ERROR] ${lane_gpu_count} GPUs cannot make an effective batch of 64" >&2
    exit 2
  fi
  echo "[JOB ${VERSION}] ${lane_gpu_count} gpus x micro ${lane_micro} x accum ${lane_accum} = 64"
  OURS_ST_RUN_ROOT="${RUN_ROOT}" OURS_LORAMOE_GPUS="${GPUS}" \
  OURS_LORAMOE_MICRO_BATCH="${lane_micro}" OURS_LORAMOE_GRAD_ACCUM="${lane_accum}" \
    bash "${LAUNCHER}"
  echo "[JOB ${VERSION}] training done $(date '+%F %T')"
elif [[ "${PHASE}" != "eval" ]]; then
  echo "[JOB ${VERSION}] training already complete"
fi

if [[ "${PHASE}" == "train" ]]; then
  echo "[JOB ${VERSION}] train-only phase finished $(date '+%F %T')"
  exit 0
fi
if [[ ! -e "${OUTPUT}/7/lora_moe_meta.json" ]]; then
  echo "[ERROR] ${VERSION} has no round-7 checkpoint; cannot evaluate" >&2
  exit 2
fi

# --max_prompt_len 0 means prompts are never truncated, and both of these tasks
# have a brutal tail: MeetingBank runs median 1,230 tokens but reaches 74,300,
# Py150 median 206 but reaches 20,125.  A typical batch sits near 18GB while a
# batch that happens to collect a few of those long documents takes the whole
# card, which is why raising these OOMed late in a cell rather than at its
# start.  MeetingBank stays at 4 because its tail is 60x its median.  Py150's
# Py150 has now OOMed at 48, 24 and 16 -- its 20,125-token tail is enough to
# take the card whatever the p90 says -- so it drops to 8.  MeetingBank at 4 is
# the only long-task setting with completed cells and no OOM behind it.
#
# Do not raise either without pointing at a cell that completed at the higher
# value.  Reasoning from mean or p90 prompt length has been wrong three times
# and each wrong guess costs about an hour per lost cell.
#
# MeetingBank, Py150 and ScienceQA are split four ways so each of those cells
# runs across all four GPUs instead of pinning one card for an hour.  Sharding
# divides a cell's samples and merges the results, so the matrix is still the
# same 15 cells and the scores are unchanged.
#
# Sharding is wall-clock only -- each shard still runs a full batch, so it does
# nothing for the OOMs above.  Batch size is the memory knob, shard count is the
# parallelism knob; both had to be set.
#
# Nothing may be inserted between the assignments below -- they are one
# continued command, and a comment line inside it silently comments out the
# python invocation at the end.
LOG_DIR="${RUNS}/logs/${SLUG}_sparse15"
mkdir -p "${LOG_DIR}"
OURS_LORAMOE_OUTPUT_ROOT="${OUTPUT}" \
SPARSE15_METHODS="ours_lora_moe_${VERSION}" \
SPARSE15_GPUS="${GPUS}" \
SPARSE15_NUM_SAMPLE_SHARDS="${SPARSE15_NUM_SAMPLE_SHARDS:-4}" \
SPARSE15_EXACT_STOP_MARKERS=1 \
SPARSE15_EVAL_BATCH="${SPARSE15_EVAL_BATCH:-96}" \
SPARSE15_SCIENCEQA_BATCH="${SPARSE15_SCIENCEQA_BATCH:-256}" \
SPARSE15_20MINUTEN_BATCH="${SPARSE15_20MINUTEN_BATCH:-64}" \
SPARSE15_MEETINGBANK_BATCH="${SPARSE15_MEETINGBANK_BATCH:-4}" \
SPARSE15_PY150_BATCH="${SPARSE15_PY150_BATCH:-8}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
SPARSE15_LOG_ROOT="${LOG_DIR}/queue_logs" \
SPARSE15_STATUS_INTERVAL=120 SPARSE15_CONTINUE_ON_CELL_ERROR=1 \
TRACE_PYTHON="${ROOT}/.venv-runtime/bin/python" \
PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 \
  "${ROOT}/.venv-runtime/bin/python" -u \
  "${ROOT}/scripts/run_ours_sparse15_optimized.py" \
  2>&1 | tee "${LOG_DIR}/eval.log"

echo "[JOB ${VERSION}] eval done $(date '+%F %T')"
