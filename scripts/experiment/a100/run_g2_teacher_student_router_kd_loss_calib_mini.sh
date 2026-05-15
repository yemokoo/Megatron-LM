#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

export LAMBDA_SWEEP="${LAMBDA_SWEEP:-1}"
export TRAIN_ITERS="${TRAIN_ITERS:-20}"
export STAGE1_REQUIRED_ITERS="${STAGE1_REQUIRED_ITERS:-1800}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-72}"
export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"
export USE_GUARD="${USE_GUARD:-0}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-999999}"
export LOG_INTERVAL="${LOG_INTERVAL:-1}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-999999}"
export RUN_INITIAL_VALID_EVAL="${RUN_INITIAL_VALID_EVAL:-0}"
export RUN_INITIAL_PROBE_EVAL="${RUN_INITIAL_PROBE_EVAL:-0}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-999999}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-999999}"
export TRAIN_LOG_STEP_TIME_ONLY="${TRAIN_LOG_STEP_TIME_ONLY:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS="${SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS:-1}"

BASE_STAGE_DIR="${BASE_STAGE_DIR:-$PROJECT_ROOT/.local/weights/a100/mha/shared-router-granularity-qkvo}"
RUN_PREFIX="${RUN_PREFIX:-g2-ts-routerkd-allrouter-losscalib}"

echo "[CONFIG] G2 teacher-student router KD loss calibration mini run"
echo "[CONFIG] lambdas=${LAMBDA_SWEEP}"
echo "[CONFIG] train_iters=${TRAIN_ITERS}, micro_batch_size=${MICRO_BATCH_SIZE}, save_checkpoints=${SAVE_CHECKPOINTS}"
echo "[CONFIG] wandb_mode=${WANDB_MODE}"

for lambda in $LAMBDA_SWEEP; do
    lambda_tag="${lambda//./p}"
    timestamp="$(date +%Y%m%d-%H%M%S)"
    run_id="${RUN_PREFIX}-kl${lambda_tag}-mini${TRAIN_ITERS}-${timestamp}"
    log_file="g2_ts_routerkd_allrouter_losscalib_kl${lambda_tag}_mini${TRAIN_ITERS}_${timestamp}.log"

    echo
    echo "[RUN] lambda=${lambda} run_id=${run_id}"
    echo "[RUN] log=${log_file}"

    WANDB_MODE="$WANDB_MODE" \
    RUN_ID="$run_id" \
    TRAIN_WEIGHTS="$BASE_STAGE_DIR/code/$run_id" \
    WANDB_EXP_NAME="G2 ts-routerKD allrouter loss calib kl${lambda_tag} mini${TRAIN_ITERS}" \
    TRAIN_ITERS="$TRAIN_ITERS" \
    STAGE1_REQUIRED_ITERS="$STAGE1_REQUIRED_ITERS" \
    LOG_INTERVAL="$LOG_INTERVAL" \
    SAVE_CHECKPOINTS="$SAVE_CHECKPOINTS" \
    USE_GUARD="$USE_GUARD" \
    SAVE_INTERVAL="$SAVE_INTERVAL" \
    EVAL_INTERVAL="$EVAL_INTERVAL" \
    RUN_INITIAL_VALID_EVAL="$RUN_INITIAL_VALID_EVAL" \
    RUN_INITIAL_PROBE_EVAL="$RUN_INITIAL_PROBE_EVAL" \
    PROBE_EVAL_INTERVAL="$PROBE_EVAL_INTERVAL" \
    SECONDARY_PROBE_EVAL_INTERVAL="$SECONDARY_PROBE_EVAL_INTERVAL" \
    TRAIN_LOG_STEP_TIME_ONLY="$TRAIN_LOG_STEP_TIME_ONLY" \
    MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" \
    ROUTER_MEMORY_KL_COEFF="$lambda" \
    SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS="$SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS" \
    bash "$SCRIPT_DIR/run_g2_teacher_student_router_kd_fullwiki_mha.sh" \
        > "$log_file" 2>&1

    echo "[DONE] ${run_id}"
    tail -40 "$log_file"
done

python - <<'PY'
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

base = Path(".local/weights/a100/mha/shared-router-granularity-qkvo/code")
tags = [
    "lm loss",
    "router_memory_teacher_student/kl",
    "router_memory_teacher_student/scaled_kl",
    "router_memory_teacher_student/lm_to_raw_kl",
    "router_memory_teacher_student/lm_to_scaled_kl",
    "combined/lm_plus_router_kd",
]

print("\n[SUMMARY] latest loss-calibration runs")
for run in sorted(base.glob("g2-ts-routerkd-allrouter-losscalib-kl*-mini*"))[-20:]:
    values = {}
    for event_file in run.rglob("events.out.tfevents.*"):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        scalar_tags = set(accumulator.Tags().get("scalars", []))
        for tag in tags:
            if tag in scalar_tags:
                scalars = accumulator.Scalars(tag)
                if scalars:
                    values[tag] = scalars[-1].value
    if not values:
        continue
    print(f"\n### {run.name}")
    for tag in tags:
        if tag in values:
            print(f"{tag}: {values[tag]:.6g}")
PY
