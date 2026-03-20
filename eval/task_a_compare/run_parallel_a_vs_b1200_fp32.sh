#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"

export STAGE_A_WEIGHTS_DIR="${STAGE_A_WEIGHTS_DIR:-$LOCAL_WEIGHTS/continual-stage-A-local/stage-a-local-fp32-20260310-103742}"
export SOURCE_B_WEIGHTS_DIR="${SOURCE_B_WEIGHTS_DIR:-$LOCAL_WEIGHTS/continual-stage-B-local/stage-b-local-fp32-20260311-005559}"
export B_EVAL_ITERATION="${B_EVAL_ITERATION:-1200}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$LOCAL_WEIGHTS/task-a-compare-b1200}"
export EVAL_LOAD_ROOT="${EVAL_LOAD_ROOT:-$LOCAL_WEIGHTS/task-a-compare-loads}"

mkdir -p "$EVAL_LOAD_ROOT"

export B_EVAL_LOAD_DIR="${EVAL_LOAD_ROOT}/stage-b-eval-iter-$(printf '%07d' "$B_EVAL_ITERATION")"

checkpoint_dir="${SOURCE_B_WEIGHTS_DIR}/iter_$(printf '%07d' "$B_EVAL_ITERATION")"
if [ ! -d "$checkpoint_dir" ]; then
    echo "ERROR: missing B checkpoint directory $checkpoint_dir"
    exit 1
fi

echo "Preparing Task B eval load dir at: $B_EVAL_LOAD_DIR"
rsync -rlptD \
    --delete \
    --exclude 'logs/' \
    --exclude 'wandb/' \
    --exclude 'events.out.tfevents*' \
    --exclude 'progress.txt' \
    "$SOURCE_B_WEIGHTS_DIR/" "$B_EVAL_LOAD_DIR/"
printf '%s\n' "$B_EVAL_ITERATION" > "$B_EVAL_LOAD_DIR/latest_checkpointed_iteration.txt"

echo "Task A weights dir: $STAGE_A_WEIGHTS_DIR"
echo "Task B eval source: $SOURCE_B_WEIGHTS_DIR"
echo "Task B eval iteration: $B_EVAL_ITERATION"
echo "Task B eval load dir: $B_EVAL_LOAD_DIR"

GPUS_A="${GPUS_A:-0,1}"
GPUS_B="${GPUS_B:-2,3}"
NPROC_PER_MODEL="${NPROC_PER_MODEL:-2}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
EVAL_ITERS="${EVAL_ITERS:-10}"

GPUS_A="$GPUS_A" \
GPUS_B="$GPUS_B" \
NPROC_PER_MODEL="$NPROC_PER_MODEL" \
MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" \
EVAL_ITERS="$EVAL_ITERS" \
STAGE_A_WEIGHTS_DIR="$STAGE_A_WEIGHTS_DIR" \
STAGE_B_WEIGHTS_DIR="$B_EVAL_LOAD_DIR" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
LABEL_A="${LABEL_A:-stage-a}" \
LABEL_B="${LABEL_B:-stage-b-iter-$(printf '%07d' "$B_EVAL_ITERATION")}" \
bash "$PROJECT_ROOT/eval/task_a_compare/run_parallel_2x2_fp32.sh"
