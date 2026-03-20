#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_DATASET="${LOCAL_DATASET:-$LOCAL_BASE/dataset}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"

export RUNNER_SCRIPT="${RUNNER_SCRIPT:-$PROJECT_ROOT/eval/task_a_compare/run_serial_1x1_fp32.sh}"
export BASE_WEIGHTS_DIR="${BASE_WEIGHTS_DIR:-$LOCAL_WEIGHTS/continual-stage-A/stage-a-local-fp32-20260310-103742}"
export TARGET_WEIGHTS_DIR="${TARGET_WEIGHTS_DIR:-$LOCAL_WEIGHTS/continual-stage-A-to_B/stage-b-resume-local-fp32-20260311-171347}"
export BASE_ITERATION="${BASE_ITERATION:-latest}"
export TARGET_ITERATION="${TARGET_ITERATION:-latest}"
export EVAL_DATASET="${EVAL_DATASET:-$LOCAL_DATASET/wikipedia-full/tokenized/EleutherAI/pythia-12b}"
export EVAL_ITERS="${EVAL_ITERS:-10}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
export NPROC_PER_MODEL="${NPROC_PER_MODEL:-1}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$LOCAL_WEIGHTS/routing-compare/default-pair}"
export EVAL_LOAD_ROOT="${EVAL_LOAD_ROOT:-$LOCAL_WEIGHTS/task-a-compare-loads}"
export LABEL_A="${LABEL_A:-baseline}"
export LABEL_B="${LABEL_B:-target}"
export LOAD_TAG_A="${LOAD_TAG_A:-$LABEL_A}"
export LOAD_TAG_B="${LOAD_TAG_B:-$LABEL_B}"
export GPU_DEVICE="${GPU_DEVICE:-6}"
export MASTER_PORT="${MASTER_PORT:-29610}"

prepare_eval_load_dir() {
    local source_dir="$1"
    local requested_iteration="$2"
    local tag="$3"

    if [ ! -f "$source_dir/latest_checkpointed_iteration.txt" ]; then
        echo "ERROR: missing latest_checkpointed_iteration.txt in $source_dir"
        exit 1
    fi

    if [ "$requested_iteration" = "latest" ]; then
        printf '%s
' "$source_dir"
        return 0
    fi

    local checkpoint_dir
    checkpoint_dir="$source_dir/iter_$(printf '%07d' "$requested_iteration")"
    if [ ! -d "$checkpoint_dir" ]; then
        echo "ERROR: missing checkpoint directory $checkpoint_dir"
        exit 1
    fi

    mkdir -p "$EVAL_LOAD_ROOT"
    local load_dir
    load_dir="$EVAL_LOAD_ROOT/${tag}-minimal-iter-$(printf '%07d' "$requested_iteration")"
    local load_checkpoint_dir
    load_checkpoint_dir="$load_dir/iter_$(printf '%07d' "$requested_iteration")"

    if [ -f "$load_dir/latest_checkpointed_iteration.txt" ] && [ -d "$load_checkpoint_dir" ]; then
        local existing_iteration
        existing_iteration="$(tr -d '[:space:]' < "$load_dir/latest_checkpointed_iteration.txt" || true)"
        if [ "$existing_iteration" = "$requested_iteration" ]; then
            printf '%s\n' "$load_dir"
            return 0
        fi
    fi

    mkdir -p "$load_checkpoint_dir"
    rsync -rlptD --delete "$checkpoint_dir/" "$load_checkpoint_dir/"
    printf '%s\n' "$requested_iteration" > "$load_dir/latest_checkpointed_iteration.txt"

    printf '%s
' "$load_dir"
}

if [ ! -f "$RUNNER_SCRIPT" ]; then
    echo "ERROR: missing runner script $RUNNER_SCRIPT"
    exit 1
fi

BASE_LOAD_DIR="$(prepare_eval_load_dir "$BASE_WEIGHTS_DIR" "$BASE_ITERATION" "$LOAD_TAG_A")"
TARGET_LOAD_DIR="$(prepare_eval_load_dir "$TARGET_WEIGHTS_DIR" "$TARGET_ITERATION" "$LOAD_TAG_B")"

echo "Runner script:             $RUNNER_SCRIPT"
echo "Routing compare baseline: $BASE_WEIGHTS_DIR (iteration=$BASE_ITERATION)"
echo "Routing compare target:   $TARGET_WEIGHTS_DIR (iteration=$TARGET_ITERATION)"
echo "Resolved baseline load:   $BASE_LOAD_DIR"
echo "Resolved target load:     $TARGET_LOAD_DIR"
echo "Eval dataset:             $EVAL_DATASET"
echo "Output root:              $OUTPUT_ROOT"
echo "GPU device:               $GPU_DEVICE"

GPU_DEVICE="$GPU_DEVICE" NPROC_PER_MODEL="$NPROC_PER_MODEL" MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" EVAL_ITERS="$EVAL_ITERS" STAGE_A_WEIGHTS_DIR="$BASE_LOAD_DIR" STAGE_B_WEIGHTS_DIR="$TARGET_LOAD_DIR" EVAL_DATASET="$EVAL_DATASET" OUTPUT_ROOT="$OUTPUT_ROOT" LABEL_A="$LABEL_A" LABEL_B="$LABEL_B" MASTER_PORT="$MASTER_PORT" bash "$RUNNER_SCRIPT"
