#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_DATASET="${LOCAL_DATASET:-$LOCAL_BASE/dataset}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"

export RUNNER_SCRIPT="${RUNNER_SCRIPT:-$PROJECT_ROOT/eval/task_a_compare/run_serial_1x1_fp32.sh}"
export GPU_DEVICE="${GPU_DEVICE:-6}"
export MASTER_PORT="${MASTER_PORT:-29610}"
export SUITE_OUTPUT_ROOT="${SUITE_OUTPUT_ROOT:-$LOCAL_WEIGHTS/routing-compare-suite}"

export STAGE_A_BASE_DIR="${STAGE_A_BASE_DIR:-$LOCAL_WEIGHTS/continual-stage-A/stage-a-local-fp32-20260310-103742}"
export STAGE_B_BASE_DIR="${STAGE_B_BASE_DIR:-$LOCAL_WEIGHTS/continual-stage-B/stage-b-first-local-fp32-20260312-043202}"

export A_TO_B_RESUME_DIR="${A_TO_B_RESUME_DIR:-$LOCAL_WEIGHTS/continual-stage-A-to_B/stage-b-resume-local-fp32-20260311-171347}"
export A_TO_B_RESUME_ITERATION="${A_TO_B_RESUME_ITERATION:-1800}"

export A_TO_B_NEW_ONLY_DIR="${A_TO_B_NEW_ONLY_DIR:-$LOCAL_WEIGHTS/continual-stage-A-to-B-new-only/stage-b-new-expert-router-only-local-fp32-20260314-090733}"
export A_TO_B_NEW_ONLY_ITERATION="${A_TO_B_NEW_ONLY_ITERATION:-1800}"
export RUN_A_TO_B_NEW_ONLY="${RUN_A_TO_B_NEW_ONLY:-1}"

export B_TO_A_DIR="${B_TO_A_DIR:-$LOCAL_WEIGHTS/continual-stage-B-to-A/stage-a-after-b-local-fp32-20260312-175645}"
export B_TO_A_ITERATION="${B_TO_A_ITERATION:-1800}"

export WIKI_DATASET="${WIKI_DATASET:-$LOCAL_DATASET/wikipedia-full/tokenized/EleutherAI/pythia-12b}"
export CODE_DATASET="${CODE_DATASET:-$LOCAL_DATASET/python-code-full/tokenized/EleutherAI/pythia-12b}"

run_pair() {
    local name="$1"
    local eval_dataset="$2"
    local base_dir="$3"
    local base_iteration="$4"
    local target_dir="$5"
    local target_iteration="$6"
    local label_a="$7"
    local label_b="$8"

    echo
    echo "=== Running $name ==="

    RUNNER_SCRIPT="$RUNNER_SCRIPT"     GPU_DEVICE="$GPU_DEVICE"     MASTER_PORT="$MASTER_PORT"     OUTPUT_ROOT="$SUITE_OUTPUT_ROOT/$name"     EVAL_DATASET="$eval_dataset"     BASE_WEIGHTS_DIR="$base_dir"     BASE_ITERATION="$base_iteration"     TARGET_WEIGHTS_DIR="$target_dir"     TARGET_ITERATION="$target_iteration"     LABEL_A="$label_a"     LABEL_B="$label_b"     LOAD_TAG_A="$name-$label_a"     LOAD_TAG_B="$name-$label_b"     bash "$PROJECT_ROOT/eval/task_a_compare/run_compare_pair_fp32.sh"
}

echo "Routing comparison suite"
echo "  runner script: $RUNNER_SCRIPT"
echo "  gpu device:    $GPU_DEVICE"
echo "  output root:   $SUITE_OUTPUT_ROOT"

run_pair     "a-base-vs-a-to-b-resume-2100"     "$WIKI_DATASET"     "$STAGE_A_BASE_DIR"     latest     "$A_TO_B_RESUME_DIR"     "$A_TO_B_RESUME_ITERATION"     "a-base"     "a-to-b-resume-2100"

if [ "$RUN_A_TO_B_NEW_ONLY" = "1" ]; then
    run_pair         "a-base-vs-a-to-b-new-only"         "$WIKI_DATASET"         "$STAGE_A_BASE_DIR"         latest         "$A_TO_B_NEW_ONLY_DIR"         "$A_TO_B_NEW_ONLY_ITERATION"         "a-base"         "a-to-b-new-only"
else
    echo
    echo "=== Skipping a-base-vs-a-to-b-new-only ==="
    echo "RUN_A_TO_B_NEW_ONLY=1 로 켜고, 가능하면 iteration도 고정해서 사용하세요."
fi

run_pair     "b-base-vs-b-to-a"     "$CODE_DATASET"     "$STAGE_B_BASE_DIR"     latest     "$B_TO_A_DIR"     "$B_TO_A_ITERATION"     "b-base"     "b-to-a"

echo
echo "Routing comparison suite complete. Outputs under: $SUITE_OUTPUT_ROOT"
