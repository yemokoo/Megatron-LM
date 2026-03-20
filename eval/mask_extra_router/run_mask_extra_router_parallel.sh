#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_DATASET="${LOCAL_DATASET:-$LOCAL_BASE/dataset}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.conda/envs/flame3090/bin/python}"

export BASELINE_A_DIR="${BASELINE_A_DIR:-$LOCAL_WEIGHTS/continual-stage-A/stage-a-local-fp32-20260310-103742}"
export BASELINE_A_ITERATION="${BASELINE_A_ITERATION:-1800}"
export BASELINE_B_DIR="${BASELINE_B_DIR:-$LOCAL_WEIGHTS/continual-stage-B/stage-b-first-local-fp32-20260312-043202}"
export BASELINE_B_ITERATION="${BASELINE_B_ITERATION:-1800}"
export A_TO_B_DIR="${A_TO_B_DIR:-$LOCAL_WEIGHTS/continual-stage-A-to_B/stage-b-resume-local-fp32-20260311-171347}"
export A_TO_B_ITERATION="${A_TO_B_ITERATION:-1800}"
export A_TO_B_NEW_ONLY_DIR="${A_TO_B_NEW_ONLY_DIR:-$LOCAL_WEIGHTS/continual-stage-A-to-B-new-only/stage-b-new-expert-router-only-local-fp32-20260317-113542}"
export A_TO_B_NEW_ONLY_ITERATION="${A_TO_B_NEW_ONLY_ITERATION:-1800}"
export B_TO_A_DIR="${B_TO_A_DIR:-$LOCAL_WEIGHTS/continual-stage-B-to-A/stage-a-after-b-local-fp32-20260312-175645}"
export B_TO_A_ITERATION="${B_TO_A_ITERATION:-1800}"
export B_TO_A_NEW_ONLY_DIR="${B_TO_A_NEW_ONLY_DIR:-$LOCAL_WEIGHTS/continual-stage-B-to-A-new-only/stage-a-after-b-new-expert-router-only-local-fp32-20260315-135509}"
export B_TO_A_NEW_ONLY_ITERATION="${B_TO_A_NEW_ONLY_ITERATION:-1800}"

export TASK_A_DATASET="${TASK_A_DATASET:-$LOCAL_DATASET/wikipedia-full/tokenized/EleutherAI/pythia-12b-step1800-test-fullremainder-exact}"
export TASK_B_DATASET="${TASK_B_DATASET:-$LOCAL_DATASET/python-code-full/tokenized/EleutherAI/pythia-12b-step1800-test-matchwiki-exact}"
export TARGET_EVAL_TOKENS="${TARGET_EVAL_TOKENS:-10000000}"
export TARGET_EVAL_SAMPLES="${TARGET_EVAL_SAMPLES:-0}"
export EVAL_SEQ_LENGTH="${EVAL_SEQ_LENGTH:-512}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-24}"
export NPROC_PER_MODEL="${NPROC_PER_MODEL:-1}"
export GPU_DEVICES="${GPU_DEVICES:-0,1,2,3}"
export MASTER_PORT_BASE="${MASTER_PORT_BASE:-29640}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$LOCAL_WEIGHTS/mask-extra-router-suite-parallel}"
export EVAL_LOAD_ROOT="${EVAL_LOAD_ROOT:-$LOCAL_WEIGHTS/mask-extra-router-loads}"
export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-4}"
export DATASET_SPLIT="${DATASET_SPLIT:-100,0,0}"
export DATASET_SPLIT_NAME="${DATASET_SPLIT_NAME:-train}"
export CONSUMED_SAMPLES="${CONSUMED_SAMPLES:-0}"

export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((MICRO_BATCH_SIZE * NPROC_PER_MODEL))}"
if [ "$TARGET_EVAL_SAMPLES" -gt 0 ]; then
    export EVAL_ITERS=$(((TARGET_EVAL_SAMPLES + GLOBAL_BATCH_SIZE - 1) / GLOBAL_BATCH_SIZE))
elif [ "$TARGET_EVAL_TOKENS" -gt 0 ]; then
    tokens_per_iter=$((EVAL_SEQ_LENGTH * GLOBAL_BATCH_SIZE))
    export EVAL_ITERS=$(((TARGET_EVAL_TOKENS + tokens_per_iter - 1) / tokens_per_iter))
    export TARGET_EVAL_SAMPLES=$((EVAL_ITERS * GLOBAL_BATCH_SIZE))
else
    echo "ERROR: set TARGET_EVAL_TOKENS>0 or TARGET_EVAL_SAMPLES>0"
    exit 1
fi

IFS=',' read -r -a GPU_LIST <<< "$GPU_DEVICES"
if [ "${#GPU_LIST[@]}" -lt 4 ]; then
    echo "ERROR: GPU_DEVICES must contain at least 4 GPU ids, e.g. 0,1,2,3"
    exit 1
fi

prepare_eval_load_dir() {
    local source_dir="$1"
    local requested_iteration="$2"
    local tag="$3"

    if [ ! -f "$source_dir/latest_checkpointed_iteration.txt" ]; then
        echo "ERROR: missing latest_checkpointed_iteration.txt in $source_dir"
        exit 1
    fi

    if [ "$requested_iteration" = "latest" ]; then
        printf '%s\n' "$source_dir"
        return 0
    fi

    local checkpoint_dir="$source_dir/iter_$(printf '%07d' "$requested_iteration")"
    if [ ! -d "$checkpoint_dir" ]; then
        echo "ERROR: missing checkpoint directory $checkpoint_dir"
        exit 1
    fi

    mkdir -p "$EVAL_LOAD_ROOT"
    local load_dir="$EVAL_LOAD_ROOT/${tag}-minimal-iter-$(printf '%07d' "$requested_iteration")"
    local load_checkpoint_dir="$load_dir/iter_$(printf '%07d' "$requested_iteration")"

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
    printf '%s\n' "$load_dir"
}

dataset_blend_args() {
    local dataset_dir="$1"
    find "$dataset_dir" -type f -name '*.bin' -exec sh -c 'printf "1.0 %s " "${1%.bin}"' _ {} \; | sed 's/ $//'
}

run_eval() {
    local gpu_id="$1"
    local label="$2"
    local load_dir="$3"
    local dataset_dir="$4"
    local output_json="$5"
    local mask_from_expert="$6"
    local port="$7"
    local mask_args=()

    if [ -n "$mask_from_expert" ]; then
        mask_args+=(--mask-from-expert "$mask_from_expert")
    fi

    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" -m torch.distributed.run \
        --standalone \
        --nnodes 1 \
        --nproc_per_node "$NPROC_PER_MODEL" \
        --master_port "$port" \
        "$PROJECT_ROOT/eval/mask_extra_router/eval_masked_router_acc.py" \
        --load "$load_dir" \
        --output-json "$output_json" \
        --debug-router-json "${output_json%.json}_router_debug.json" \
        --compare-label "$label" \
        "${mask_args[@]}" \
        --data-path $(dataset_blend_args "$dataset_dir") \
        --dataset-split "$DATASET_SPLIT" \
        --dataset-split-name "$DATASET_SPLIT_NAME" \
        --consumed-samples "$CONSUMED_SAMPLES" \
        --split 0,1,0 \
        --eval-iters "$EVAL_ITERS" \
        --micro-batch-size "$MICRO_BATCH_SIZE" \
        --global-batch-size "$GLOBAL_BATCH_SIZE" \
        --pipeline-model-parallel-size 1 \
        --expert-model-parallel-size 1 \
        --tensor-model-parallel-size 1 \
        --transformer-impl local \
        --no-persist-layer-norm \
        --no-gradient-accumulation-fusion \
        --no-masked-softmax-fusion \
        --attention-softmax-in-fp32 \
        --no-load-optim \
        --no-load-rng \
        --exit-on-missing-checkpoint
}

run_bucket() {
    local gpu_id="$1"
    shift
    while [ "$#" -gt 0 ]; do
        local label="$1"; shift
        local load_dir="$1"; shift
        local dataset_dir="$1"; shift
        local output_json="$1"; shift
        local mask_from_expert="$1"; shift
        local port="$1"; shift
        echo "Running $label on GPU $gpu_id"
        run_eval "$gpu_id" "$label" "$load_dir" "$dataset_dir" "$output_json" "$mask_from_expert" "$port"
    done
}

mkdir -p "$OUTPUT_ROOT"

BASELINE_A_LOAD="$(prepare_eval_load_dir "$BASELINE_A_DIR" "$BASELINE_A_ITERATION" baseline-a)"
BASELINE_B_LOAD="$(prepare_eval_load_dir "$BASELINE_B_DIR" "$BASELINE_B_ITERATION" baseline-b)"
A_TO_B_LOAD="$(prepare_eval_load_dir "$A_TO_B_DIR" "$A_TO_B_ITERATION" a-to-b)"
A_TO_B_NEW_ONLY_LOAD="$(prepare_eval_load_dir "$A_TO_B_NEW_ONLY_DIR" "$A_TO_B_NEW_ONLY_ITERATION" a-to-b-new-only)"
B_TO_A_LOAD="$(prepare_eval_load_dir "$B_TO_A_DIR" "$B_TO_A_ITERATION" b-to-a)"
B_TO_A_NEW_ONLY_LOAD="$(prepare_eval_load_dir "$B_TO_A_NEW_ONLY_DIR" "$B_TO_A_NEW_ONLY_ITERATION" b-to-a-new-only)"

run_bucket "${GPU_LIST[0]}" \
    "baseline_a" "$BASELINE_A_LOAD" "$TASK_A_DATASET" "$OUTPUT_ROOT/baseline_a.json" "" "$MASTER_PORT_BASE" \
    "a_to_b" "$A_TO_B_LOAD" "$TASK_A_DATASET" "$OUTPUT_ROOT/a_to_b.json" "" "$((MASTER_PORT_BASE + 1))" \
    "a_to_b_masked" "$A_TO_B_LOAD" "$TASK_A_DATASET" "$OUTPUT_ROOT/a_to_b_masked.json" "$SOURCE_NUM_EXPERTS" "$((MASTER_PORT_BASE + 2))" &
pid0=$!

run_bucket "${GPU_LIST[1]}" \
    "baseline_b" "$BASELINE_B_LOAD" "$TASK_B_DATASET" "$OUTPUT_ROOT/baseline_b.json" "" "$((MASTER_PORT_BASE + 10))" \
    "b_to_a" "$B_TO_A_LOAD" "$TASK_B_DATASET" "$OUTPUT_ROOT/b_to_a.json" "" "$((MASTER_PORT_BASE + 11))" \
    "b_to_a_masked" "$B_TO_A_LOAD" "$TASK_B_DATASET" "$OUTPUT_ROOT/b_to_a_masked.json" "$SOURCE_NUM_EXPERTS" "$((MASTER_PORT_BASE + 12))" &
pid1=$!

run_bucket "${GPU_LIST[2]}" \
    "a_to_b_new_only" "$A_TO_B_NEW_ONLY_LOAD" "$TASK_A_DATASET" "$OUTPUT_ROOT/a_to_b_new_only.json" "" "$((MASTER_PORT_BASE + 20))" \
    "a_to_b_new_only_masked" "$A_TO_B_NEW_ONLY_LOAD" "$TASK_A_DATASET" "$OUTPUT_ROOT/a_to_b_new_only_masked.json" "$SOURCE_NUM_EXPERTS" "$((MASTER_PORT_BASE + 21))" &
pid2=$!

run_bucket "${GPU_LIST[3]}" \
    "b_to_a_new_only" "$B_TO_A_NEW_ONLY_LOAD" "$TASK_B_DATASET" "$OUTPUT_ROOT/b_to_a_new_only.json" "" "$((MASTER_PORT_BASE + 30))" \
    "b_to_a_new_only_masked" "$B_TO_A_NEW_ONLY_LOAD" "$TASK_B_DATASET" "$OUTPUT_ROOT/b_to_a_new_only_masked.json" "$SOURCE_NUM_EXPERTS" "$((MASTER_PORT_BASE + 31))" &
pid3=$!

wait "$pid0" "$pid1" "$pid2" "$pid3"

"$PYTHON_BIN" "$PROJECT_ROOT/eval/mask_extra_router/plot_mask_extra_router_results.py" \
    --baseline-a "$OUTPUT_ROOT/baseline_a.json" \
    --baseline-b "$OUTPUT_ROOT/baseline_b.json" \
    --a-to-b "$OUTPUT_ROOT/a_to_b.json" \
    --a-to-b-masked "$OUTPUT_ROOT/a_to_b_masked.json" \
    --a-to-b-new-only "$OUTPUT_ROOT/a_to_b_new_only.json" \
    --a-to-b-new-only-masked "$OUTPUT_ROOT/a_to_b_new_only_masked.json" \
    --b-to-a "$OUTPUT_ROOT/b_to_a.json" \
    --b-to-a-masked "$OUTPUT_ROOT/b_to_a_masked.json" \
    --b-to-a-new-only "$OUTPUT_ROOT/b_to_a_new_only.json" \
    --b-to-a-new-only-masked "$OUTPUT_ROOT/b_to_a_new_only_masked.json" \
    --output-json "$OUTPUT_ROOT/summary.json" \
    --output-svg "$OUTPUT_ROOT/task1_accuracy_comparison.svg"

echo "Mask-extra-router summary: $OUTPUT_ROOT/summary.json"
echo "Mask-extra-router chart:   $OUTPUT_ROOT/task1_accuracy_comparison.svg"
echo "Mask-extra-router A->B:    $OUTPUT_ROOT/task1_accuracy_a_to_b.svg"
echo "Mask-extra-router A->B new-only: $OUTPUT_ROOT/task1_accuracy_a_to_b_new_only.svg"
echo "Mask-extra-router B->A:    $OUTPUT_ROOT/task1_accuracy_b_to_a.svg"
echo "Mask-extra-router B->A new-only: $OUTPUT_ROOT/task1_accuracy_b_to_a_new_only.svg"
