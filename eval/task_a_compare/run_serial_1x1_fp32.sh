#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_DATASET="${LOCAL_DATASET:-$LOCAL_BASE/dataset}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"

export STAGE_A_WEIGHTS_DIR="${STAGE_A_WEIGHTS_DIR:-$LOCAL_WEIGHTS/continual-stage-A/stage-a-local-fp32-20260310-103742}"
export STAGE_B_WEIGHTS_DIR="${STAGE_B_WEIGHTS_DIR:-$LOCAL_WEIGHTS/continual-stage-B/stage-b-first-local-fp32-20260312-043202}"
export EVAL_DATASET="${EVAL_DATASET:-$LOCAL_DATASET/wikipedia-full/tokenized/EleutherAI/pythia-12b}"
export EVAL_ITERS="${EVAL_ITERS:-10}"
export TARGET_EVAL_TOKENS="${TARGET_EVAL_TOKENS:-0}"
export TARGET_EVAL_SAMPLES="${TARGET_EVAL_SAMPLES:-170496}"
export EVAL_SEQ_LENGTH="${EVAL_SEQ_LENGTH:-512}"
export DUMP_SHARD_SIZE="${DUMP_SHARD_SIZE:-512}"
export COMPACT_DUMP="${COMPACT_DUMP:-1}"
export KEEP_RAW_DUMPS="${KEEP_RAW_DUMPS:-0}"
export CLEAN_OUTPUT="${CLEAN_OUTPUT:-0}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
export NPROC_PER_MODEL="${NPROC_PER_MODEL:-1}"
export GPU_DEVICE="${GPU_DEVICE:-6}"
export MASTER_PORT="${MASTER_PORT:-29610}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((MICRO_BATCH_SIZE * NPROC_PER_MODEL))}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$LOCAL_WEIGHTS/task-a-compare-serial-1x1}"
export LABEL_A="${LABEL_A:-stage-a}"
export LABEL_B="${LABEL_B:-stage-b}"
DEFAULT_PROJECT_PYTHON="$PROJECT_ROOT/.conda/envs/flame3090/bin/python"
if [[ -n "${PYTHON_BIN:-}" ]]; then
    export PYTHON_BIN
elif [[ -x "$DEFAULT_PROJECT_PYTHON" ]]; then
    export PYTHON_BIN="$DEFAULT_PROJECT_PYTHON"
elif command -v python >/dev/null 2>&1; then
    export PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
    export PYTHON_BIN="$(command -v python3)"
else
    echo "ERROR: could not find a usable python interpreter"
    exit 1
fi
export SHOW_PROGRESS="${SHOW_PROGRESS:-0}"
export DATASET_SPLIT="${DATASET_SPLIT:-95,5,0}"
export DATASET_SPLIT_NAME="${DATASET_SPLIT_NAME:-train}"
export CONSUMED_SAMPLES="${CONSUMED_SAMPLES:-4147200}"

if [ "$NPROC_PER_MODEL" != "1" ]; then
    echo "Single-GPU serial runner only supports NPROC_PER_MODEL=1; forcing it to 1."
    export NPROC_PER_MODEL=1
    export GLOBAL_BATCH_SIZE="$MICRO_BATCH_SIZE"
fi

if [ "$TARGET_EVAL_SAMPLES" -gt 0 ]; then
    export EVAL_ITERS=$(((TARGET_EVAL_SAMPLES + GLOBAL_BATCH_SIZE - 1) / GLOBAL_BATCH_SIZE))
    export EFFECTIVE_EVAL_TOKENS=$((EVAL_ITERS * EVAL_SEQ_LENGTH * GLOBAL_BATCH_SIZE))
elif [ "$TARGET_EVAL_TOKENS" -gt 0 ]; then
    tokens_per_iter=$((EVAL_SEQ_LENGTH * GLOBAL_BATCH_SIZE))
    export EVAL_ITERS=$(((TARGET_EVAL_TOKENS + tokens_per_iter - 1) / tokens_per_iter))
    export EFFECTIVE_EVAL_TOKENS=$((EVAL_ITERS * tokens_per_iter))
else
    export EFFECTIVE_EVAL_TOKENS=$((EVAL_ITERS * EVAL_SEQ_LENGTH * GLOBAL_BATCH_SIZE))
fi

if [ "$CLEAN_OUTPUT" = "1" ] && [ -d "$OUTPUT_ROOT" ]; then
    rm -rf "$OUTPUT_ROOT"
fi

data_path_args=($(find "$EVAL_DATASET" -type f -name '*.bin' -exec sh -c 'printf "1.0 %s " "${1%.bin}"' _ {} \;))

run_dump() {
    local label="$1"
    local load_dir="$2"
    local run_log="$3"
    local dump_args=()
    local tids_save="$OUTPUT_ROOT/$label/tids"
    local eact_save="$OUTPUT_ROOT/$label/eact"

    if [ ! -f "$load_dir/latest_checkpointed_iteration.txt" ]; then
        echo "ERROR: missing latest_checkpointed_iteration.txt in $load_dir"
        exit 1
    fi

    echo "Running $label on GPU $GPU_DEVICE"
    echo "  load dir: $load_dir"
    echo "  log file: $run_log"

    if [ "$COMPACT_DUMP" = "1" ]; then
        dump_args+=(--compact-dump --dump-shard-size "$DUMP_SHARD_SIZE")
    fi

    cd "$PROJECT_ROOT/Megatron-LM"
    mkdir -p "$tids_save" "$eact_save"

    if [ "$SHOW_PROGRESS" = "1" ]; then
        CUDA_VISIBLE_DEVICES="$GPU_DEVICE" SHOW_PROGRESS="$SHOW_PROGRESS" TIDS_SAVE="$tids_save" EACT_SAVE="$eact_save" "$PYTHON_BIN" -m torch.distributed.run \
            --standalone \
            --nnodes 1 \
            --nproc_per_node "$NPROC_PER_MODEL" \
            --master_port "$MASTER_PORT" \
            "$PROJECT_ROOT/eval/task_a_compare/dump_checkpoint_eval.py" \
            --load "$load_dir" \
            --dump-dir "$OUTPUT_ROOT" \
            --compare-label "$label" \
            --data-path "${data_path_args[@]}" \
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
            --exit-on-missing-checkpoint \
            --test-mode \
            "${dump_args[@]}" 2>&1 | tee "$run_log"
    else
        CUDA_VISIBLE_DEVICES="$GPU_DEVICE" SHOW_PROGRESS="$SHOW_PROGRESS" TIDS_SAVE="$tids_save" EACT_SAVE="$eact_save" "$PYTHON_BIN" -m torch.distributed.run \
            --standalone \
            --nnodes 1 \
            --nproc_per_node "$NPROC_PER_MODEL" \
            --master_port "$MASTER_PORT" \
            "$PROJECT_ROOT/eval/task_a_compare/dump_checkpoint_eval.py" \
            --load "$load_dir" \
            --dump-dir "$OUTPUT_ROOT" \
            --compare-label "$label" \
            --data-path "${data_path_args[@]}" \
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
            --exit-on-missing-checkpoint \
            --test-mode \
            "${dump_args[@]}" > "$run_log" 2>&1
    fi
    cd "$PROJECT_ROOT"
}

mkdir -p "$OUTPUT_ROOT/logs"

echo "Single-GPU serial routing compare"
echo "  runner gpu: $GPU_DEVICE"
echo "  eval dataset: $EVAL_DATASET"
echo "  output root: $OUTPUT_ROOT"
echo "  eval iters: $EVAL_ITERS"
echo "  target eval samples: $TARGET_EVAL_SAMPLES"
echo "  target eval tokens: $TARGET_EVAL_TOKENS"
echo "  effective eval tokens: $EFFECTIVE_EVAL_TOKENS"
echo "  dataset split: $DATASET_SPLIT ($DATASET_SPLIT_NAME)"
echo "  consumed samples: $CONSUMED_SAMPLES"
echo "  dump shard size: $DUMP_SHARD_SIZE"
echo "  compact dump: $COMPACT_DUMP"
echo "  show progress: $SHOW_PROGRESS"

run_dump "$LABEL_A" "$STAGE_A_WEIGHTS_DIR" "$OUTPUT_ROOT/logs/${LABEL_A}.log"
run_dump "$LABEL_B" "$STAGE_B_WEIGHTS_DIR" "$OUTPUT_ROOT/logs/${LABEL_B}.log"

compare_args=(
    --root "$OUTPUT_ROOT"
    --label-a "$LABEL_A"
    --label-b "$LABEL_B"
    --source-num-experts 4
)

if [ "$KEEP_RAW_DUMPS" != "1" ]; then
    compare_args+=(--cleanup-raw-dumps)
fi

"$PYTHON_BIN" "$PROJECT_ROOT/eval/task_a_compare/compare_outputs.py" "${compare_args[@]}"
