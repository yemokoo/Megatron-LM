#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_DATASET="${LOCAL_DATASET:-$LOCAL_BASE/dataset}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
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

export BASE_WEIGHTS_DIR="${BASE_WEIGHTS_DIR:-$LOCAL_WEIGHTS/continual-stage-A/stage-a-local-fp32-20260310-103742}"
export BASE_ITERATION="${BASE_ITERATION:-1800}"
export TARGET_FULL_WEIGHTS_DIR="${TARGET_FULL_WEIGHTS_DIR:-$LOCAL_WEIGHTS/continual-stage-A-to_B/stage-b-resume-local-fp32-20260311-171347}"
export TARGET_FULL_ITERATION="${TARGET_FULL_ITERATION:-1800}"
export TARGET_NEW_ONLY_WEIGHTS_DIR="${TARGET_NEW_ONLY_WEIGHTS_DIR:-$LOCAL_WEIGHTS/continual-stage-A-to-B-new-only/stage-b-new-expert-router-only-local-fp32-20260317-113542}"
export TARGET_NEW_ONLY_ITERATION="${TARGET_NEW_ONLY_ITERATION:-1800}"
export EVAL_DATASET="${EVAL_DATASET:-$LOCAL_DATASET/python-code-full/tokenized/EleutherAI/pythia-12b}"
export TARGET_EVAL_SAMPLES="${TARGET_EVAL_SAMPLES:-170496}"
export EVAL_ITERS="${EVAL_ITERS:-10}"
export EVAL_SEQ_LENGTH="${EVAL_SEQ_LENGTH:-512}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
export NPROC_PER_MODEL="${NPROC_PER_MODEL:-1}"
export GPU_DEVICE="${GPU_DEVICE:-6}"
export MASTER_PORT="${MASTER_PORT:-29630}"
export EVAL_LOAD_ROOT="${EVAL_LOAD_ROOT:-$LOCAL_WEIGHTS/interpolation-eval-loads}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$LOCAL_WEIGHTS/interpolation-task-b-suite}"
export ALPHAS="${ALPHAS:-0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
export SHOW_PROGRESS="${SHOW_PROGRESS:-1}"
export DATASET_SPLIT="${DATASET_SPLIT:-95,5,0}"
export DATASET_SPLIT_NAME="${DATASET_SPLIT_NAME:-train}"
export CONSUMED_SAMPLES="${CONSUMED_SAMPLES:-4147200}"

export NUM_LAYERS="${NUM_LAYERS:-9}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-5472}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-704}"
export MOE_LAYER_FREQ="${MOE_LAYER_FREQ:-[0]*1+[1]*8}"
export NUM_EXPERTS="${NUM_EXPERTS:-7}"
export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-4}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-2}"
export TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-local}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-configs/model/flame-moe.sh}"

# Reuse the canonical model-argument bundle from training scripts.
# shellcheck disable=SC1090
source "$PROJECT_ROOT/$MODEL_CONFIG_SCRIPT"

export EVAL_ITERS=$(((TARGET_EVAL_SAMPLES + GLOBAL_BATCH_SIZE - 1) / GLOBAL_BATCH_SIZE))

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
    printf '%s\n' "$load_dir"
}

data_path_args=($(find "$EVAL_DATASET" -type f -name '*.bin' -exec sh -c 'printf "1.0 %s " "${1%.bin}"' _ {} \;))

run_pair() {
    local source_load="$1"
    local target_load="$2"
    local label="$3"
    local output_dir="$4"
    local port="$5"
    local tids_save="$output_dir/tids"
    local eact_save="$output_dir/eact"

    mkdir -p "$output_dir" "$tids_save" "$eact_save"
    cd "$PROJECT_ROOT/Megatron-LM"
    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" SHOW_PROGRESS="$SHOW_PROGRESS" TIDS_SAVE="$tids_save" EACT_SAVE="$eact_save" "$PYTHON_BIN" -m torch.distributed.run \
        --standalone \
        --nnodes 1 \
        --nproc_per_node "$NPROC_PER_MODEL" \
        --master_port "$port" \
        "$PROJECT_ROOT/eval/interpolation/eval_interpolated_checkpoint.py" \
        --source-load "$source_load" \
        --target-load "$target_load" \
        --source-expand-from-num-experts "$SOURCE_NUM_EXPERTS" \
        --alphas "$ALPHAS" \
        --output-dir "$output_dir" \
        --plot-title-prefix "$label on Task B" \
        --data-path "${data_path_args[@]}" \
        --dataset-split "$DATASET_SPLIT" \
        --dataset-split-name "$DATASET_SPLIT_NAME" \
        --consumed-samples "$CONSUMED_SAMPLES" \
        --split 0,1,0 \
        --eval-iters "$EVAL_ITERS" \
        --micro-batch-size "$MICRO_BATCH_SIZE" \
        --global-batch-size "$GLOBAL_BATCH_SIZE" \
        --seq-length "$EVAL_SEQ_LENGTH" \
        --pipeline-model-parallel-size 1 \
        --expert-model-parallel-size 1 \
        --tensor-model-parallel-size 1 \
        --transformer-impl "$TRANSFORMER_IMPL" \
        --no-persist-layer-norm \
        --no-gradient-accumulation-fusion \
        --no-masked-softmax-fusion \
        --attention-softmax-in-fp32 \
        --no-load-optim \
        --no-load-rng \
        --exit-on-missing-checkpoint \
        --test-mode \
        "${MODEL_ARGS[@]}"
    cd "$PROJECT_ROOT"
}

BASE_LOAD_DIR="$(prepare_eval_load_dir "$BASE_WEIGHTS_DIR" "$BASE_ITERATION" "interp-a-base")"
TARGET_FULL_LOAD_DIR="$(prepare_eval_load_dir "$TARGET_FULL_WEIGHTS_DIR" "$TARGET_FULL_ITERATION" "interp-a-to-b")"
TARGET_NEW_ONLY_LOAD_DIR="$(prepare_eval_load_dir "$TARGET_NEW_ONLY_WEIGHTS_DIR" "$TARGET_NEW_ONLY_ITERATION" "interp-a-to-b-new-only")"

echo "Base load dir:            $BASE_LOAD_DIR"
echo "Target full load dir:     $TARGET_FULL_LOAD_DIR"
echo "Target new-only load dir: $TARGET_NEW_ONLY_LOAD_DIR"
echo "Eval dataset:             $EVAL_DATASET"
echo "Dataset split:            $DATASET_SPLIT ($DATASET_SPLIT_NAME)"
echo "Consumed samples:         $CONSUMED_SAMPLES"
echo "Target eval samples:      $TARGET_EVAL_SAMPLES"
echo "Output root:              $OUTPUT_ROOT"
echo "Alphas:                   $ALPHAS"

run_pair "$BASE_LOAD_DIR" "$TARGET_FULL_LOAD_DIR" "A-to-B interpolation" "$OUTPUT_ROOT/a-to-b" "$MASTER_PORT"
run_pair "$BASE_LOAD_DIR" "$TARGET_NEW_ONLY_LOAD_DIR" "A-to-B new-only interpolation" "$OUTPUT_ROOT/a-to-b-new-only" "$((MASTER_PORT + 1))"

echo "Generated graphs:"
echo "  $OUTPUT_ROOT/a-to-b/next_token_accuracy.svg"
echo "  $OUTPUT_ROOT/a-to-b/ppl.svg"
echo "  $OUTPUT_ROOT/a-to-b-new-only/next_token_accuracy.svg"
echo "  $OUTPUT_ROOT/a-to-b-new-only/ppl.svg"
