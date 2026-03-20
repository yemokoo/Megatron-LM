#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_DATASET="${LOCAL_DATASET:-$LOCAL_BASE/dataset}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.conda/envs/flame3090/bin/python}"

export BASE_WEIGHTS_DIR="${BASE_WEIGHTS_DIR:-$LOCAL_WEIGHTS/continual-stage-A/stage-a-local-fp32-20260310-103742}"
export TARGET_FULL_WEIGHTS_DIR="${TARGET_FULL_WEIGHTS_DIR:-$LOCAL_WEIGHTS/continual-stage-A-to_B/stage-b-resume-local-fp32-20260311-171347}"
export BASE_ITERATION="${BASE_ITERATION:-1800}"
export TARGET_STEPS="${TARGET_STEPS:-300,600,900,1200,1500,1800}"
export EVAL_DATASET="${EVAL_DATASET:-$LOCAL_DATASET/python-code-full/tokenized/EleutherAI/pythia-12b-step1800-test-matchwiki-exact}"
export TARGET_EVAL_TOKENS="${TARGET_EVAL_TOKENS:-10000000}"
export EVAL_SEQ_LENGTH="${EVAL_SEQ_LENGTH:-512}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-24}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-24}"
export NPROC_PER_MODEL="${NPROC_PER_MODEL:-1}"
export GPU_DEVICE="${GPU_DEVICE:-0}"
export MASTER_PORT="${MASTER_PORT:-29720}"
export EVAL_LOAD_ROOT="${EVAL_LOAD_ROOT:-$LOCAL_WEIGHTS/interpolation-eval-loads}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$LOCAL_WEIGHTS/probe-comparison-suite/a_to_b_full_code_schedule}"
export SHOW_PROGRESS="${SHOW_PROGRESS:-1}"
export DATASET_SPLIT="${DATASET_SPLIT:-100,0,0}"
export DATASET_SPLIT_NAME="${DATASET_SPLIT_NAME:-train}"
export CONSUMED_SAMPLES="${CONSUMED_SAMPLES:-0}"
export TIDS_SAVE="${TIDS_SAVE:-$OUTPUT_ROOT/tids}"
export EACT_SAVE="${EACT_SAVE:-$OUTPUT_ROOT/eact}"

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

# shellcheck disable=SC1090
source "$PROJECT_ROOT/$MODEL_CONFIG_SCRIPT"

export TARGET_EVAL_SAMPLES=$(((TARGET_EVAL_TOKENS + EVAL_SEQ_LENGTH - 1) / EVAL_SEQ_LENGTH))
export EVAL_ITERS=$(((TARGET_EVAL_SAMPLES + GLOBAL_BATCH_SIZE - 1) / GLOBAL_BATCH_SIZE))

data_path_args=($(find "$EVAL_DATASET" -type f -name '*.bin' -exec sh -c 'printf "1.0 %s " "${1%.bin}"' _ {} \;))

prepare_eval_load_dir() {
    local source_dir="$1"
    local requested_iteration="$2"
    local tag="$3"

    if [ "$requested_iteration" = "latest" ]; then
        printf '%s\n' "$source_dir"
        return 0
    fi

    mkdir -p "$EVAL_LOAD_ROOT"
    local load_dir
    load_dir="$EVAL_LOAD_ROOT/${tag}-minimal-iter-$(printf '%07d' "$requested_iteration")"
    local checkpoint_dir
    checkpoint_dir="$source_dir/iter_$(printf '%07d' "$requested_iteration")"
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

BASE_LOAD_DIR="$(prepare_eval_load_dir "$BASE_WEIGHTS_DIR" "$BASE_ITERATION" "probe-a-base")"

mkdir -p "$OUTPUT_ROOT" "$TIDS_SAVE" "$EACT_SAVE"

cd "$PROJECT_ROOT/Megatron-LM"
CUDA_VISIBLE_DEVICES="$GPU_DEVICE" SHOW_PROGRESS="$SHOW_PROGRESS" TIDS_SAVE="$TIDS_SAVE" EACT_SAVE="$EACT_SAVE" "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nnodes 1 \
    --nproc_per_node "$NPROC_PER_MODEL" \
    --master_port "$MASTER_PORT" \
    "$PROJECT_ROOT/eval/interpolation/eval_checkpoint_schedule.py" \
    --base-load "$BASE_LOAD_DIR" \
    --target-run-root "$TARGET_FULL_WEIGHTS_DIR" \
    --target-steps "$TARGET_STEPS" \
    --source-expand-from-num-experts "$SOURCE_NUM_EXPERTS" \
    --eval-load-root "$EVAL_LOAD_ROOT" \
    --output-dir "$OUTPUT_ROOT" \
    --plot-title-prefix "A-to-B full checkpoints on code test" \
    --series-name "a_to_b_full_code" \
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
