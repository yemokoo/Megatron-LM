#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_DATASET="${LOCAL_DATASET:-$LOCAL_BASE/dataset}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"

export STAGE_A_WEIGHTS_DIR="${STAGE_A_WEIGHTS_DIR:-$LOCAL_WEIGHTS/continual-stage-A-local/stage-a-local-fp32-20260310-103742}"
export STAGE_B_WEIGHTS_DIR="${STAGE_B_WEIGHTS_DIR:-$LOCAL_WEIGHTS/continual-stage-B-local/stage-b-local-fp32-20260311-005559}"
export EVAL_DATASET="${EVAL_DATASET:-$LOCAL_DATASET/wikipedia-full/tokenized/EleutherAI/pythia-12b}"
export EVAL_ITERS="${EVAL_ITERS:-10}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
export NPROC_PER_MODEL="${NPROC_PER_MODEL:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((MICRO_BATCH_SIZE * NPROC_PER_MODEL))}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$LOCAL_WEIGHTS/task-a-compare-parallel-2x2}"
export LABEL_A="${LABEL_A:-stage-a}"
export LABEL_B="${LABEL_B:-stage-b}"
export GPUS_A="${GPUS_A:-0,1}"
export GPUS_B="${GPUS_B:-2,3}"
export MASTER_PORT_A="${MASTER_PORT_A:-29610}"
export MASTER_PORT_B="${MASTER_PORT_B:-29620}"

data_path_args=($(find "$EVAL_DATASET" -type f -name '*.bin' -exec sh -c 'printf "1.0 %s " "${1%.bin}"' _ {} \;))

run_dump() {
    local label="$1"
    local load_dir="$2"
    local visible_devices="$3"
    local master_port="$4"
    local run_log="$5"
    local tids_save="$OUTPUT_ROOT/$label/tids"
    local eact_save="$OUTPUT_ROOT/$label/eact"

    cd "$PROJECT_ROOT/Megatron-LM"
    CUDA_VISIBLE_DEVICES="$visible_devices" \
    TIDS_SAVE="$tids_save" \
    EACT_SAVE="$eact_save" \
    python -m torch.distributed.run \
        --standalone \
        --nnodes 1 \
        --nproc_per_node "$NPROC_PER_MODEL" \
        --master_port "$master_port" \
        "$PROJECT_ROOT/eval/task_a_compare/dump_checkpoint_eval.py" \
        --load "$load_dir" \
        --dump-dir "$OUTPUT_ROOT" \
        --compare-label "$label" \
        --data-path "${data_path_args[@]}" \
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
        --test-mode > "$run_log" 2>&1
    cd "$PROJECT_ROOT"
}

mkdir -p "$OUTPUT_ROOT/logs"

run_dump "$LABEL_A" "$STAGE_A_WEIGHTS_DIR" "$GPUS_A" "$MASTER_PORT_A" "$OUTPUT_ROOT/logs/${LABEL_A}.log" &
PID_A=$!
run_dump "$LABEL_B" "$STAGE_B_WEIGHTS_DIR" "$GPUS_B" "$MASTER_PORT_B" "$OUTPUT_ROOT/logs/${LABEL_B}.log" &
PID_B=$!

wait "$PID_A"
wait "$PID_B"

python3 "$PROJECT_ROOT/eval/task_a_compare/compare_outputs.py" \
    --root "$OUTPUT_ROOT" \
    --label-a "$LABEL_A" \
    --label-b "$LABEL_B" \
    --source-num-experts 4
