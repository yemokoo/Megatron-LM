#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_DATASET="${LOCAL_DATASET:-$LOCAL_BASE/dataset}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"

export STAGE_A_WEIGHTS_DIR="${STAGE_A_WEIGHTS_DIR:-$LOCAL_WEIGHTS/continual-stage-A-local/stage-a-local-fp32-20260310-103742}"
export STAGE_B_WEIGHTS_DIR="${STAGE_B_WEIGHTS_DIR:-$LOCAL_WEIGHTS/continual-stage-B-local/stage-b-local-fp32-20260311-005559}"
export EVAL_DATASET="${EVAL_DATASET:-$LOCAL_DATASET/wikipedia-full/tokenized/EleutherAI/pythia-12b}"
export EVAL_ITERS="${EVAL_ITERS:-10}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((MICRO_BATCH_SIZE * NPROC_PER_NODE))}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$LOCAL_WEIGHTS/task-a-compare}"
export LABEL_A="${LABEL_A:-stage-a}"
export LABEL_B="${LABEL_B:-stage-b}"

data_path_args=($(find "$EVAL_DATASET" -type f -name '*.bin' -exec sh -c 'printf "1.0 %s " "${1%.bin}"' _ {} \;))

run_dump() {
    local label="$1"
    local load_dir="$2"
    local tids_save="$OUTPUT_ROOT/$label/tids"
    local eact_save="$OUTPUT_ROOT/$label/eact"

    python3 - <<'PY' "$load_dir"
import sys
from pathlib import Path
tracker = Path(sys.argv[1]) / "latest_checkpointed_iteration.txt"
if not tracker.exists():
    raise SystemExit(f"Missing tracker file: {tracker}")
PY

    cd "$PROJECT_ROOT/Megatron-LM"
    TIDS_SAVE="$tids_save" \
    EACT_SAVE="$eact_save" \
    python -m torch.distributed.run \
        --standalone \
        --nnodes 1 \
        --nproc_per_node "$NPROC_PER_NODE" \
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
        --test-mode
    cd "$PROJECT_ROOT"
}

run_dump "$LABEL_A" "$STAGE_A_WEIGHTS_DIR"
run_dump "$LABEL_B" "$STAGE_B_WEIGHTS_DIR"

python3 "$PROJECT_ROOT/eval/task_a_compare/compare_outputs.py" \
    --root "$OUTPUT_ROOT" \
    --label-a "$LABEL_A" \
    --label-b "$LABEL_B" \
    --source-num-experts 4
