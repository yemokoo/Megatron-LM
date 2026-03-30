#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

resolve_python() {
    if [ -x "$PROJECT_ROOT/.conda/envs/flame3090/bin/python" ]; then
        echo "$PROJECT_ROOT/.conda/envs/flame3090/bin/python"
        return
    fi
    command -v python3 || command -v python
}

build_data_path() {
    "$PYTHON_BIN" - "$1" <<'PY'
import sys
from pathlib import Path
dataset_dir = Path(sys.argv[1])
parts = []
for bin_path in sorted(dataset_dir.glob('*.bin')):
    parts.extend(['1.0', str(bin_path.with_suffix(''))])
print(' '.join(parts))
PY
}

ceil_div() {
    "$PYTHON_BIN" - "$@" <<'PY'
import math, sys
print(math.ceil(int(sys.argv[1]) / int(sys.argv[2])))
PY
}

export PYTHON_BIN="${PYTHON_BIN:-$(resolve_python)}"
export MODEL_KIND="${MODEL_KIND:-ffn}"
export MODEL_RUN_DIR="${MODEL_RUN_DIR:-}"
export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-4}"
export TOTAL_NUM_EXPERTS="${TOTAL_NUM_EXPERTS:-7}"
export GPU_DEVICE="${GPU_DEVICE:-0}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export SEQ_LENGTH="${SEQ_LENGTH:-512}"
export TARGET_EVAL_TOKENS="${TARGET_EVAL_TOKENS:-1000000}"
export WIKI_EVAL_DATASET="${WIKI_EVAL_DATASET:-$PROJECT_ROOT/data/wiki/test}"
export CODE_EVAL_DATASET="${CODE_EVAL_DATASET:-$PROJECT_ROOT/data/code/test}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/analysis_outputs/expert_group_masks/${MODEL_KIND}}"

if [ -z "$MODEL_RUN_DIR" ]; then
  echo "ERROR: set MODEL_RUN_DIR"
  exit 1
fi

wiki_group="$(seq 0 $((SOURCE_NUM_EXPERTS - 1)) | paste -sd, -)"
code_group="$(seq "$SOURCE_NUM_EXPERTS" $((TOTAL_NUM_EXPERTS - 1)) | paste -sd, -)"
export EVAL_ITERS="$(ceil_div "$TARGET_EVAL_TOKENS" "$((MICRO_BATCH_SIZE * SEQ_LENGTH))")"
mkdir -p "$OUTPUT_ROOT"
WIKI_DATA_PATH="$(build_data_path "$WIKI_EVAL_DATASET")"
CODE_DATA_PATH="$(build_data_path "$CODE_EVAL_DATASET")"

run_one() {
  local dataset_name="$1"
  local data_path="$2"
  local mode="$3"
  local allowed="$4"
  local label="${MODEL_KIND}_${dataset_name}_${mode}"
  args=(
    analysis/eval_controlled_expert_inference.py
    --model-kind "$MODEL_KIND"
    --load "$MODEL_RUN_DIR"
    --compare-label "$label"
    --output-json "$OUTPUT_ROOT/${label}.json"
    --debug-router-json "$OUTPUT_ROOT/${label}_routing.json"
    --source-num-experts "$SOURCE_NUM_EXPERTS"
    --topk-override 2
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --eval-iters "$EVAL_ITERS"
    --seq-length "$SEQ_LENGTH"
    --dataset-split 100,0,0
    --dataset-split-name train
    --consumed-samples 0
    --transformer-impl local
    --no-persist-layer-norm
    --no-gradient-accumulation-fusion
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
    --no-load-optim
    --no-load-rng
    --exit-on-missing-checkpoint
    --data-path
  )
  CUDA_VISIBLE_DEVICES="$GPU_DEVICE" torchrun --nproc_per_node 1 --master_port "$MASTER_PORT" "${args[@]}" $data_path ${allowed:+--allowed-experts "$allowed"}
}

run_one wiki "$WIKI_DATA_PATH" unrestricted ""
run_one wiki "$WIKI_DATA_PATH" wiki_only "$wiki_group"
run_one wiki "$WIKI_DATA_PATH" code_only "$code_group"
run_one code "$CODE_DATA_PATH" unrestricted ""
run_one code "$CODE_DATA_PATH" wiki_only "$wiki_group"
run_one code "$CODE_DATA_PATH" code_only "$code_group"

echo "Saved outputs under: $OUTPUT_ROOT"
