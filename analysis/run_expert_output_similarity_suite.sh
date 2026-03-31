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

export PYTHON_BIN="${PYTHON_BIN:-$(resolve_python)}"
export MODEL_KIND="${MODEL_KIND:-ffn}"
export MODEL_RUN_DIR="${MODEL_RUN_DIR:-}"
export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-4}"
export GPU_DEVICE="${GPU_DEVICE:-0}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
export SEQ_LENGTH="${SEQ_LENGTH:-512}"
export MAX_BATCHES="${MAX_BATCHES:-16}"
export MAX_TOKENS_PER_LAYER="${MAX_TOKENS_PER_LAYER:-16384}"
export PLOT_LAYERS="${PLOT_LAYERS:-}"
export SAVE_HIDDEN_CACHE="${SAVE_HIDDEN_CACHE:-1}"
export HEATMAP_VMIN="${HEATMAP_VMIN:--0.25}"
export HEATMAP_VMAX="${HEATMAP_VMAX:-0.75}"
export MASTER_PORT_BASE="${MASTER_PORT_BASE:-29500}"
export WIKI_EVAL_DATASET="${WIKI_EVAL_DATASET:-$PROJECT_ROOT/data/wiki/test}"
export CODE_EVAL_DATASET="${CODE_EVAL_DATASET:-$PROJECT_ROOT/data/code/test}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/analysis_outputs/expert_output_similarity/${MODEL_KIND}}"

if [ -z "$MODEL_RUN_DIR" ]; then
  echo "ERROR: set MODEL_RUN_DIR"
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
WIKI_DATA_PATH="$(build_data_path "$WIKI_EVAL_DATASET")"
CODE_DATA_PATH="$(build_data_path "$CODE_EVAL_DATASET")"

run_one() {
  local dataset_name="$1"
  local data_path="$2"
  local master_port="$3"
  local out_dir="$OUTPUT_ROOT/${dataset_name}"
  local extra_args=()
  if [ "$SAVE_HIDDEN_CACHE" = "1" ]; then
    extra_args+=(--save-hidden-cache)
  fi
  CUDA_VISIBLE_DEVICES="$GPU_DEVICE" torchrun --nproc_per_node 1 --master_port "$master_port" \
    analysis/eval_expert_output_similarity.py \
    --model-kind "$MODEL_KIND" \
    --load "$MODEL_RUN_DIR" \
    --compare-label "${MODEL_KIND}_${dataset_name}_expert_output_similarity" \
    --output-dir "$out_dir" \
    --source-num-experts "$SOURCE_NUM_EXPERTS" \
    --micro-batch-size "$MICRO_BATCH_SIZE" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --seq-length "$SEQ_LENGTH" \
    --max-batches "$MAX_BATCHES" \
    --max-tokens-per-layer "$MAX_TOKENS_PER_LAYER" \
    --heatmap-vmin "$HEATMAP_VMIN" \
    --heatmap-vmax "$HEATMAP_VMAX" \
    "${extra_args[@]}" \
    ${PLOT_LAYERS:+--plot-layers "$PLOT_LAYERS"} \
    --data-path $data_path \
    --dataset-split 100,0,0 \
    --dataset-split-name train \
    --consumed-samples 0 \
    --transformer-impl local \
    --no-persist-layer-norm \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --no-load-optim \
    --no-load-rng \
    --exit-on-missing-checkpoint
}

run_one wiki "$WIKI_DATA_PATH" "$MASTER_PORT_BASE"
run_one code "$CODE_DATA_PATH" "$((MASTER_PORT_BASE + 1))"

echo "Saved outputs under: $OUTPUT_ROOT"
