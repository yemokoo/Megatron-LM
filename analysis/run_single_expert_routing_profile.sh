#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DEFAULT_PROJECT_PYTHON="$PROJECT_ROOT/.conda/envs/flame3090/bin/python"
if [[ -n "${PYTHON_BIN:-}" ]]; then
    export PYTHON_BIN
elif [[ -x "$DEFAULT_PROJECT_PYTHON" ]]; then
    export PYTHON_BIN="$DEFAULT_PROJECT_PYTHON"
elif command -v python >/dev/null 2>&1; then
    export PYTHON_BIN="$(command -v python)"
else
    export PYTHON_BIN="$(command -v python3)"
fi

export MODEL_KIND="${MODEL_KIND:-ffn}"
export MODEL_RUN_DIR="${MODEL_RUN_DIR:?MODEL_RUN_DIR must be set}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/analysis_outputs/single_expert_routing/$(basename "$MODEL_RUN_DIR")}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(basename "$MODEL_RUN_DIR")}"
export TASK_WIKI="${TASK_WIKI:-$PROJECT_ROOT/data/wiki/test}"
export TASK_CODE="${TASK_CODE:-$PROJECT_ROOT/data/code/test}"
export TARGET_EVAL_TOKENS="${TARGET_EVAL_TOKENS:-2000000}"
export EVAL_SEQ_LENGTH="${EVAL_SEQ_LENGTH:-512}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export GPU_DEVICE="${GPU_DEVICE:-0}"
export MASTER_PORT_BASE="${MASTER_PORT_BASE:-29950}"
export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-4}"

tokens_per_iter=$((EVAL_SEQ_LENGTH * GLOBAL_BATCH_SIZE))
export EVAL_ITERS=$(((TARGET_EVAL_TOKENS + tokens_per_iter - 1) / tokens_per_iter))

mkdir -p "$OUTPUT_ROOT"

dataset_blend_args() {
  find "$1" -type f -name '*.bin' -exec sh -c 'printf "1.0 %s " "${1%.bin}"' _ {} \; | sed 's/ $//'
}

run_eval() {
  local dataset_name="$1"
  local dataset_dir="$2"
  local output_json="$OUTPUT_ROOT/${dataset_name}_metrics.json"
  local routing_json="$OUTPUT_ROOT/${dataset_name}_routing.json"
  local port="$3"

  CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nnodes 1 \
    --nproc_per_node 1 \
    --master_port "$port" \
    "$PROJECT_ROOT/analysis/eval_controlled_expert_inference.py" \
    --load "$MODEL_RUN_DIR" \
    --output-json "$output_json" \
    --debug-router-json "$routing_json" \
    --compare-label "${EXPERIMENT_NAME}__${dataset_name}" \
    --model-kind "$MODEL_KIND" \
    --source-num-experts "$SOURCE_NUM_EXPERTS" \
    --data-path $(dataset_blend_args "$dataset_dir") \
    --dataset-split 100,0,0 \
    --dataset-split-name train \
    --consumed-samples 0 \
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

run_eval wiki "$TASK_WIKI" "$MASTER_PORT_BASE"
run_eval code "$TASK_CODE" "$((MASTER_PORT_BASE + 1))"

"$PYTHON_BIN" "$PROJECT_ROOT/analysis/plot_single_expert_routing_profile.py" \
  --wiki-routing-json "$OUTPUT_ROOT/wiki_routing.json" \
  --code-routing-json "$OUTPUT_ROOT/code_routing.json" \
  --output-dir "$OUTPUT_ROOT" \
  --title "$EXPERIMENT_NAME"

echo "Routing profile complete: $OUTPUT_ROOT"
echo "Wiki graph: $OUTPUT_ROOT/wiki_expert_group_usage.png"
echo "Code graph: $OUTPUT_ROOT/code_expert_group_usage.png"
echo "Dataset comparison: $OUTPUT_ROOT/dataset_comparison_expert_group_usage.png"
