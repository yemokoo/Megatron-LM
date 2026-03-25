#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

resolve_python() {
    if [ -x "$PROJECT_ROOT/.conda/envs/flame3090/bin/python" ]; then
        echo "$PROJECT_ROOT/.conda/envs/flame3090/bin/python"
        return
    fi
    if command -v python >/dev/null 2>&1; then
        command -v python
        return
    fi
    command -v python3
}

export PYTHON_BIN="${PYTHON_BIN:-$(resolve_python)}"
export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export MODEL_RUN_DIR="${MODEL_RUN_DIR:-$LOCAL_WEIGHTS/code-from-wiki-qv-lora-expand-local/wiki-to-code-qv-lora-mb32-1800}"
export MODEL_WEIGHTS_DIR="${MODEL_WEIGHTS_DIR:-}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$LOCAL_WEIGHTS/qv-lora-routing-a100/$(basename "$MODEL_RUN_DIR")}"
export WIKI_EVAL_DATASET="${WIKI_EVAL_DATASET:-$PROJECT_ROOT/data/wiki/test}"
export CODE_EVAL_DATASET="${CODE_EVAL_DATASET:-$PROJECT_ROOT/data/code/test}"
export TARGET_EVAL_TOKENS="${TARGET_EVAL_TOKENS:-10000000}"
export TARGET_EVAL_SAMPLES="${TARGET_EVAL_SAMPLES:-0}"
export EVAL_ITERS="${EVAL_ITERS:-10}"
export EVAL_SEQ_LENGTH="${EVAL_SEQ_LENGTH:-512}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
export NPROC_PER_MODEL="${NPROC_PER_MODEL:-1}"
export GPU_DEVICE="${GPU_DEVICE:-0}"
export MASTER_PORT="${MASTER_PORT:-29670}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((MICRO_BATCH_SIZE * NPROC_PER_MODEL))}"
export DATASET_SPLIT="${DATASET_SPLIT:-100,0,0}"
export DATASET_SPLIT_NAME="${DATASET_SPLIT_NAME:-train}"
export CONSUMED_SAMPLES="${CONSUMED_SAMPLES:-0}"
export SHOW_PROGRESS="${SHOW_PROGRESS:-0}"

resolve_model_weights_dir() {
    if [ -n "$MODEL_WEIGHTS_DIR" ]; then
        printf '%s\n' "$MODEL_WEIGHTS_DIR"
        return 0
    fi
    if [ -f "$MODEL_RUN_DIR/target_weights/latest_checkpointed_iteration.txt" ]; then
        printf '%s\n' "$MODEL_RUN_DIR/target_weights"
        return 0
    fi
    if [ -f "$MODEL_RUN_DIR/weights/latest_checkpointed_iteration.txt" ]; then
        printf '%s\n' "$MODEL_RUN_DIR/weights"
        return 0
    fi
    if [ -f "$MODEL_RUN_DIR/latest_checkpointed_iteration.txt" ]; then
        printf '%s\n' "$MODEL_RUN_DIR"
        return 0
    fi
    echo "ERROR: could not resolve checkpoint load directory from MODEL_RUN_DIR=$MODEL_RUN_DIR"
    exit 1
}

read_run_metadata_value() {
    local key="$1"
    "$PYTHON_BIN" - "$MODEL_RUN_DIR" "$key" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
key = sys.argv[2]
metadata_path = run_dir / "logs" / "run_metadata.json"
if not metadata_path.exists():
    raise SystemExit("")
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
value = metadata.get(key, "")
if value != "":
    print(value)
PY
}

build_data_path() {
    "$PYTHON_BIN" - "$@" <<'PY'
import sys
from pathlib import Path
parts = []
for dataset_dir in sys.argv[1:]:
    for bin_path in sorted(Path(dataset_dir).glob('*.bin')):
        parts.extend(['1.0', str(bin_path.with_suffix(''))])
print(' '.join(parts))
PY
}

MODEL_WEIGHTS_DIR="$(resolve_model_weights_dir)"
SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-$(read_run_metadata_value 'attn_lora_source_num_experts')}"
TARGET_NUM_EXPERTS="${TARGET_NUM_EXPERTS:-$(read_run_metadata_value 'attn_lora_target_num_experts')}"
ATTN_LORA_NUM_EXPERTS="${ATTN_LORA_NUM_EXPERTS:-$(read_run_metadata_value 'attn_lora_num_experts')}"
ATTN_LORA_TOPK="${ATTN_LORA_TOPK:-$(read_run_metadata_value 'attn_lora_topk')}"
ATTN_LORA_RANK="${ATTN_LORA_RANK:-$(read_run_metadata_value 'attn_lora_rank')}"
ATTN_LORA_ALPHA="${ATTN_LORA_ALPHA:-$(read_run_metadata_value 'attn_lora_alpha')}"
SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-4}"
TARGET_NUM_EXPERTS="${TARGET_NUM_EXPERTS:-7}"
ATTN_LORA_NUM_EXPERTS="${ATTN_LORA_NUM_EXPERTS:-$TARGET_NUM_EXPERTS}"
ATTN_LORA_TOPK="${ATTN_LORA_TOPK:-1}"
ATTN_LORA_RANK="${ATTN_LORA_RANK:-16}"
ATTN_LORA_ALPHA="${ATTN_LORA_ALPHA:-16}"

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

mkdir -p "$OUTPUT_ROOT/logs"

run_dump() {
    local label="$1"
    local dataset_dir="$2"
    local output_json="$OUTPUT_ROOT/$label/summary.json"
    local run_log="$OUTPUT_ROOT/logs/${label}.log"
    local data_path_args

    data_path_args=($(build_data_path "$dataset_dir"))

    cd "$PROJECT_ROOT/Megatron-LM"
    if [ "$SHOW_PROGRESS" = "1" ]; then
        CUDA_VISIBLE_DEVICES="$GPU_DEVICE" SHOW_PROGRESS="$SHOW_PROGRESS" "$PYTHON_BIN" -m torch.distributed.run \
            --standalone \
            --nnodes 1 \
            --nproc_per_node "$NPROC_PER_MODEL" \
            --master_port "$MASTER_PORT" \
            "$PROJECT_ROOT/eval/qv_lora_routing/dump_qv_lora_routing_eval.py" \
            --load "$MODEL_WEIGHTS_DIR" \
            --output-json "$output_json" \
            --compare-label "$label" \
            --source-num-experts "$SOURCE_NUM_EXPERTS" \
            --attn-lora-num-experts "$ATTN_LORA_NUM_EXPERTS" \
            --attn-lora-topk "$ATTN_LORA_TOPK" \
            --attn-lora-rank "$ATTN_LORA_RANK" \
            --attn-lora-alpha "$ATTN_LORA_ALPHA" \
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
            --transformer-impl local \
            --spec megatron.core.models.gpt.qv_lora_layer_specs gpt_qv_lora_local_spec \
            --bf16 \
            --no-persist-layer-norm \
            --no-gradient-accumulation-fusion \
            --no-masked-softmax-fusion \
            --attention-softmax-in-fp32 \
            --no-load-optim \
            --no-load-rng \
            --exit-on-missing-checkpoint \
            --test-mode 2>&1 | tee "$run_log"
    else
        CUDA_VISIBLE_DEVICES="$GPU_DEVICE" SHOW_PROGRESS="$SHOW_PROGRESS" "$PYTHON_BIN" -m torch.distributed.run \
            --standalone \
            --nnodes 1 \
            --nproc_per_node "$NPROC_PER_MODEL" \
            --master_port "$MASTER_PORT" \
            "$PROJECT_ROOT/eval/qv_lora_routing/dump_qv_lora_routing_eval.py" \
            --load "$MODEL_WEIGHTS_DIR" \
            --output-json "$output_json" \
            --compare-label "$label" \
            --source-num-experts "$SOURCE_NUM_EXPERTS" \
            --attn-lora-num-experts "$ATTN_LORA_NUM_EXPERTS" \
            --attn-lora-topk "$ATTN_LORA_TOPK" \
            --attn-lora-rank "$ATTN_LORA_RANK" \
            --attn-lora-alpha "$ATTN_LORA_ALPHA" \
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
            --transformer-impl local \
            --spec megatron.core.models.gpt.qv_lora_layer_specs gpt_qv_lora_local_spec \
            --bf16 \
            --no-persist-layer-norm \
            --no-gradient-accumulation-fusion \
            --no-masked-softmax-fusion \
            --attention-softmax-in-fp32 \
            --no-load-optim \
            --no-load-rng \
            --exit-on-missing-checkpoint \
            --test-mode > "$run_log" 2>&1
    fi
    cd "$PROJECT_ROOT"
}

echo "QV LoRA routing profile (bf16)"
echo "  model run dir:         $MODEL_RUN_DIR"
echo "  model weights dir:     $MODEL_WEIGHTS_DIR"
echo "  source num experts:    $SOURCE_NUM_EXPERTS"
echo "  target num experts:    $TARGET_NUM_EXPERTS"
echo "  wiki eval dataset:     $WIKI_EVAL_DATASET"
echo "  code eval dataset:     $CODE_EVAL_DATASET"
echo "  output root:           $OUTPUT_ROOT"
echo "  eval iters:            $EVAL_ITERS"
echo "  effective eval tokens: $EFFECTIVE_EVAL_TOKENS"

run_dump "wiki_test" "$WIKI_EVAL_DATASET"
run_dump "code_test" "$CODE_EVAL_DATASET"

"$PYTHON_BIN" "$PROJECT_ROOT/eval/qv_lora_routing/plot_qv_lora_routing.py" \
    --wiki-summary "$OUTPUT_ROOT/wiki_test/summary.json" \
    --code-summary "$OUTPUT_ROOT/code_test/summary.json" \
    --output-dir "$OUTPUT_ROOT/comparison"

echo "QV LoRA routing profile complete. Outputs under: $OUTPUT_ROOT"
