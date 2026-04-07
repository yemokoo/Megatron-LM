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

read_run_metadata_field() {
    "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
field_spec = sys.argv[2]
metadata_path = run_dir / "logs" / "run_metadata.json"
if not metadata_path.exists():
    raise SystemExit(0)
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
for key in field_spec.split(","):
    key = key.strip()
    if key and key in metadata and metadata[key] is not None:
        print(metadata[key])
        raise SystemExit(0)
raise SystemExit(0)
PY
}

dataset_blend_args() {
  find "$1" -type f -name '*.bin' -exec sh -c 'printf "1.0 %s " "${1%.bin}"' _ {} \; | sed 's/ $//'
}

truthy() {
    case "${1:-}" in
        1|true|True|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

export PYTHON_BIN="${PYTHON_BIN:-$(resolve_python)}"
export MODEL_WIKI_RUN_DIR="${MODEL_WIKI_RUN_DIR:?MODEL_WIKI_RUN_DIR must be set}"
export MODEL_CODE_RUN_DIR="${MODEL_CODE_RUN_DIR:?MODEL_CODE_RUN_DIR must be set}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/analysis_outputs/shared_router_pair_transition/$(basename "$MODEL_CODE_RUN_DIR")}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(basename "$MODEL_CODE_RUN_DIR")}"
export TASK_WIKI="${TASK_WIKI:-$PROJECT_ROOT/data/wiki/test}"
export TARGET_EVAL_TOKENS="${TARGET_EVAL_TOKENS:-1000000}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-96}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-96}"
export EVAL_SEQ_LENGTH="${EVAL_SEQ_LENGTH:-512}"
export GPU_DEVICE="${GPU_DEVICE:-0}"
export MASTER_PORT_BASE="${MASTER_PORT_BASE:-29980}"
export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-$(read_run_metadata_field "$MODEL_WIKI_RUN_DIR" 'num_experts')}"
export TOTAL_NUM_EXPERTS="${TOTAL_NUM_EXPERTS:-$(read_run_metadata_field "$MODEL_CODE_RUN_DIR" 'num_experts')}"
export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-4}"
export TOTAL_NUM_EXPERTS="${TOTAL_NUM_EXPERTS:-7}"

export WIKI_ATTN_LORA_NUM_EXPERTS="${WIKI_ATTN_LORA_NUM_EXPERTS:-$(read_run_metadata_field "$MODEL_WIKI_RUN_DIR" 'attn_lora_target_num_experts,attn_lora_num_experts,num_experts')}"
export WIKI_ATTN_LORA_RANK="${WIKI_ATTN_LORA_RANK:-$(read_run_metadata_field "$MODEL_WIKI_RUN_DIR" 'attn_lora_rank')}"
export WIKI_ATTN_LORA_TOPK="${WIKI_ATTN_LORA_TOPK:-$(read_run_metadata_field "$MODEL_WIKI_RUN_DIR" 'attn_lora_topk,moe_router_topk')}"
export WIKI_ATTN_LORA_ALPHA="${WIKI_ATTN_LORA_ALPHA:-$(read_run_metadata_field "$MODEL_WIKI_RUN_DIR" 'attn_lora_alpha')}"
export WIKI_ATTN_LORA_INCLUDE_PROJ="${WIKI_ATTN_LORA_INCLUDE_PROJ:-$(read_run_metadata_field "$MODEL_WIKI_RUN_DIR" 'attn_lora_include_proj')}"
export WIKI_ATTN_LORA_NUM_EXPERTS="${WIKI_ATTN_LORA_NUM_EXPERTS:-$SOURCE_NUM_EXPERTS}"
export WIKI_ATTN_LORA_RANK="${WIKI_ATTN_LORA_RANK:-16}"
export WIKI_ATTN_LORA_TOPK="${WIKI_ATTN_LORA_TOPK:-2}"
export WIKI_ATTN_LORA_ALPHA="${WIKI_ATTN_LORA_ALPHA:-16}"
export WIKI_ATTN_LORA_INCLUDE_PROJ="${WIKI_ATTN_LORA_INCLUDE_PROJ:-0}"

export CODE_ATTN_LORA_NUM_EXPERTS="${CODE_ATTN_LORA_NUM_EXPERTS:-$(read_run_metadata_field "$MODEL_CODE_RUN_DIR" 'attn_lora_target_num_experts,attn_lora_num_experts,num_experts')}"
export CODE_ATTN_LORA_RANK="${CODE_ATTN_LORA_RANK:-$(read_run_metadata_field "$MODEL_CODE_RUN_DIR" 'attn_lora_rank')}"
export CODE_ATTN_LORA_TOPK="${CODE_ATTN_LORA_TOPK:-$(read_run_metadata_field "$MODEL_CODE_RUN_DIR" 'attn_lora_topk,moe_router_topk')}"
export CODE_ATTN_LORA_ALPHA="${CODE_ATTN_LORA_ALPHA:-$(read_run_metadata_field "$MODEL_CODE_RUN_DIR" 'attn_lora_alpha')}"
export CODE_ATTN_LORA_INCLUDE_PROJ="${CODE_ATTN_LORA_INCLUDE_PROJ:-$(read_run_metadata_field "$MODEL_CODE_RUN_DIR" 'attn_lora_include_proj')}"
export CODE_ATTN_LORA_NUM_EXPERTS="${CODE_ATTN_LORA_NUM_EXPERTS:-$TOTAL_NUM_EXPERTS}"
export CODE_ATTN_LORA_RANK="${CODE_ATTN_LORA_RANK:-16}"
export CODE_ATTN_LORA_TOPK="${CODE_ATTN_LORA_TOPK:-2}"
export CODE_ATTN_LORA_ALPHA="${CODE_ATTN_LORA_ALPHA:-16}"
export CODE_ATTN_LORA_INCLUDE_PROJ="${CODE_ATTN_LORA_INCLUDE_PROJ:-0}"

if [ -z "$SOURCE_NUM_EXPERTS" ] || [ -z "$TOTAL_NUM_EXPERTS" ]; then
  echo "ERROR: failed to resolve SOURCE_NUM_EXPERTS / TOTAL_NUM_EXPERTS"
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
WIKI_DATA_PATH="$(dataset_blend_args "$TASK_WIKI")"
WIKI_DUMP_PT="$OUTPUT_ROOT/wiki_pairs.pt"
CODE_DUMP_PT="$OUTPUT_ROOT/code_pairs.pt"

run_dump() {
  local run_dir="$1"
  local output_pt="$2"
  local compare_label="$3"
  local port="$4"
  local num_experts="$5"
  local rank="$6"
  local topk="$7"
  local alpha="$8"
  local include_proj="$9"

  args=(
    analysis/dump_shared_router_pairs.py
    --load "$run_dir"
    --output-pt "$output_pt"
    --compare-label "$compare_label"
    --data-path
    --target-eval-tokens "$TARGET_EVAL_TOKENS"
    --max-batches 8
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --seq-length "$EVAL_SEQ_LENGTH"
    --dataset-split 100,0,0
    --dataset-split-name train
    --consumed-samples 0
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 1
    --tensor-model-parallel-size 1
    --transformer-impl local
    --spec megatron.core.models.gpt.shared_router_hybrid_layer_specs gpt_shared_router_hybrid_local_spec
    --shared-router-hybrid-model
    --attn-lora-num-experts "$num_experts"
    --attn-lora-rank "$rank"
    --attn-lora-topk "$topk"
    --attn-lora-alpha "$alpha"
    --bf16
    --no-persist-layer-norm
    --no-gradient-accumulation-fusion
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
    --no-load-optim
    --no-load-rng
    --exit-on-missing-checkpoint
  )
  if truthy "$include_proj"; then
    args+=(--attn-lora-include-proj)
  fi

  CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nnodes 1 \
    --nproc_per_node 1 \
    --master_port "$port" \
    "${args[@]}" \
    $WIKI_DATA_PATH
}

run_dump \
  "$MODEL_WIKI_RUN_DIR" \
  "$WIKI_DUMP_PT" \
  "${EXPERIMENT_NAME}__wiki_pairs" \
  "$MASTER_PORT_BASE" \
  "$WIKI_ATTN_LORA_NUM_EXPERTS" \
  "$WIKI_ATTN_LORA_RANK" \
  "$WIKI_ATTN_LORA_TOPK" \
  "$WIKI_ATTN_LORA_ALPHA" \
  "$WIKI_ATTN_LORA_INCLUDE_PROJ"

run_dump \
  "$MODEL_CODE_RUN_DIR" \
  "$CODE_DUMP_PT" \
  "${EXPERIMENT_NAME}__code_pairs" \
  "$((MASTER_PORT_BASE + 1))" \
  "$CODE_ATTN_LORA_NUM_EXPERTS" \
  "$CODE_ATTN_LORA_RANK" \
  "$CODE_ATTN_LORA_TOPK" \
  "$CODE_ATTN_LORA_ALPHA" \
  "$CODE_ATTN_LORA_INCLUDE_PROJ"

"$PYTHON_BIN" analysis/compare_shared_router_pair_transitions.py \
  --wiki-routing-pt "$WIKI_DUMP_PT" \
  --code-routing-pt "$CODE_DUMP_PT" \
  --output-root "$OUTPUT_ROOT" \
  --compare-label "$EXPERIMENT_NAME" \
  --source-num-experts "$SOURCE_NUM_EXPERTS" \
  --total-num-experts "$TOTAL_NUM_EXPERTS"

echo "Saved outputs under: $OUTPUT_ROOT"
echo "Aggregate heatmap: $OUTPUT_ROOT/aggregate_transition_heatmap.png"
echo "Layer heatmaps:    $OUTPUT_ROOT/layer_transition_heatmaps.png"
