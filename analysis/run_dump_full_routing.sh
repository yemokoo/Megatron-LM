#!/bin/bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.conda/envs/flame3090/bin/python}"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN="$(command -v python)"

LOAD="${LOAD:?}"
OUTPUT_PT="${OUTPUT_PT:?}"
LABEL="${LABEL:?}"
GPU="${GPU:?}"
PORT="${PORT:?}"
SPEC_MODULE="${SPEC_MODULE:-}"
SPEC_FN="${SPEC_FN:-}"
DATA_PATH="${DATA_PATH:?}"
ATTN_LORA_NUM_EXPERTS="${ATTN_LORA_NUM_EXPERTS:-}"
ATTN_LORA_TOPK="${ATTN_LORA_TOPK:-}"
ATTN_LORA_RANK="${ATTN_LORA_RANK:-}"
ATTN_FULL_RANK_LORA_RANK="${ATTN_FULL_RANK_LORA_RANK:-}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-}"
SPEC_ARGS=()
if [ -n "$SPEC_MODULE" ]; then
  SPEC_ARGS=(--spec "$SPEC_MODULE" "$SPEC_FN" --shared-router-hybrid-model)
  [ -n "$ATTN_LORA_NUM_EXPERTS" ] && SPEC_ARGS+=(--attn-lora-num-experts "$ATTN_LORA_NUM_EXPERTS")
  [ -n "$ATTN_LORA_TOPK" ] && SPEC_ARGS+=(--attn-lora-topk "$ATTN_LORA_TOPK")
  [ -n "$ATTN_LORA_RANK" ] && SPEC_ARGS+=(--attn-lora-rank "$ATTN_LORA_RANK")
  [ -n "$ATTN_FULL_RANK_LORA_RANK" ] && SPEC_ARGS+=(--attn-full-rank-lora-rank "$ATTN_FULL_RANK_LORA_RANK")
fi
TOKENIZER_ARGS=()
[ -n "$TOKENIZER_MODEL" ] && TOKENIZER_ARGS=(--tokenizer-model "$TOKENIZER_MODEL" --no-use-tokenizer-model-from-checkpoint-args)
TARGET_EVAL_TOKENS="${TARGET_EVAL_TOKENS:-300000}"
SEQ_LENGTH="${SEQ_LENGTH:-512}"
MICRO_BATCH="${MICRO_BATCH:-8}"
GLOBAL_BATCH="${GLOBAL_BATCH:-8}"
tokens_per_iter=$((SEQ_LENGTH * GLOBAL_BATCH))
EVAL_ITERS=$(((TARGET_EVAL_TOKENS + tokens_per_iter - 1) / tokens_per_iter))

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nnodes 1 --nproc_per_node 1 --master_port "$PORT" \
  "$PROJECT_ROOT/analysis/dump_full_routing.py" \
  --load "$LOAD" \
  --output-pt "$OUTPUT_PT" \
  --compare-label "$LABEL" \
  --dataset-split 100,0,0 --dataset-split-name train --consumed-samples 0 \
  --split 0,1,0 \
  --eval-iters "$EVAL_ITERS" \
  --micro-batch-size "$MICRO_BATCH" --global-batch-size "$GLOBAL_BATCH" \
  --pipeline-model-parallel-size 1 --expert-model-parallel-size 1 \
  --tensor-model-parallel-size 1 \
  --transformer-impl local --no-persist-layer-norm \
  --no-gradient-accumulation-fusion --no-masked-softmax-fusion \
  --attention-softmax-in-fp32 --bf16 \
  --no-load-optim --no-load-rng --exit-on-missing-checkpoint \
  "${SPEC_ARGS[@]}" \
  "${TOKENIZER_ARGS[@]}" \
  --data-path 1.0 "$DATA_PATH" \
  > "${OUTPUT_PT%.pt}.log" 2>&1
echo "done: $OUTPUT_PT (log: ${OUTPUT_PT%.pt}.log)"
