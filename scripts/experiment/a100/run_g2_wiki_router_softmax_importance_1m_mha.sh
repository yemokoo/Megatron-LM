#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

source "$SCRIPT_DIR/common.sh"

export PYTHONPATH="$PROJECT_ROOT/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29790}"

export G2_ROOT="${G2_ROOT:-$PROJECT_ROOT/.local/weights/a100/mha/g2-checkpoints}"
export WIKI_WEIGHTS="${WIKI_WEIGHTS:-$G2_ROOT/wiki/g2-top4-e8-ffn352-r256-wiki-shared-router-qkvo-mha-a100-bf16-mb72-1800}"
export TRAIN_DATASET="${TRAIN_DATASET:-$(dataset_dir_for_task wiki)}"
export OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/analysis_outputs/g2_wiki_router_softmax_importance_1m}"

export NUM_LAYERS="${NUM_LAYERS:-9}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-5472}"
export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export NUM_EXPERTS="${NUM_EXPERTS:-8}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_LAYER_FREQ="${MOE_LAYER_FREQ:-[0,1,1,1,1,1,1,1,1]}"
export ATTN_LORA_RANK="${ATTN_LORA_RANK:-256}"
export ATTN_LORA_ALPHA="${ATTN_LORA_ALPHA:-$ATTN_LORA_RANK}"
export ATTN_FULL_RANK_LORA_RANK="${ATTN_FULL_RANK_LORA_RANK:-256}"
export ATTN_FULL_RANK_LORA_ALPHA="${ATTN_FULL_RANK_LORA_ALPHA:-$ATTN_FULL_RANK_LORA_RANK}"
export ATTN_FULL_RANK_LORA_TARGETS="${ATTN_FULL_RANK_LORA_TARGETS:-qkvo}"
export ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="${ATTN_FULL_RANK_LORA_ACTIVE_TARGETS:-}"
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"
export ATTN_LORA_GROUPED_GEMM="${ATTN_LORA_GROUPED_GEMM:-1}"
export MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-72}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=1
export TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-local}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-configs/model/flame-shared-router-hybrid-experts.sh}"

export ROUTER_SOFTMAX_LAYERS="${ROUTER_SOFTMAX_LAYERS:-2,3,4,5,6,7,8,9}"
export ROUTER_SOFTMAX_EVAL_ITERS="${ROUTER_SOFTMAX_EVAL_ITERS:-50}"
export ROUTER_SOFTMAX_MAX_TOKENS="${ROUTER_SOFTMAX_MAX_TOKENS:-1048576}"
export ROUTER_SOFTMAX_MASK_TOPK="${ROUTER_SOFTMAX_MASK_TOPK:-4}"
export LOG_INTERVAL="${LOG_INTERVAL:-5}"

if [ ! -f "$WIKI_WEIGHTS/latest_checkpointed_iteration.txt" ]; then
    echo "[ERROR] wiki checkpoint tracker is missing: $WIKI_WEIGHTS/latest_checkpointed_iteration.txt" >&2
    exit 1
fi

echo "[CONFIG] G2 wiki router softmax importance"
echo "[CONFIG] checkpoint=$WIKI_WEIGHTS"
echo "[CONFIG] data=$TRAIN_DATASET"
echo "[CONFIG] out_dir=$OUT_DIR"
echo "[CONFIG] layers=$ROUTER_SOFTMAX_LAYERS, topk=$ROUTER_SOFTMAX_MASK_TOPK, max_tokens=$ROUTER_SOFTMAX_MAX_TOKENS"

mkdir -p "$OUT_DIR"
source "$MODEL_CONFIG_SCRIPT"

torchrun \
    --nproc_per_node "$NPROC_PER_NODE" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    scripts/analysis/plot_wiki_router_softmax_importance.py \
    "${MODEL_ARGS[@]}" \
    --transformer-impl "$TRANSFORMER_IMPL" \
    --pipeline-model-parallel-size "$PIPELINE_MODEL_PARALLEL_SIZE" \
    --expert-model-parallel-size "$EXPERT_MODEL_PARALLEL_SIZE" \
    --distributed-timeout-minutes 30 \
    --no-persist-layer-norm \
    --bf16 \
    --micro-batch-size "$MICRO_BATCH_SIZE" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --train-iters 1 \
    --seq-length "${SEQ_LENGTH:-512}" \
    --data-path $(build_data_path "$TRAIN_DATASET") \
    --split 100,0,0 \
    --log-interval "$LOG_INTERVAL" \
    --load "$WIKI_WEIGHTS" \
    --no-load-optim \
    --no-load-rng \
    --eval-interval 1 \
    --router-softmax-out-dir "$OUT_DIR" \
    --router-softmax-eval-iters "$ROUTER_SOFTMAX_EVAL_ITERS" \
    --router-softmax-max-tokens "$ROUTER_SOFTMAX_MAX_TOKENS" \
    --router-softmax-mask-topk "$ROUTER_SOFTMAX_MASK_TOPK" \
    --router-softmax-layers "$ROUTER_SOFTMAX_LAYERS"

echo "[DONE] wrote router softmax importance outputs to $OUT_DIR"
