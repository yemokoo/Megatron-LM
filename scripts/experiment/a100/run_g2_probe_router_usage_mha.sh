#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

DEFAULT_CHECKPOINT_RUN_ID="g2-ts-routerkd-allrouter-kl10p0-log20-save60-1800"
BASE_WEIGHTS_DIR="$PROJECT_ROOT/.local/weights/a100/mha/shared-router-granularity-qkvo"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$BASE_WEIGHTS_DIR/code/$DEFAULT_CHECKPOINT_RUN_ID}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ID="${RUN_ID:-router-usage-code-${DEFAULT_CHECKPOINT_RUN_ID}-${TIMESTAMP}}"
RUN_NAME="${RUN_NAME:-Router usage on code probe - ${DEFAULT_CHECKPOINT_RUN_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_WEIGHTS_DIR/diagnostics/$RUN_ID}"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

export HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-5472}"
export NUM_LAYERS="${NUM_LAYERS:-9}"
export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-8}"
export NUM_EXPERTS="${NUM_EXPERTS:-16}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_LAYER_FREQ="${MOE_LAYER_FREQ:-[0,1,1,1,1,1,1,1,1]}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"
export ATTN_LORA_GROUPED_GEMM="${ATTN_LORA_GROUPED_GEMM:-1}"
export MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}"
export ATTN_LORA_RANK="${ATTN_LORA_RANK:-256}"
export ATTN_LORA_ALPHA="${ATTN_LORA_ALPHA:-256}"
export ATTN_FULL_RANK_LORA_RANK="${ATTN_FULL_RANK_LORA_RANK:-256}"
export ATTN_FULL_RANK_LORA_ALPHA="${ATTN_FULL_RANK_LORA_ALPHA:-256}"
export ATTN_FULL_RANK_LORA_TARGETS="${ATTN_FULL_RANK_LORA_TARGETS:-qkvo}"
export ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="${ATTN_FULL_RANK_LORA_ACTIVE_TARGETS:-}"

source "$PROJECT_ROOT/configs/model/flame-shared-router-hybrid-experts.sh"

PROBE_DATASET="${PROBE_DATASET:-$(probe_dir_for_task code)}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-72}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29732}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"

WANDB_ARGS=(
    --wandb-project "$WANDB_PROJECT"
    --wandb-exp-name "$RUN_NAME"
    --wandb-save-dir "$OUTPUT_DIR/wandb"
    --wandb-run-id "$RUN_ID"
    --wandb-resume allow
)

echo "[CONFIG] G2 shared-router code router-usage diagnostic"
echo "[CONFIG] checkpoint=$CHECKPOINT_DIR"
echo "[CONFIG] probe_dataset=$PROBE_DATASET"
echo "[CONFIG] probe_eval_iters=$PROBE_EVAL_ITERS, old_experts=$SOURCE_NUM_EXPERTS, num_experts=$NUM_EXPERTS"
echo "[CONFIG] output=$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
WANDB_MODE="$WANDB_MODE" \
torchrun \
    --nproc_per_node "$NPROC_PER_NODE" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    Megatron-LM/pretrain_gpt.py \
    "${MODEL_ARGS[@]}" \
    --transformer-impl local \
    --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --distributed-timeout-minutes 30 \
    --no-persist-layer-norm \
    --bf16 \
    --micro-batch-size "$MICRO_BATCH_SIZE" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --lr 3e-4 \
    --min-lr 3e-5 \
    --lr-decay-style WSD \
    --lr-decay-iters 1 \
    --lr-warmup-fraction 0.01 \
    --lr-wsd-decay-iters 1 \
    --train-iters 1 \
    --skip-train \
    --shared-router-hybrid-resume-from-num-experts "$SOURCE_NUM_EXPERTS" \
    --shared-router-hybrid-train-new-experts-and-router-only \
    --seq-length "${SEQ_LENGTH:-512}" \
    --data-path $(build_data_path "$PROBE_DATASET") \
    --split 100,0,0 \
    --log-interval 1 \
    --load "$CHECKPOINT_DIR" \
    --eval-interval 1 \
    --eval-iters 0 \
    --tensorboard-dir "$OUTPUT_DIR" \
    --no-load-optim \
    --no-load-rng \
    --finetune \
    --probe-name code_probe \
    --probe-eval-iters "$PROBE_EVAL_ITERS" \
    --probe-eval-interval 1 \
    --probe-step-offset 0 \
    --probe-data-path $(build_data_path "$PROBE_DATASET") \
    --run-initial-probe-eval \
    --probe-router-usage \
    --probe-router-usage-num-existing-experts "$SOURCE_NUM_EXPERTS" \
    "${WANDB_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/run.log"

echo "[DONE] router usage diagnostic $(date)"
