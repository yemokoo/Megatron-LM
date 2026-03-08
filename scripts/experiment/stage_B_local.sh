#!/bin/bash
set -euo pipefail

# Stage B (local): expand a Stage A FLAME-MoE checkpoint to a larger expert count,
# freeze the original experts/router rows, and continue training on code data.
# Example:
#   LOCAL_BASE=/mnt/nas/flame-moe \
#   LOCAL_SSD_ROOT=/local_scratch/flame-moe \
#   STAGE_A_RUN_ID=stage-a-local-20260306-120000 \
#   NPROC_PER_NODE=8 \
#   bash scripts/experiment/stage_B_local.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export RUN_ID="${RUN_ID:-stage-b-local-$(date -u +%Y%m%d-%H%M%S)}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29501}"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_DATASET="${LOCAL_DATASET:-$LOCAL_BASE/dataset}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"

export SSD_MOUNT="${LOCAL_SSD_ROOT}/${RUN_ID}"
export SSD_DATASET="${SSD_MOUNT}/dataset"
export SSD_WEIGHTS="${SSD_MOUNT}/weights"

if [ -z "${STAGE_A_WEIGHTS_DIR:-}" ] && [ -z "${STAGE_A_RUN_ID:-}" ]; then
    echo "ERROR: set STAGE_A_RUN_ID or STAGE_A_WEIGHTS_DIR."
    exit 1
fi

# FLAME-MoE-38M architecture
export NUM_LAYERS=9
export HIDDEN_SIZE=256
export FFN_HIDDEN_SIZE=1368
export MOE_FFN_HIDDEN_SIZE=176
export MOE_LAYER_FREQ="[0]*1+[1]*8"
export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-4}"
export NUM_EXPERTS="${NUM_EXPERTS:-7}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-2}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=1
export TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-local}"

export TRAIN_ITERS="${TRAIN_ITERS:-2121}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-212}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-212}"
export OLD_MODEL_KL_COEFF="${OLD_MODEL_KL_COEFF:-0.01}"
export OLD_MODEL_KL_TEMPERATURE="${OLD_MODEL_KL_TEMPERATURE:-1.0}"

export TRAIN_DATASET="${TRAIN_DATASET:-$LOCAL_DATASET/python-code/tokenized/EleutherAI/pythia-12b}"
STAGE_A_WEIGHTS_DIR="${STAGE_A_WEIGHTS_DIR:-$LOCAL_WEIGHTS/continual-stage-A-local/$STAGE_A_RUN_ID}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$LOCAL_WEIGHTS/continual-stage-B-local/$RUN_ID}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-8}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"

if [ "$SOURCE_NUM_EXPERTS" -ge "$NUM_EXPERTS" ]; then
    echo "ERROR: SOURCE_NUM_EXPERTS must be smaller than NUM_EXPERTS."
    exit 1
fi

mkdir -p "$SSD_DATASET" "$SSD_WEIGHTS" "$TRAIN_WEIGHTS"
rsync -a "$STAGE_A_WEIGHTS_DIR/" "$SSD_WEIGHTS/"
rsync -a --info=progress2 "$TRAIN_DATASET/" "$SSD_DATASET/"

source configs/model/flame-moe.sh

INFRA_ARGS=(
    --transformer-impl "$TRANSFORMER_IMPL"
    --pipeline-model-parallel-size "$PIPELINE_MODEL_PARALLEL_SIZE"
    --expert-model-parallel-size "$EXPERT_MODEL_PARALLEL_SIZE"
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --moe-token-dispatcher-type alltoall
    --distributed-timeout-minutes 30
    --bf16
)

TRAIN_ARGS=(
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "${GLOBAL_BATCH_SIZE:-1024}"
    --lr "${LR:-3e-4}"
    --min-lr "${MIN_LR:-3e-5}"
    --lr-decay-style WSD
    --lr-warmup-fraction "${LR_WARMUP_FRACTION:-0.01}"
    --lr-wsd-decay-iters "$((TRAIN_ITERS / 10))"
    --train-iters "$TRAIN_ITERS"
)

DATA_ARGS=(
    --seq-length "${SEQ_LENGTH:-2048}"
    --data-path $(find "$SSD_DATASET" -type f -name '*.bin' -exec sh -c 'printf "1.0 %s " "${1%.bin}"' _ {} \; | sed 's/ $//')
    --split 95,5,0
)

SAVE_ARGS=(
    --log-interval 10
    --log-throughput
    --save "$SSD_WEIGHTS"
    --save-interval "$SAVE_INTERVAL"
    --load "$SSD_WEIGHTS"
    --eval-interval "$EVAL_INTERVAL"
    --tensorboard-dir "$SSD_WEIGHTS"
    --no-load-optim
    --no-load-rng
    --finetune
    --moe-expand-from-num-experts "$SOURCE_NUM_EXPERTS"
    --moe-freeze-existing-experts
    --moe-freeze-existing-router
    --moe-old-model-kl-coeff "$OLD_MODEL_KL_COEFF"
    --moe-old-model-kl-temperature "$OLD_MODEL_KL_TEMPERATURE"
)

cd Megatron-LM
torchrun \
    --standalone \
    --nnodes 1 \
    --nproc_per_node "$NPROC_PER_NODE" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    pretrain_gpt.py \
    "${MODEL_ARGS[@]}" "${INFRA_ARGS[@]}" "${TRAIN_ARGS[@]}" \
    "${DATA_ARGS[@]}" "${SAVE_ARGS[@]}" &
TORCHRUN_PID=$!

(
    while kill -0 "$TORCHRUN_PID" 2>/dev/null; do
        rsync -a "$SSD_WEIGHTS/" "$TRAIN_WEIGHTS/"
        sleep 15m
    done
) &

wait "$TORCHRUN_PID"
rsync -a "$SSD_WEIGHTS/" "$TRAIN_WEIGHTS/"

echo "Stage B local complete. Checkpoint at: $TRAIN_WEIGHTS"
