#!/bin/bash
set -euo pipefail

# Apex-enabled backup of the Stage A local launcher.
# Keep this variant around so the current local-safe script can disable
# Apex-dependent optimizations without losing the optimized path.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export RUN_ID="${RUN_ID:-stage-a-local-$(date -u +%Y%m%d-%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_DATASET="${LOCAL_DATASET:-$LOCAL_BASE/dataset}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"

export SSD_MOUNT="${LOCAL_SSD_ROOT}/${RUN_ID}"
export SSD_DATASET="${SSD_MOUNT}/dataset"
export SSD_WEIGHTS="${SSD_MOUNT}/weights"

export NUM_LAYERS="${NUM_LAYERS:-9}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-5504}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-1408}"
export MOE_LAYER_FREQ="${MOE_LAYER_FREQ:-[0]*1+[1]*8}"
export NUM_EXPERTS="${NUM_EXPERTS:-4}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-2}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=1
export TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-local}"
export PRECISION="${PRECISION:-fp16}"

export TRAIN_ITERS="${TRAIN_ITERS:-20}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"

export TRAIN_DATASET="${TRAIN_DATASET:-$LOCAL_DATASET/wikipedia/tokenized/EleutherAI/pythia-12b}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$LOCAL_WEIGHTS/continual-stage-A-local/$RUN_ID}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-8}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"

mkdir -p "$SSD_DATASET" "$SSD_WEIGHTS" "$TRAIN_WEIGHTS"
rsync -rlptD --info=progress2 "$TRAIN_DATASET/" "$SSD_DATASET/"

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
)

case "$PRECISION" in
    fp16)
        INFRA_ARGS+=(--fp16)
        ;;
    bf16)
        INFRA_ARGS+=(--bf16)
        ;;
    *)
        echo "ERROR: PRECISION must be 'fp16' or 'bf16'."
        exit 1
        ;;
esac

TRAIN_ARGS=(
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "${GLOBAL_BATCH_SIZE:-16}"
    --lr "${LR:-3e-4}"
    --min-lr "${MIN_LR:-3e-5}"
    --lr-decay-style WSD
    --lr-warmup-fraction "${LR_WARMUP_FRACTION:-0.01}"
    --lr-wsd-decay-iters "$((TRAIN_ITERS / 10))"
    --train-iters "$TRAIN_ITERS"
)

DATA_ARGS=(
    --seq-length "${SEQ_LENGTH:-512}"
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
)

cd Megatron-LM
python -m torch.distributed.run \
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
        rsync -rlptD "$SSD_WEIGHTS/" "$TRAIN_WEIGHTS/"
        sleep 15m
    done
) &

wait "$TORCHRUN_PID"
rsync -rlptD "$SSD_WEIGHTS/" "$TRAIN_WEIGHTS/"

echo "Stage A local complete. Checkpoint at: $TRAIN_WEIGHTS"
echo "To run Stage B locally:"
echo "  STAGE_A_RUN_ID=$RUN_ID bash scripts/experiment/stage_B_local.sh"
