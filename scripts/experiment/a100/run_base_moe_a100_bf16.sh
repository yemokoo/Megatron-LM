#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

export TASK_NAME="${TASK_NAME:?TASK_NAME must be set to wiki or code}"
export TASK_LABEL="${TASK_LABEL:-$(stage_label_for_task "$TASK_NAME")}"
export RUN_ID="${RUN_ID:-${TASK_LABEL}-a100-bf16-$(date -u +%Y%m%d-%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29610}"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_DATASET="${LOCAL_DATASET:-$LOCAL_BASE/dataset}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"

export SSD_MOUNT="${LOCAL_SSD_ROOT}/${RUN_ID}"
export SSD_TRAIN_DATASET="${SSD_MOUNT}/dataset/train"
export SSD_WEIGHTS="${SSD_MOUNT}/weights"

export NUM_LAYERS="${NUM_LAYERS:-9}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-5472}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-704}"
export MOE_LAYER_FREQ="${MOE_LAYER_FREQ:-[0]*1+[1]*8}"
export NUM_EXPERTS="${NUM_EXPERTS:-4}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-2}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=1
export TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-local}"
export PRECISION="bf16"

export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-300}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
export GPU_LOG_INTERVAL_SECONDS="${GPU_LOG_INTERVAL_SECONDS:-30}"
export LR="${LR:-3e-4}"
export MIN_LR="${MIN_LR:-3e-5}"
export LR_DECAY_STYLE="${LR_DECAY_STYLE:-WSD}"
export LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.01}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-$TRAIN_ITERS}"
export LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-$((TRAIN_ITERS / 10))}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-scripts/experiment/a100/flame-moe-bf16-no-shared.sh}"
export DATASET_SPLIT="${DATASET_SPLIT:-100,0,0}"

export TRAIN_DATASET="${TRAIN_DATASET:-$(dataset_dir_for_task "$TASK_NAME")}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$LOCAL_WEIGHTS/$(weights_subdir_for_task "$TASK_NAME")/$RUN_ID}"
export LOG_DIR="${LOG_DIR:-$TRAIN_WEIGHTS/logs}"
export RUN_LOG="${RUN_LOG:-$LOG_DIR/${TASK_LABEL}.log}"
export GPU_LOG="${GPU_LOG:-$LOG_DIR/gpu_usage.csv}"
export RUN_METADATA="${RUN_METADATA:-$LOG_DIR/run_metadata.json}"
export DATASET_NAME="${DATASET_NAME:-$(dataset_name_for_task "$TASK_NAME")}"
export DATASET_SOURCE="${DATASET_SOURCE:-$(dataset_source_for_task "$TASK_NAME")}"
export PROBE_DATASET="${PROBE_DATASET:-$(probe_dir_for_task "$TASK_NAME")}"
export PROBE_NAME="${PROBE_NAME:-${TASK_LABEL}_probe}"
export PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-25}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-40}"

if [ "$TASK_NAME" = "wiki" ]; then
    export SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET:-$(probe_dir_for_task code)}"
    export SECONDARY_PROBE_NAME="${SECONDARY_PROBE_NAME:-code_probe}"
else
    export SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET:-$(probe_dir_for_task wiki)}"
    export SECONDARY_PROBE_NAME="${SECONDARY_PROBE_NAME:-wiki_probe}"
fi
export SECONDARY_PROBE_EVAL_ITERS="${SECONDARY_PROBE_EVAL_ITERS:-25}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-40}"
export SECONDARY_PROBE_STEP_OFFSET="${SECONDARY_PROBE_STEP_OFFSET:-0}"
export WANDB_STEP_OFFSET="${WANDB_STEP_OFFSET:-0}"
export WANDB_PROJECT="${WANDB_PROJECT:-}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-$RUN_ID}"
export WANDB_SAVE_DIR="${WANDB_SAVE_DIR:-$TRAIN_WEIGHTS/wandb}"
export STAGE_NAME="${STAGE_NAME:-$TASK_LABEL}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

mkdir -p "$SSD_TRAIN_DATASET" "$SSD_WEIGHTS" "$TRAIN_WEIGHTS" "$LOG_DIR"
exec > >(
    tee -a "$RUN_LOG" | "$PYTHON_BIN" -u -c '
import re, sys
iter_re = re.compile(r"(\[[^]]+\]) iteration\s+(\d+)/\s*(\d+).*throughput per GPU \(TFLOP/s/GPU\):\s*([0-9.]+)")
val_re = re.compile(r"validation loss at iteration\s+(\d+).*lm loss value:\s*([^|]+)")
save_re = re.compile(r"saving checkpoint at iteration\s+(\d+)")
keep_re = re.compile(r"a100 bf16|ERROR:|Traceback|failed \(exitcode|checkpoint at|probe ")
for line in sys.stdin:
    line = line.rstrip("\n")
    m = iter_re.search(line)
    if m:
        print(f"{m.group(1)} step {m.group(2)}/{m.group(3)} | GPU {m.group(4)} TFLOP/s", flush=True)
        continue
    m = val_re.search(line)
    if m:
        print(f"validation step {m.group(1)} | lm loss {m.group(2).strip()}", flush=True)
        continue
    m = save_re.search(line)
    if m:
        print(f"saving checkpoint step {m.group(1)}", flush=True)
        continue
    if keep_re.search(line):
        print(line, flush=True)
'
) 2>&1

echo "a100 bf16 run log: $RUN_LOG"
echo "a100 bf16 gpu log: $GPU_LOG"
echo "a100 bf16 metadata: $RUN_METADATA"
rsync -rlptD --info=progress2 "$TRAIN_DATASET/" "$SSD_TRAIN_DATASET/"

write_base_metadata
source "$MODEL_CONFIG_SCRIPT"

INFRA_ARGS=(
    --transformer-impl "$TRANSFORMER_IMPL"
    --pipeline-model-parallel-size "$PIPELINE_MODEL_PARALLEL_SIZE"
    --expert-model-parallel-size "$EXPERT_MODEL_PARALLEL_SIZE"
    --moe-token-dispatcher-type alltoall
    --distributed-timeout-minutes 30
    --no-persist-layer-norm
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
)

TRAIN_ARGS=(
    --bf16
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --lr "$LR"
    --min-lr "$MIN_LR"
    --lr-decay-style "$LR_DECAY_STYLE"
    --lr-decay-iters "$LR_DECAY_ITERS"
    --lr-warmup-fraction "$LR_WARMUP_FRACTION"
    --lr-wsd-decay-iters "$LR_WSD_DECAY_ITERS"
    --train-iters "$TRAIN_ITERS"
)

DATA_ARGS=(
    --seq-length "${SEQ_LENGTH:-512}"
    --data-path $(build_data_path "$SSD_TRAIN_DATASET")
    --split "$DATASET_SPLIT"
)

SAVE_ARGS=(
    --log-interval 10
    --log-throughput
    --log-progress
    --save "$SSD_WEIGHTS"
    --save-interval "$SAVE_INTERVAL"
    --load "$SSD_WEIGHTS"
    --eval-interval "$EVAL_INTERVAL"
    --tensorboard-dir "$SSD_WEIGHTS"
)

PROBE_ARGS=(
    --probe-name "$PROBE_NAME"
    --probe-eval-iters "$PROBE_EVAL_ITERS"
    --probe-eval-interval "$PROBE_EVAL_INTERVAL"
    --probe-data-path $(build_data_path "$PROBE_DATASET")
)

if [ -n "$SECONDARY_PROBE_DATASET" ]; then
    PROBE_ARGS+=(
        --secondary-probe-name "$SECONDARY_PROBE_NAME"
        --secondary-probe-eval-iters "$SECONDARY_PROBE_EVAL_ITERS"
        --secondary-probe-eval-interval "$SECONDARY_PROBE_EVAL_INTERVAL"
        --secondary-probe-step-offset "$SECONDARY_PROBE_STEP_OFFSET"
        --secondary-probe-data-path $(build_data_path "$SECONDARY_PROBE_DATASET")
    )
fi

WANDB_ARGS=()
if [ -n "$WANDB_PROJECT" ]; then
    WANDB_ARGS+=(
        --wandb-project "$WANDB_PROJECT"
        --wandb-exp-name "$WANDB_EXP_NAME"
        --wandb-save-dir "$WANDB_SAVE_DIR"
        --wandb-step-offset "$WANDB_STEP_OFFSET"
    )
fi

cd Megatron-LM
(
    while true; do
        nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits >> "$GPU_LOG"
        sleep "$GPU_LOG_INTERVAL_SECONDS"
    done
) &
GPU_LOG_PID=$!

"$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nnodes 1 \
    --nproc_per_node "$NPROC_PER_NODE" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    pretrain_gpt.py \
    "${MODEL_ARGS[@]}" "${INFRA_ARGS[@]}" "${TRAIN_ARGS[@]}" \
    "${DATA_ARGS[@]}" "${SAVE_ARGS[@]}" "${PROBE_ARGS[@]}" "${WANDB_ARGS[@]}"

kill "$GPU_LOG_PID" 2>/dev/null || true
rsync -rlptD "$SSD_WEIGHTS/" "$TRAIN_WEIGHTS/"

echo "a100 bf16 base training complete. checkpoint at: $TRAIN_WEIGHTS"
