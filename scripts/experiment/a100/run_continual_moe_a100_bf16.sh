#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

export SOURCE_TASK="${SOURCE_TASK:?SOURCE_TASK must be set to wiki or code}"
export TARGET_TASK="${TARGET_TASK:?TARGET_TASK must be set to wiki or code}"
export FREEZE_SHARED="${FREEZE_SHARED:-0}"

if [ "$SOURCE_TASK" = "wiki" ] && [ "$TARGET_TASK" = "code" ]; then
    export STAGE_NAME="${STAGE_NAME:-a_to_b}"
    export STAGE_DIR_NAME="${STAGE_DIR_NAME:-a100/a-to-b-moe-bf16}"
    export STAGE_LABEL="${STAGE_LABEL:-a_to_b}"
elif [ "$SOURCE_TASK" = "code" ] && [ "$TARGET_TASK" = "wiki" ]; then
    export STAGE_NAME="${STAGE_NAME:-b_to_a}"
    export STAGE_DIR_NAME="${STAGE_DIR_NAME:-a100/b-to-a-moe-bf16}"
    export STAGE_LABEL="${STAGE_LABEL:-b_to_a}"
else
    echo "ERROR: unsupported continual direction ${SOURCE_TASK} -> ${TARGET_TASK}" >&2
    exit 1
fi

if [ "$FREEZE_SHARED" = "1" ]; then
    export STAGE_NAME="${STAGE_NAME}_freeze"
    export STAGE_DIR_NAME="${STAGE_DIR_NAME}-freeze"
    export STAGE_LABEL="${STAGE_LABEL}_freeze"
fi

export RUN_ID="${RUN_ID:-${STAGE_LABEL}-a100-bf16-$(date -u +%Y%m%d-%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29611}"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_DATASET="${LOCAL_DATASET:-$LOCAL_BASE/dataset}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
export DIRECT_LOCAL_SAVE="${DIRECT_LOCAL_SAVE:-0}"

export SSD_MOUNT="${LOCAL_SSD_ROOT}/${RUN_ID}"
export SSD_TRAIN_DATASET="${SSD_MOUNT}/dataset/train"
export SSD_SOURCE_WEIGHTS="${SSD_MOUNT}/source_weights"
export SSD_TARGET_WEIGHTS="${SSD_MOUNT}/target_weights"

export NUM_LAYERS="${NUM_LAYERS:-9}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-5472}"
export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-704}"
export MOE_LAYER_FREQ="${MOE_LAYER_FREQ:-[0]*1+[1]*8}"
export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-4}"
export NUM_EXPERTS="${NUM_EXPERTS:-7}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-2}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=1
export TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-local}"
export PRECISION="bf16"
export TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY="${TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY:-$FREEZE_SHARED}"

export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-300}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export OLD_MODEL_KL_COEFF="${OLD_MODEL_KL_COEFF:-1.0}"
export OLD_MODEL_KL_TEMPERATURE="${OLD_MODEL_KL_TEMPERATURE:-1.0}"
export GPU_LOG_INTERVAL_SECONDS="${GPU_LOG_INTERVAL_SECONDS:-30}"
export LR="${LR:-3e-4}"
export MIN_LR="${MIN_LR:-3e-5}"
export LR_DECAY_STYLE="${LR_DECAY_STYLE:-WSD}"
export LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.01}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-$TRAIN_ITERS}"
export LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-$((TRAIN_ITERS / 10))}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-scripts/experiment/a100/flame-moe-bf16-no-shared.sh}"
export DATASET_SPLIT="${DATASET_SPLIT:-100,0,0}"

export TRAIN_DATASET="${TRAIN_DATASET:-$(dataset_dir_for_task "$TARGET_TASK")}"
export SOURCE_RUN_SUBDIR="${SOURCE_RUN_SUBDIR:-$(weights_subdir_for_task "$SOURCE_TASK")}"
export SOURCE_WEIGHTS_DIR="${SOURCE_WEIGHTS_DIR:-}"
export SOURCE_REQUIRED_ITERS="${SOURCE_REQUIRED_ITERS:-1}"
export SOURCE_RUN_ID="${SOURCE_RUN_ID:-}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$LOCAL_WEIGHTS/$STAGE_DIR_NAME/$RUN_ID}"
export LOG_DIR="${LOG_DIR:-$TRAIN_WEIGHTS/logs}"
export RUN_LOG="${RUN_LOG:-$LOG_DIR/${STAGE_LABEL}.log}"
export GPU_LOG="${GPU_LOG:-$LOG_DIR/gpu_usage.csv}"
export RUN_METADATA="${RUN_METADATA:-$LOG_DIR/run_metadata.json}"
export DATASET_NAME="${DATASET_NAME:-$(dataset_name_for_task "$TARGET_TASK")}"
export DATASET_SOURCE="${DATASET_SOURCE:-$(dataset_source_for_task "$TARGET_TASK")}"
export PROBE_DATASET="${PROBE_DATASET:-$(probe_dir_for_task "$TARGET_TASK")}"
export PROBE_NAME="${PROBE_NAME:-${TARGET_TASK}_probe}"
export PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-25}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-100}"
export SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET:-$(probe_dir_for_task "$SOURCE_TASK")}"
export SECONDARY_PROBE_NAME="${SECONDARY_PROBE_NAME:-${SOURCE_TASK}_probe}"
export SECONDARY_PROBE_EVAL_ITERS="${SECONDARY_PROBE_EVAL_ITERS:-25}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-100}"
export PROBE_STEP_OFFSET="${PROBE_STEP_OFFSET:-}"
export SECONDARY_PROBE_STEP_OFFSET="${SECONDARY_PROBE_STEP_OFFSET:-}"
export RUN_INITIAL_PROBE_EVAL="${RUN_INITIAL_PROBE_EVAL:-1}"
export RUN_INITIAL_VALID_EVAL="${RUN_INITIAL_VALID_EVAL:-1}"
export WANDB_STEP_OFFSET="${WANDB_STEP_OFFSET:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-$RUN_ID}"
export WANDB_SAVE_DIR="${WANDB_SAVE_DIR:-$TRAIN_WEIGHTS/wandb}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$RUN_ID}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND="${LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND:-1}"
export RESUME_CONTINUAL_FROM_TRAIN_WEIGHTS="${RESUME_CONTINUAL_FROM_TRAIN_WEIGHTS:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

if [ "$DIRECT_LOCAL_SAVE" = "1" ]; then
    export SSD_TARGET_WEIGHTS="$TRAIN_WEIGHTS"
fi

export SOURCE_WEIGHTS_DIR="$(resolve_completed_run_dir)"
export PROBE_STEP_OFFSET="${PROBE_STEP_OFFSET:-$(read_train_iters_from_run)}"
export SECONDARY_PROBE_STEP_OFFSET="${SECONDARY_PROBE_STEP_OFFSET:-$PROBE_STEP_OFFSET}"
export WANDB_STEP_OFFSET="${WANDB_STEP_OFFSET:-$PROBE_STEP_OFFSET}"

RESUME_FROM_TARGET=0
if [ "$RESUME_CONTINUAL_FROM_TRAIN_WEIGHTS" = "1" ] && [ -f "$TRAIN_WEIGHTS/latest_checkpointed_iteration.txt" ]; then
    RESUME_FROM_TARGET=1
fi

case "$SOURCE_TASK" in
    wiki)
        export SOURCE_PRIMARY_PROBE_CANDIDATES="${SOURCE_PRIMARY_PROBE_CANDIDATES:-wiki_probe,wiki_a_probe}"
        ;;
    code)
        export SOURCE_PRIMARY_PROBE_CANDIDATES="${SOURCE_PRIMARY_PROBE_CANDIDATES:-code_probe,code_b_probe}"
        ;;
esac

case "$TARGET_TASK" in
    wiki)
        export SOURCE_SECONDARY_PROBE_CANDIDATES="${SOURCE_SECONDARY_PROBE_CANDIDATES:-wiki_probe,wiki_a_probe}"
        ;;
    code)
        export SOURCE_SECONDARY_PROBE_CANDIDATES="${SOURCE_SECONDARY_PROBE_CANDIDATES:-code_probe,code_b_probe}"
        ;;
esac

if [ "$LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND" = "1" ] && [ -n "$WANDB_PROJECT" ]; then
    export RUN_INITIAL_PROBE_EVAL=0
fi

mkdir -p "$SSD_TRAIN_DATASET" "$SSD_SOURCE_WEIGHTS" "$SSD_TARGET_WEIGHTS" "$TRAIN_WEIGHTS" "$LOG_DIR"

GPU_LOG_PID=""
SYNC_DONE=0
cleanup() {
    local exit_code=$?
    if [ -n "${GPU_LOG_PID:-}" ]; then
        kill "$GPU_LOG_PID" 2>/dev/null || true
    fi
    if [ "${SYNC_DONE:-0}" != "1" ] && [ -d "$SSD_TARGET_WEIGHTS" ] && [ "$SSD_TARGET_WEIGHTS" != "$TRAIN_WEIGHTS" ]; then
        rsync -rlptD "$SSD_TARGET_WEIGHTS/" "$TRAIN_WEIGHTS/" || true
        SYNC_DONE=1
    fi
    return "$exit_code"
}
trap cleanup EXIT INT TERM

exec > >(
    tee -a "$RUN_LOG" | "$PYTHON_BIN" -u -c '
import re, sys
iter_re = re.compile(r"(\[[^]]+\]) iteration\s+(\d+)/\s*(\d+).*throughput per GPU \(TFLOP/s/GPU\):\s*([0-9.]+)")
val_re = re.compile(r"validation loss at iteration\s+(\d+).*lm loss value:\s*([^|]+)")
save_re = re.compile(r"saving checkpoint at iteration\s+(\d+)")
keep_re = re.compile(r"a100 bf16 continual|ERROR:|Traceback|failed \(exitcode|checkpoint at|probe |Expanded MoE checkpoint")
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

echo "a100 bf16 continual run log: $RUN_LOG"
echo "a100 bf16 continual gpu log: $GPU_LOG"
echo "a100 bf16 continual metadata: $RUN_METADATA"
echo "a100 bf16 source checkpoint: $SOURCE_WEIGHTS_DIR"
if [ "$RESUME_FROM_TARGET" = "1" ]; then
    echo "a100 bf16 continual resume checkpoint: $TRAIN_WEIGHTS"
fi

rsync -rlptD \
    --exclude 'logs/' \
    --exclude 'wandb/' \
    --exclude 'events.out.tfevents*' \
    --exclude 'progress.txt' \
    "$SOURCE_WEIGHTS_DIR/" "$SSD_SOURCE_WEIGHTS/"
rsync -rlptD --info=progress2 "$TRAIN_DATASET/" "$SSD_TRAIN_DATASET/"
if [ "$RESUME_FROM_TARGET" = "1" ] && [ "$SSD_TARGET_WEIGHTS" != "$TRAIN_WEIGHTS" ]; then
    rsync -rlptD \
        --exclude 'logs/' \
        --exclude 'wandb/' \
        --exclude 'events.out.tfevents*' \
        --exclude 'progress.txt' \
        "$TRAIN_WEIGHTS/" "$SSD_TARGET_WEIGHTS/"
fi

write_continual_metadata
source "$MODEL_CONFIG_SCRIPT"

INFRA_ARGS=(
    --transformer-impl "$TRANSFORMER_IMPL"
    --pipeline-model-parallel-size "$PIPELINE_MODEL_PARALLEL_SIZE"
    --expert-model-parallel-size "$EXPERT_MODEL_PARALLEL_SIZE"
    --moe-token-dispatcher-type alltoall
    --distributed-timeout-minutes 30
    --no-persist-layer-norm
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
    --log-interval "$LOG_INTERVAL"
    --log-throughput
    --log-progress
    --save "$SSD_TARGET_WEIGHTS"
    --save-interval "$SAVE_INTERVAL"
    --eval-interval "$EVAL_INTERVAL"
    --tensorboard-dir "$SSD_TARGET_WEIGHTS"
    --moe-freeze-existing-experts
    --moe-freeze-existing-router
)

if [ "$RESUME_FROM_TARGET" = "1" ]; then
    SAVE_ARGS+=(
        --load "$SSD_TARGET_WEIGHTS"
        --moe-resume-from-num-experts "$SOURCE_NUM_EXPERTS"
    )
else
    SAVE_ARGS+=(
        --load "$SSD_SOURCE_WEIGHTS"
        --no-load-optim
        --no-load-rng
        --finetune
        --moe-expand-from-num-experts "$SOURCE_NUM_EXPERTS"
    )
fi

if [ "$RUN_INITIAL_VALID_EVAL" = "1" ]; then
    SAVE_ARGS+=(--run-initial-valid-eval)
fi

if [ "$TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY" = "1" ]; then
    SAVE_ARGS+=(--moe-train-new-experts-and-router-only)
else
    SAVE_ARGS+=(
        --moe-old-model-kl-load "$SSD_SOURCE_WEIGHTS"
        --moe-old-model-kl-coeff "$OLD_MODEL_KL_COEFF"
        --moe-old-model-kl-temperature "$OLD_MODEL_KL_TEMPERATURE"
    )
fi

PROBE_ARGS=(
    --probe-name "$PROBE_NAME"
    --probe-eval-iters "$PROBE_EVAL_ITERS"
    --probe-eval-interval "$PROBE_EVAL_INTERVAL"
    --probe-step-offset "$PROBE_STEP_OFFSET"
    --probe-data-path $(build_data_path "$PROBE_DATASET")
)

if [ "$RUN_INITIAL_PROBE_EVAL" = "1" ]; then
    PROBE_ARGS+=(--run-initial-probe-eval)
fi

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
        --wandb-run-id "$WANDB_RUN_ID"
        --wandb-resume "$WANDB_RESUME"
    )
fi

if [ "$LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND" = "1" ] && [ -n "$WANDB_PROJECT" ]; then
    echo "logging source-model probe baseline at step $PROBE_STEP_OFFSET before expert expansion"
    "$PYTHON_BIN" analysis/log_source_probe_baseline_to_wandb.py \
        --source-run-dir "$SOURCE_WEIGHTS_DIR" \
        --project "$WANDB_PROJECT" \
        --run-name "$WANDB_EXP_NAME" \
        --run-id "$WANDB_RUN_ID" \
        --resume "$WANDB_RESUME" \
        --save-dir "$WANDB_SAVE_DIR" \
        --step "$PROBE_STEP_OFFSET" \
        --primary-source-candidates "$SOURCE_SECONDARY_PROBE_CANDIDATES" \
        --primary-target-name "$PROBE_NAME" \
        --secondary-source-candidates "$SOURCE_PRIMARY_PROBE_CANDIDATES" \
        --secondary-target-name "$SECONDARY_PROBE_NAME"
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
if [ "$SSD_TARGET_WEIGHTS" != "$TRAIN_WEIGHTS" ]; then
    rsync -rlptD "$SSD_TARGET_WEIGHTS/" "$TRAIN_WEIGHTS/"
fi
SYNC_DONE=1

echo "a100 bf16 continual training complete. checkpoint at: $TRAIN_WEIGHTS"
