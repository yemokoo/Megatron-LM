#!/bin/bash
set -euo pipefail

# Generalized N-way mixed (blended) trainer for the cumulative continual-learning
# upper bound (exp 6 / exp 7).
#
# It trains on a *variable* set of datasets blended 1:1(:1) in a single stage, and
# can optionally continue from a previous-stage checkpoint (STAGE_SOURCE_WEIGHTS),
# which is what turns a sequence of these into a cumulative-replay chain:
#     stage 1: wiki                    (from scratch, 1800 iters)
#     stage 2: wiki + code             (from stage-1 ckpt, 3600 iters)
#     stage 3: wiki + code + conv      (from stage-2 ckpt, 5400 iters)
#
# This is the CL-internal upper bound: every stage re-mixes *all* data seen so far
# (full replay), so there is no forgetting from replay's point of view, while still
# respecting the "1800 steps per dataset" budget. Contrast with:
#   - sequential (exp 1/2): one dataset per stage, KD to fight forgetting  -> lower bound
#   - joint mixed (exp 4/5): all 3 datasets from step 0                    -> non-CL ceiling
#
# Architecture is chosen via MODEL_CONFIG_SCRIPT (+ MoE env vars), so the same
# script serves both the dense(active) (exp 6) and fixed24 (exp 7) models. All three
# probes (wiki/code/conversation) are always logged regardless of what is trained.
#
# No KD here: replay is the forgetting-mitigation mechanism, so keeping it pure
# (kl0) is what makes "replay upper bound" a clean reading. (There is no teacher
# wiring in this trainer at all.)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"
# Puts Megatron-LM on PYTHONPATH; without it "import megatron" fails for the
# metadata-summary heredocs below (they run before `cd Megatron-LM`).
source "$PROJECT_ROOT/scripts/miscellaneous/activate_kt_env.sh"

resolve_python() {
    if command -v python >/dev/null 2>&1; then command -v python; return; fi
    command -v python3
}
export PYTHON_BIN="${PYTHON_BIN:-$(resolve_python)}"

build_data_path() {
    "$PYTHON_BIN" - "$@" <<'PY'
import sys
from pathlib import Path
parts = []
for dataset_dir in sys.argv[1:]:
    # Each *dataset* (directory) gets combined weight 1.0, split evenly across its
    # own shards -- NOT one weight of 1.0 per shard. A directory with N shards
    # (e.g. conversation/train has 42 vs wiki/code's 1) would otherwise get ~N x
    # the sampling weight of a single-shard dataset, badly skewing the intended
    # 1:1(:1) blend (conversation ended up at ~95% instead of ~33%/50%).
    bin_paths = sorted(Path(dataset_dir).glob('*.bin'))
    per_shard_weight = 1.0 / len(bin_paths)
    for bin_path in bin_paths:
        parts.extend([str(per_shard_weight), str(bin_path.with_suffix(''))])
print(' '.join(parts))
PY
}

export RUN_ID="${RUN_ID:-mixed-nway-pretrain-local-bf16}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29570}"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
export STAGE_INPUTS_TO_SCRATCH="${STAGE_INPUTS_TO_SCRATCH:-0}"
export DIRECT_LOCAL_SAVE="${DIRECT_LOCAL_SAVE:-1}"
export STORAGE_PLAN_ONLY="${STORAGE_PLAN_ONLY:-0}"

export SSD_MOUNT="${LOCAL_SSD_ROOT}/${RUN_ID}"
export SSD_DATA_ROOT="${SSD_MOUNT}/dataset"
export SSD_WEIGHTS="${SSD_MOUNT}/weights"
export SSD_SOURCE_WEIGHTS="${SSD_MOUNT}/source_weights"

# --- which datasets to blend this stage (space-separated train dirs) ---
# Default = full 3-way blend (equivalent to the old 3way trainer).
export MIXED_TRAIN_DATASETS="${MIXED_TRAIN_DATASETS:-$PROJECT_ROOT/data/wiki/train $PROJECT_ROOT/data/code/train $PROJECT_ROOT/data/conversation/train}"

# --- optional previous-stage checkpoint to continue from (empty = from scratch) ---
export STAGE_SOURCE_WEIGHTS="${STAGE_SOURCE_WEIGHTS:-}"

# --- architecture (defaults = FFN-only MoE; wrappers override experts/topk/ffn) ---
export NUM_LAYERS="${NUM_LAYERS:-9}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-5472}"
export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-configs/model/flame-moe-ffn-only-no-shared.sh}"
export NUM_EXPERTS="${NUM_EXPERTS:-24}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_LAYER_FREQ="${MOE_LAYER_FREQ:-[0,1,1,1,1,1,1,1,1]}"
export MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}"
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.01}"
export MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.001}"
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"

export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-48}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export SEQ_LENGTH="${SEQ_LENGTH:-512}"
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=1
export TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-local}"
export PRECISION="bf16"
export SEED="${SEED:-1234}"

# Per-stage iteration budget (wrappers pass 1800 / 3600 / 5400). Each stage runs a
# fresh WSD schedule keyed off its own TRAIN_ITERS.
export TRAIN_ITERS="${TRAIN_ITERS:-5400}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-600}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export GPU_LOG_INTERVAL_SECONDS="${GPU_LOG_INTERVAL_SECONDS:-30}"
export LR="${LR:-3e-4}"
export MIN_LR="${MIN_LR:-3e-5}"
export LR_DECAY_STYLE="${LR_DECAY_STYLE:-WSD}"
export LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.01}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-$TRAIN_ITERS}"
export LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-$((TRAIN_ITERS / 10))}"
export DATASET_SPLIT="${DATASET_SPLIT:-100,0,0}"

export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$LOCAL_WEIGHTS/a100/mha/mixed-nway/$RUN_ID}"
export LOG_DIR="${LOG_DIR:-$TRAIN_WEIGHTS/logs}"
export RUN_LOG="${RUN_LOG:-$LOG_DIR/mixed_nway_pretrain.log}"
export GPU_LOG="${GPU_LOG:-$LOG_DIR/gpu_usage.csv}"
export RUN_METADATA="${RUN_METADATA:-$LOG_DIR/run_metadata.json}"
export DATASET_NAME="${DATASET_NAME:-mixed_nway}"
export DATASET_SOURCE="${DATASET_SOURCE:-cumulative mixed blend}"

# Probes: always all three tasks (wiki primary, code secondary, conversation tertiary).
export PROBE_DATASET="${PROBE_DATASET:-$PROJECT_ROOT/data/wiki/test}"
export PROBE_NAME="${PROBE_NAME:-wiki_probe}"
export PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-25}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}"
export PROBE_STEP_OFFSET="${PROBE_STEP_OFFSET:-0}"
export SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET:-$PROJECT_ROOT/data/code/test}"
export SECONDARY_PROBE_NAME="${SECONDARY_PROBE_NAME:-code_probe}"
export SECONDARY_PROBE_EVAL_ITERS="${SECONDARY_PROBE_EVAL_ITERS:-25}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_STEP_OFFSET="${SECONDARY_PROBE_STEP_OFFSET:-$PROBE_STEP_OFFSET}"
export TERTIARY_PROBE_DATASET="${TERTIARY_PROBE_DATASET:-$PROJECT_ROOT/data/conversation/test}"
export TERTIARY_PROBE_NAME="${TERTIARY_PROBE_NAME:-conversation_probe}"
export TERTIARY_PROBE_EVAL_ITERS="${TERTIARY_PROBE_EVAL_ITERS:-25}"
export TERTIARY_PROBE_EVAL_INTERVAL="${TERTIARY_PROBE_EVAL_INTERVAL:-50}"
export TERTIARY_PROBE_STEP_OFFSET="${TERTIARY_PROBE_STEP_OFFSET:-$PROBE_STEP_OFFSET}"

export WANDB_STEP_OFFSET="${WANDB_STEP_OFFSET:-$PROBE_STEP_OFFSET}"
export WANDB_PROJECT="${WANDB_PROJECT:-}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-$RUN_ID}"
export WANDB_SAVE_DIR="${WANDB_SAVE_DIR:-$TRAIN_WEIGHTS/wandb}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

case "$STAGE_INPUTS_TO_SCRATCH" in 0|1) ;; *) echo "ERROR: STAGE_INPUTS_TO_SCRATCH must be 0 or 1" >&2; exit 1 ;; esac
case "$DIRECT_LOCAL_SAVE" in 0|1) ;; *) echo "ERROR: DIRECT_LOCAL_SAVE must be 0 or 1" >&2; exit 1 ;; esac
if [ "$STAGE_INPUTS_TO_SCRATCH" = "0" ] || [ "$DIRECT_LOCAL_SAVE" = "1" ]; then
    export SSD_WEIGHTS="$TRAIN_WEIGHTS"
fi
if [ "$STAGE_INPUTS_TO_SCRATCH" = "0" ]; then
    export SSD_SOURCE_WEIGHTS="$STAGE_SOURCE_WEIGHTS"
else
    mkdir -p "$SSD_DATA_ROOT"
fi
mkdir -p "$SSD_WEIGHTS" "$TRAIN_WEIGHTS" "$LOG_DIR"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "mixed nway run log: $RUN_LOG"
echo "mixed nway metadata: $RUN_METADATA"
echo "mixed nway datasets: $MIXED_TRAIN_DATASETS"
echo "mixed nway source ckpt: ${STAGE_SOURCE_WEIGHTS:-<from scratch>}"

# --- resolve the variable-length list of train datasets ---
SSD_TRAIN_DIRS=()
i=0
for src in $MIXED_TRAIN_DATASETS; do
    [[ -d "$src" ]] || { echo "ERROR: mixed training dataset missing: $src" >&2; exit 1; }
    if [ "$STAGE_INPUTS_TO_SCRATCH" = "1" ]; then
        dst="$SSD_DATA_ROOT/ds${i}"
        mkdir -p "$dst"
        rsync -rlptD --info=progress2 "$src/" "$dst/"
        SSD_TRAIN_DIRS+=("$dst")
    else
        SSD_TRAIN_DIRS+=("$src")
    fi
    i=$((i + 1))
done

# --- own-progress resume: if THIS stage already saved a checkpoint in persistent
# storage (e.g. a prior attempt OOM'd mid-stage), restore it and do a TRUE resume
# (keep optimizer/rng/iteration) instead of restarting the stage from the
# previous-stage source checkpoint at iteration 0.
RESUME_OWN_PROGRESS=0
if [ -f "$TRAIN_WEIGHTS/latest_checkpointed_iteration.txt" ]; then
    echo "mixed nway: found own in-progress checkpoint at $TRAIN_WEIGHTS, resuming from it (not restarting this stage from source)"
    if [ "$SSD_WEIGHTS" != "$TRAIN_WEIGHTS" ]; then
        mkdir -p "$SSD_WEIGHTS"
        rsync -rlptD \
            --exclude 'logs/' \
            --exclude 'wandb/' \
            --exclude 'events.out.tfevents*' \
            --exclude 'progress.txt' \
            "$TRAIN_WEIGHTS/" "$SSD_WEIGHTS/"
    fi
    RESUME_OWN_PROGRESS=1
fi

# --- optional: stage the previous-stage checkpoint to continue from ---
LOAD_ARGS=()
if [ "$RESUME_OWN_PROGRESS" = "1" ]; then
    # true resume: keep optimizer/rng state and the in-progress iteration count.
    LOAD_ARGS=(--load "$SSD_WEIGHTS")
elif [ -n "$STAGE_SOURCE_WEIGHTS" ]; then
    echo "continuing from source checkpoint: $STAGE_SOURCE_WEIGHTS"
    if [ "$STAGE_INPUTS_TO_SCRATCH" = "1" ]; then
        mkdir -p "$SSD_SOURCE_WEIGHTS"
        rsync -rlptD \
            --exclude 'logs/' \
            --exclude 'wandb/' \
            --exclude 'events.out.tfevents*' \
            --exclude 'progress.txt' \
            "$STAGE_SOURCE_WEIGHTS/" "$SSD_SOURCE_WEIGHTS/"
    fi
    # finetune: load model weights only, reset iteration to 0, fresh WSD + optimizer.
    LOAD_ARGS=(
        --load "$SSD_SOURCE_WEIGHTS"
        --finetune
        --no-load-optim
        --no-load-rng
    )
else
    # from scratch, no prior progress.
    LOAD_ARGS=(--load "$SSD_WEIGHTS")
fi

if [ "$STORAGE_PLAN_ONLY" = "1" ]; then
    printf 'STAGE_INPUTS_TO_SCRATCH=%s\nDATASETS=%s\nSOURCE=%s\nSAVE=%s\n' \
        "$STAGE_INPUTS_TO_SCRATCH" "${SSD_TRAIN_DIRS[*]}" "${SSD_SOURCE_WEIGHTS:-}" "$SSD_WEIGHTS"
    exit 0
fi

"$PYTHON_BIN" - "${SSD_TRAIN_DIRS[@]}" <<'PY'
import json, os, sys
from pathlib import Path
from megatron.core.datasets import indexed_dataset

def summarize(dataset_dir):
    total_tokens = 0
    total_documents = 0
    for idx_path in sorted(Path(dataset_dir).glob('*.idx')):
        ds = indexed_dataset.IndexedDataset(str(idx_path.with_suffix('')), multimodal=False, mmap=True)
        total_tokens += int(ds.sequence_lengths.sum())
        total_documents += int(ds.document_indices.shape[0] - 1)
    return {'path': str(dataset_dir), 'tokens': total_tokens, 'documents': total_documents}

datasets = [summarize(d) for d in sys.argv[1:]]
metadata = {
    'stage': 'mixed_nway_pretrain',
    'run_id': os.environ['RUN_ID'],
    'dataset_name': os.environ['DATASET_NAME'],
    'dataset_source': os.environ['DATASET_SOURCE'],
    'blend': '1:1(:1) over datasets listed below',
    'mixed_train_datasets': os.environ['MIXED_TRAIN_DATASETS'].split(),
    'staged_datasets': datasets,
    'combined_tokens': sum(d['tokens'] for d in datasets),
    'stage_source_weights': os.environ.get('STAGE_SOURCE_WEIGHTS', ''),
    'model_config_script': os.environ['MODEL_CONFIG_SCRIPT'],
    'num_experts': int(os.environ['NUM_EXPERTS']),
    'moe_ffn_hidden_size': int(os.environ['MOE_FFN_HIDDEN_SIZE']),
    'moe_router_topk': int(os.environ['MOE_ROUTER_TOPK']),
    'train_iters': int(os.environ['TRAIN_ITERS']),
    'micro_batch_size': int(os.environ['MICRO_BATCH_SIZE']),
    'global_batch_size': int(os.environ['GLOBAL_BATCH_SIZE']),
    'num_layers': int(os.environ['NUM_LAYERS']),
    'hidden_size': int(os.environ['HIDDEN_SIZE']),
    'ffn_hidden_size': int(os.environ['FFN_HIDDEN_SIZE']),
    'probe_step_offset': int(os.environ['PROBE_STEP_OFFSET']),
    'precision': os.environ['PRECISION'],
}
with open(os.environ['RUN_METADATA'], 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
print('metadata written; combined tokens =', metadata['combined_tokens'])
PY

source "$MODEL_CONFIG_SCRIPT"

INFRA_ARGS=(
    --transformer-impl "$TRANSFORMER_IMPL"
    --pipeline-model-parallel-size "$PIPELINE_MODEL_PARALLEL_SIZE"
    --expert-model-parallel-size "$EXPERT_MODEL_PARALLEL_SIZE"
    --distributed-timeout-minutes 30
    --no-persist-layer-norm
    --seed "$SEED"
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
    --seq-length "$SEQ_LENGTH"
    --data-path $(build_data_path "${SSD_TRAIN_DIRS[@]}")
    --split "$DATASET_SPLIT"
)

SAVE_ARGS=(
    --log-interval "$LOG_INTERVAL"
    --log-throughput
    --log-progress
    --save "$SSD_WEIGHTS"
    --save-interval "$SAVE_INTERVAL"
    "${LOAD_ARGS[@]}"
    --eval-interval "$EVAL_INTERVAL"
    --tensorboard-dir "$SSD_WEIGHTS"
)

PROBE_ARGS=(
    --probe-name "$PROBE_NAME"
    --probe-eval-iters "$PROBE_EVAL_ITERS"
    --probe-eval-interval "$PROBE_EVAL_INTERVAL"
    --probe-step-offset "$PROBE_STEP_OFFSET"
    --probe-data-path $(build_data_path "$PROBE_DATASET")
    --secondary-probe-name "$SECONDARY_PROBE_NAME"
    --secondary-probe-eval-iters "$SECONDARY_PROBE_EVAL_ITERS"
    --secondary-probe-eval-interval "$SECONDARY_PROBE_EVAL_INTERVAL"
    --secondary-probe-step-offset "$SECONDARY_PROBE_STEP_OFFSET"
    --secondary-probe-data-path $(build_data_path "$SECONDARY_PROBE_DATASET")
    --tertiary-probe-name "$TERTIARY_PROBE_NAME"
    --tertiary-probe-eval-iters "$TERTIARY_PROBE_EVAL_ITERS"
    --tertiary-probe-eval-interval "$TERTIARY_PROBE_EVAL_INTERVAL"
    --tertiary-probe-step-offset "$TERTIARY_PROBE_STEP_OFFSET"
    --tertiary-probe-data-path $(build_data_path "$TERTIARY_PROBE_DATASET")
)

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
    "${DATA_ARGS[@]}" "${SAVE_ARGS[@]}" "${PROBE_ARGS[@]}" "${WANDB_ARGS[@]}" &
TORCHRUN_PID=$!

if [ "$SSD_WEIGHTS" != "$TRAIN_WEIGHTS" ]; then
    (
        while kill -0 "$TORCHRUN_PID" 2>/dev/null; do
            rsync -rlptD "$SSD_WEIGHTS/" "$TRAIN_WEIGHTS/"
            sleep 15m
        done
    ) &
fi

set +e
wait "$TORCHRUN_PID"
TORCHRUN_EXIT=$?
set -e
kill "$GPU_LOG_PID" 2>/dev/null || true
# Always flush to persistent storage, even on crash (e.g. OOM), so a rerun can
# resume from the latest checkpoint instead of losing progress since the last
# periodic (15-minute) rsync.
if [ "$SSD_WEIGHTS" != "$TRAIN_WEIGHTS" ]; then
    rsync -rlptD "$SSD_WEIGHTS/" "$TRAIN_WEIGHTS/"
fi

if [ "$TORCHRUN_EXIT" -ne 0 ]; then
    echo "[FAIL] mixed nway pretrain exited with code $TORCHRUN_EXIT; checkpoint flushed to $TRAIN_WEIGHTS for resume"
    exit "$TORCHRUN_EXIT"
fi

echo "mixed nway pretrain complete. Checkpoint at: $TRAIN_WEIGHTS"
