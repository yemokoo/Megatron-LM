#!/bin/bash
set -euo pipefail

# Joint (mixed) pretraining over wiki + code + conversation, randomly blended.
#
# Upper-bound / oracle baseline for the continual-learning experiments: all three
# datasets are shuffled together and trained in a single stage (no forgetting),
# so it measures the ceiling a given architecture can reach on all three tasks.
#
# Blend is 1:1:1 (each dataset shard gets weight 1.0 in --data-path), matching the
# ratio used elsewhere. Architecture is chosen via MODEL_CONFIG_SCRIPT (+ MoE env
# vars), so the same script serves both the dense(active) and fixed24 models.
# All three probes (wiki/code/conversation) are logged.

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
    # 1:1:1 blend (conversation ended up at ~95% instead of ~33%).
    bin_paths = sorted(Path(dataset_dir).glob('*.bin'))
    per_shard_weight = 1.0 / len(bin_paths)
    for bin_path in bin_paths:
        parts.extend([str(per_shard_weight), str(bin_path.with_suffix(''))])
print(' '.join(parts))
PY
}

export RUN_ID="${RUN_ID:-mixed-3way-pretrain-local-bf16-$(date -u +%Y%m%d-%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29560}"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"

export SSD_MOUNT="${LOCAL_SSD_ROOT}/${RUN_ID}"
export SSD_WIKI_TRAIN="${SSD_MOUNT}/dataset/wiki_train"
export SSD_CODE_TRAIN="${SSD_MOUNT}/dataset/code_train"
export SSD_CONV_TRAIN="${SSD_MOUNT}/dataset/conversation_train"
export SSD_WEIGHTS="${SSD_MOUNT}/weights"

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

# Joint upper-bound is a single stage matched to the sequential total (3 x 1800).
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

export TRAIN_DATASET_WIKI="${TRAIN_DATASET_WIKI:-$PROJECT_ROOT/data/wiki/train}"
export TRAIN_DATASET_CODE="${TRAIN_DATASET_CODE:-$PROJECT_ROOT/data/code/train}"
export TRAIN_DATASET_CONV="${TRAIN_DATASET_CONV:-$PROJECT_ROOT/data/conversation/train}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$LOCAL_WEIGHTS/a100/mha/mixed-3way/$RUN_ID}"
export LOG_DIR="${LOG_DIR:-$TRAIN_WEIGHTS/logs}"
export RUN_LOG="${RUN_LOG:-$LOG_DIR/mixed_3way_pretrain.log}"
export GPU_LOG="${GPU_LOG:-$LOG_DIR/gpu_usage.csv}"
export RUN_METADATA="${RUN_METADATA:-$LOG_DIR/run_metadata.json}"
export DATASET_NAME="${DATASET_NAME:-wiki_code_conversation_mixed}"
export DATASET_SOURCE="${DATASET_SOURCE:-Wikipedia + Code + Conversation joint 1:1:1}"

export PROBE_DATASET="${PROBE_DATASET:-$PROJECT_ROOT/data/wiki/test}"
export PROBE_NAME="${PROBE_NAME:-wiki_probe}"
export PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-25}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET:-$PROJECT_ROOT/data/code/test}"
export SECONDARY_PROBE_NAME="${SECONDARY_PROBE_NAME:-code_probe}"
export SECONDARY_PROBE_EVAL_ITERS="${SECONDARY_PROBE_EVAL_ITERS:-25}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_STEP_OFFSET="${SECONDARY_PROBE_STEP_OFFSET:-0}"
export TERTIARY_PROBE_DATASET="${TERTIARY_PROBE_DATASET:-$PROJECT_ROOT/data/conversation/test}"
export TERTIARY_PROBE_NAME="${TERTIARY_PROBE_NAME:-conversation_probe}"
export TERTIARY_PROBE_EVAL_ITERS="${TERTIARY_PROBE_EVAL_ITERS:-25}"
export TERTIARY_PROBE_EVAL_INTERVAL="${TERTIARY_PROBE_EVAL_INTERVAL:-50}"
export TERTIARY_PROBE_STEP_OFFSET="${TERTIARY_PROBE_STEP_OFFSET:-0}"

export WANDB_STEP_OFFSET="${WANDB_STEP_OFFSET:-0}"
export WANDB_PROJECT="${WANDB_PROJECT:-}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-$RUN_ID}"
export WANDB_SAVE_DIR="${WANDB_SAVE_DIR:-$TRAIN_WEIGHTS/wandb}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

mkdir -p "$SSD_WIKI_TRAIN" "$SSD_CODE_TRAIN" "$SSD_CONV_TRAIN" "$SSD_WEIGHTS" "$TRAIN_WEIGHTS" "$LOG_DIR"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "mixed 3way run log: $RUN_LOG"
echo "mixed 3way metadata: $RUN_METADATA"
rsync -rlptD --info=progress2 "$TRAIN_DATASET_WIKI/" "$SSD_WIKI_TRAIN/"
rsync -rlptD --info=progress2 "$TRAIN_DATASET_CODE/" "$SSD_CODE_TRAIN/"
rsync -rlptD --info=progress2 "$TRAIN_DATASET_CONV/" "$SSD_CONV_TRAIN/"

# --load always points at $SSD_WEIGHTS (below), which is ephemeral /tmp storage. If
# a prior attempt already made progress and it was flushed to persistent storage
# (e.g. after an OOM crash) but the SSD copy was since wiped, restore it here so
# training resumes from the latest checkpoint instead of silently restarting from
# scratch.
if [ -f "$TRAIN_WEIGHTS/latest_checkpointed_iteration.txt" ] && [ ! -f "$SSD_WEIGHTS/latest_checkpointed_iteration.txt" ]; then
    echo "mixed 3way: restoring own in-progress checkpoint from $TRAIN_WEIGHTS to $SSD_WEIGHTS for resume"
    rsync -rlptD \
        --exclude 'logs/' \
        --exclude 'wandb/' \
        --exclude 'events.out.tfevents*' \
        --exclude 'progress.txt' \
        "$TRAIN_WEIGHTS/" "$SSD_WEIGHTS/"
fi

"$PYTHON_BIN" - <<'PY'
import json, os
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

wiki = summarize(os.environ['SSD_WIKI_TRAIN'])
code = summarize(os.environ['SSD_CODE_TRAIN'])
conv = summarize(os.environ['SSD_CONV_TRAIN'])
metadata = {
    'stage': 'mixed_3way_pretrain',
    'run_id': os.environ['RUN_ID'],
    'dataset_name': os.environ['DATASET_NAME'],
    'dataset_source': os.environ['DATASET_SOURCE'],
    'blend': '1:1:1 wiki:code:conversation',
    'wiki_train': wiki, 'code_train': code, 'conversation_train': conv,
    'combined_tokens': wiki['tokens'] + code['tokens'] + conv['tokens'],
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
    --data-path $(build_data_path "$SSD_WIKI_TRAIN" "$SSD_CODE_TRAIN" "$SSD_CONV_TRAIN")
    --split "$DATASET_SPLIT"
)

SAVE_ARGS=(
    --log-interval "$LOG_INTERVAL"
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

(
    while kill -0 "$TORCHRUN_PID" 2>/dev/null; do
        rsync -rlptD "$SSD_WEIGHTS/" "$TRAIN_WEIGHTS/"
        sleep 15m
    done
) &

set +e
wait "$TORCHRUN_PID"
TORCHRUN_EXIT=$?
set -e
kill "$GPU_LOG_PID" 2>/dev/null || true
# Always flush to persistent storage, even on crash (e.g. OOM), so a rerun can
# resume from the latest checkpoint instead of losing progress since the last
# periodic (15-minute) rsync.
rsync -rlptD "$SSD_WEIGHTS/" "$TRAIN_WEIGHTS/"

if [ "$TORCHRUN_EXIT" -ne 0 ]; then
    echo "[FAIL] mixed 3way pretrain exited with code $TORCHRUN_EXIT; checkpoint flushed to $TRAIN_WEIGHTS for resume"
    exit "$TORCHRUN_EXIT"
fi

echo "mixed 3way pretrain complete. Checkpoint at: $TRAIN_WEIGHTS"
