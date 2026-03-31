#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

resolve_python() {
    if [ -x "$PROJECT_ROOT/.conda/envs/flame3090/bin/python" ]; then
        echo "$PROJECT_ROOT/.conda/envs/flame3090/bin/python"
        return
    fi
    if command -v python >/dev/null 2>&1; then
        command -v python
        return
    fi
    command -v python3
}

export PYTHON_BIN="${PYTHON_BIN:-$(resolve_python)}"

build_data_path() {
    "$PYTHON_BIN" - "$@" <<'PY'
import sys
from pathlib import Path
parts = []
for dataset_dir in sys.argv[1:]:
    for bin_path in sorted(Path(dataset_dir).glob('*.bin')):
        parts.extend(['1.0', str(bin_path.with_suffix(''))])
print(' '.join(parts))
PY
}

export RUN_ID="${RUN_ID:-wiki-shared-router-hybrid-pretrain-local-bf16-$(date -u +%Y%m%d-%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29573}"

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
export NUM_EXPERTS="${NUM_EXPERTS:-4}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-2}"
export MOE_LAYER_FREQ="${MOE_LAYER_FREQ:-[0,1,1,1,1,1,1,1,1]}"
export ATTN_LORA_RANK="${ATTN_LORA_RANK:-16}"
export ATTN_LORA_ALPHA="${ATTN_LORA_ALPHA:-16}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=1
export TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-local}"
export PRECISION="bf16"

export TRAIN_ITERS="${TRAIN_ITERS:-}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-300}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export LR="${LR:-3e-4}"
export MIN_LR="${MIN_LR:-3e-5}"
export LR_DECAY_STYLE="${LR_DECAY_STYLE:-WSD}"
export LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.01}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-}"
export LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-configs/model/flame-shared-router-hybrid-experts.sh}"
export DATASET_SPLIT="${DATASET_SPLIT:-100,0,0}"

export TRAIN_DATASET="${TRAIN_DATASET:-$LOCAL_DATASET/wikipedia-full/tokenized/EleutherAI/pythia-12b-step1800-train-exact}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$LOCAL_WEIGHTS/a100/wiki-shared-router-hybrid-pretrain-local/$RUN_ID}"
export LOG_DIR="${LOG_DIR:-$TRAIN_WEIGHTS/logs}"
export RUN_METADATA="${RUN_METADATA:-$LOG_DIR/run_metadata.json}"
export DATASET_NAME="${DATASET_NAME:-wiki_exact}"
export DATASET_SOURCE="${DATASET_SOURCE:-Wikipedia exact train}"
export PROBE_DATASET="${PROBE_DATASET:-$LOCAL_DATASET/wikipedia-full/tokenized/EleutherAI/pythia-12b-step1800-test-fullremainder-exact}"
export PROBE_NAME="${PROBE_NAME:-wiki_probe}"
export PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-25}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-100}"
export SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET:-$LOCAL_DATASET/python-code-full/tokenized/EleutherAI/pythia-12b-step1800-test-matchwiki-exact}"
export SECONDARY_PROBE_NAME="${SECONDARY_PROBE_NAME:-code_probe}"
export SECONDARY_PROBE_EVAL_ITERS="${SECONDARY_PROBE_EVAL_ITERS:-25}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-100}"
export SECONDARY_PROBE_STEP_OFFSET="${SECONDARY_PROBE_STEP_OFFSET:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

mkdir -p "$SSD_TRAIN_DATASET" "$SSD_WEIGHTS" "$TRAIN_WEIGHTS" "$LOG_DIR"
rsync -rlptD --info=progress2 "$TRAIN_DATASET/" "$SSD_TRAIN_DATASET/"

if [ -z "${TRAIN_ITERS:-}" ]; then
    export TRAIN_ITERS="$("$PYTHON_BIN" - <<'PY'
from pathlib import Path
from megatron.core.datasets import indexed_dataset
import os

dataset_dir = Path(os.environ['SSD_TRAIN_DATASET'])
total_tokens = 0
for idx_path in sorted(dataset_dir.glob('*.idx')):
    ds = indexed_dataset.IndexedDataset(str(idx_path.with_suffix('')), multimodal=False, mmap=True)
    total_tokens += int(ds.sequence_lengths.sum())
seq_length = int(os.environ.get('SEQ_LENGTH', '512'))
global_batch_size = int(os.environ.get('GLOBAL_BATCH_SIZE', '16'))
print(max(1, (total_tokens // seq_length) // global_batch_size))
PY
)"
fi

export SAVE_INTERVAL="${SAVE_INTERVAL:-$TRAIN_ITERS}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-$TRAIN_ITERS}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-$TRAIN_ITERS}"
export LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-$((TRAIN_ITERS / 10))}"

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
from megatron.core.datasets import indexed_dataset

dataset_dir = Path(os.environ['SSD_TRAIN_DATASET'])
total_tokens = 0
total_documents = 0
shards = []
for idx_path in sorted(dataset_dir.glob('*.idx')):
    prefix = idx_path.with_suffix('')
    ds = indexed_dataset.IndexedDataset(str(prefix), multimodal=False, mmap=True)
    shard_tokens = int(ds.sequence_lengths.sum())
    shard_docs = int(ds.document_indices.shape[0] - 1)
    total_tokens += shard_tokens
    total_documents += shard_docs
    shards.append({'prefix': prefix.name, 'documents': shard_docs, 'tokens': shard_tokens})

metadata = {
    'stage': 'wiki_shared_router_hybrid_pretrain',
    'run_id': os.environ['RUN_ID'],
    'dataset_name': os.environ['DATASET_NAME'],
    'dataset_source': os.environ['DATASET_SOURCE'],
    'train_dataset': {'path': str(dataset_dir), 'tokens': total_tokens, 'documents': total_documents, 'shards': shards},
    'train_iters': int(os.environ['TRAIN_ITERS']),
    'micro_batch_size': int(os.environ['MICRO_BATCH_SIZE']),
    'global_batch_size': int(os.environ['GLOBAL_BATCH_SIZE']),
    'num_layers': int(os.environ['NUM_LAYERS']),
    'hidden_size': int(os.environ['HIDDEN_SIZE']),
    'ffn_hidden_size': int(os.environ['FFN_HIDDEN_SIZE']),
    'moe_ffn_hidden_size': int(os.environ['MOE_FFN_HIDDEN_SIZE']),
    'num_experts': int(os.environ['NUM_EXPERTS']),
    'moe_router_topk': int(os.environ['MOE_ROUTER_TOPK']),
    'attn_lora_rank': int(os.environ['ATTN_LORA_RANK']),
    'attn_lora_alpha': float(os.environ['ATTN_LORA_ALPHA']),
    'shared_router_hybrid': True,
}
with open(os.environ['RUN_METADATA'], 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
PY

source "$MODEL_CONFIG_SCRIPT"

torchrun \
    --nproc_per_node "$NPROC_PER_NODE" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    Megatron-LM/pretrain_gpt.py \
    "${MODEL_ARGS[@]}" \
    --transformer-impl "$TRANSFORMER_IMPL" \
    --pipeline-model-parallel-size "$PIPELINE_MODEL_PARALLEL_SIZE" \
    --expert-model-parallel-size "$EXPERT_MODEL_PARALLEL_SIZE" \
    --distributed-timeout-minutes 30 \
    --no-persist-layer-norm \
    --bf16 \
    --micro-batch-size "$MICRO_BATCH_SIZE" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --lr "$LR" \
    --min-lr "$MIN_LR" \
    --lr-decay-style "$LR_DECAY_STYLE" \
    --lr-decay-iters "$LR_DECAY_ITERS" \
    --lr-warmup-fraction "$LR_WARMUP_FRACTION" \
    --lr-wsd-decay-iters "$LR_WSD_DECAY_ITERS" \
    --train-iters "$TRAIN_ITERS" \
    --seq-length "${SEQ_LENGTH:-512}" \
    --data-path $(build_data_path "$SSD_TRAIN_DATASET") \
    --split "$DATASET_SPLIT" \
    --log-interval "$LOG_INTERVAL" \
    --log-throughput \
    --log-progress \
    --save "$SSD_WEIGHTS" \
    --save-interval "$SAVE_INTERVAL" \
    --load "$SSD_WEIGHTS" \
    --eval-interval "$EVAL_INTERVAL" \
    --tensorboard-dir "$SSD_WEIGHTS" \
    --probe-name "$PROBE_NAME" \
    --probe-eval-iters "$PROBE_EVAL_ITERS" \
    --probe-eval-interval "$PROBE_EVAL_INTERVAL" \
    --probe-data-path $(build_data_path "$PROBE_DATASET") \
    --secondary-probe-name "$SECONDARY_PROBE_NAME" \
    --secondary-probe-eval-iters "$SECONDARY_PROBE_EVAL_ITERS" \
    --secondary-probe-eval-interval "$SECONDARY_PROBE_EVAL_INTERVAL" \
    --secondary-probe-step-offset "$SECONDARY_PROBE_STEP_OFFSET" \
    --secondary-probe-data-path $(build_data_path "$SECONDARY_PROBE_DATASET")
