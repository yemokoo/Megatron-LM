#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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

resolve_dense_base_dir() {
    "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

explicit = os.environ.get('DENSE_BASE_WEIGHTS_DIR', '').strip()
if explicit:
    print(explicit)
    raise SystemExit(0)

local_weights = Path(os.environ['LOCAL_WEIGHTS']) / 'mixed-dense-pretrain-local'
required_iters = int(os.environ.get('DENSE_BASE_REQUIRED_ITERS', '1'))
best = None
best_mtime = -1.0
for candidate in local_weights.iterdir() if local_weights.exists() else []:
    tracker = candidate / 'latest_checkpointed_iteration.txt'
    metadata = candidate / 'logs' / 'run_metadata.json'
    if not candidate.is_dir() or not tracker.exists() or not metadata.exists():
        continue
    try:
        tracker_step = int(tracker.read_text(encoding='utf-8').strip())
        train_iters = int(json.loads(metadata.read_text(encoding='utf-8'))['train_iters'])
    except Exception:
        continue
    if tracker_step < required_iters or train_iters < required_iters:
        continue
    mtime = tracker.stat().st_mtime
    if mtime > best_mtime:
        best = candidate
        best_mtime = mtime
if best is None:
    raise SystemExit('ERROR: could not find a completed mixed dense pretrain run. Set DENSE_BASE_WEIGHTS_DIR explicitly.')
print(best)
PY
}

read_dense_train_iters() {
    "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
metadata = json.loads((Path(os.environ['DENSE_BASE_WEIGHTS_DIR']) / 'logs' / 'run_metadata.json').read_text(encoding='utf-8'))
print(int(metadata['train_iters']))
PY
}

export RUN_ID="${RUN_ID:-mixed-qv-lora-experts-local-bf16-$(date -u +%Y%m%d-%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29561}"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_DATASET="${LOCAL_DATASET:-$LOCAL_BASE/dataset}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"

export SSD_MOUNT="${LOCAL_SSD_ROOT}/${RUN_ID}"
export SSD_WIKI_TRAIN="${SSD_MOUNT}/dataset/wiki_train"
export SSD_CODE_TRAIN="${SSD_MOUNT}/dataset/code_train"
export SSD_WEIGHTS="${SSD_MOUNT}/weights"

export NUM_LAYERS="${NUM_LAYERS:-9}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-5472}"
export ATTN_LORA_NUM_EXPERTS="${ATTN_LORA_NUM_EXPERTS:-4}"
export ATTN_LORA_RANK="${ATTN_LORA_RANK:-16}"
export ATTN_LORA_TOPK="${ATTN_LORA_TOPK:-1}"
export ATTN_LORA_ALPHA="${ATTN_LORA_ALPHA:-16}"
export ATTN_LORA_ROUTER_DTYPE="${ATTN_LORA_ROUTER_DTYPE:-fp32}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=1
export TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-local}"
export PRECISION="bf16"

export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-300}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
export GPU_LOG_INTERVAL_SECONDS="${GPU_LOG_INTERVAL_SECONDS:-30}"
export LR="${LR:-3e-4}"
export MIN_LR="${MIN_LR:-3e-5}"
export LR_DECAY_STYLE="${LR_DECAY_STYLE:-WSD}"
export LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.01}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-$TRAIN_ITERS}"
export LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-$((TRAIN_ITERS / 10))}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-configs/model/flame-qv-lora-experts.sh}"
export DATASET_SPLIT="${DATASET_SPLIT:-100,0,0}"
export RESUME_ITERATION="${RESUME_ITERATION:-latest}"

export TRAIN_DATASET_WIKI="${TRAIN_DATASET_WIKI:-$LOCAL_DATASET/wikipedia-full/tokenized/EleutherAI/pythia-12b-step1800-train-exact}"
export TRAIN_DATASET_CODE="${TRAIN_DATASET_CODE:-$LOCAL_DATASET/python-code-full/tokenized/EleutherAI/pythia-12b-step1800-train-exact}"
export DENSE_BASE_WEIGHTS_DIR="${DENSE_BASE_WEIGHTS_DIR:-}"
export DENSE_BASE_REQUIRED_ITERS="${DENSE_BASE_REQUIRED_ITERS:-1}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$LOCAL_WEIGHTS/mixed-qv-lora-experts-local/$RUN_ID}"
export LOG_DIR="${LOG_DIR:-$TRAIN_WEIGHTS/logs}"
export RUN_LOG="${RUN_LOG:-$LOG_DIR/mixed_qv_lora_experts.log}"
export GPU_LOG="${GPU_LOG:-$LOG_DIR/gpu_usage.csv}"
export RUN_METADATA="${RUN_METADATA:-$LOG_DIR/run_metadata.json}"
export DATASET_NAME="${DATASET_NAME:-wiki_code_mixed_exact}"
export DATASET_SOURCE="${DATASET_SOURCE:-Wikipedia exact train + Code exact train}"
export PROBE_DATASET="${PROBE_DATASET:-$LOCAL_DATASET/wikipedia-full/tokenized/EleutherAI/pythia-12b-step1800-test-fullremainder-exact}"
export PROBE_NAME="${PROBE_NAME:-wiki_probe}"
export PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-25}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-40}"
export SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET:-$LOCAL_DATASET/python-code-full/tokenized/EleutherAI/pythia-12b-step1800-test-matchwiki-exact}"
export SECONDARY_PROBE_NAME="${SECONDARY_PROBE_NAME:-code_probe}"
export SECONDARY_PROBE_EVAL_ITERS="${SECONDARY_PROBE_EVAL_ITERS:-25}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-40}"
export PROBE_STEP_OFFSET="${PROBE_STEP_OFFSET:-}"
export SECONDARY_PROBE_STEP_OFFSET="${SECONDARY_PROBE_STEP_OFFSET:-}"
export WANDB_STEP_OFFSET="${WANDB_STEP_OFFSET:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-$RUN_ID}"
export WANDB_SAVE_DIR="${WANDB_SAVE_DIR:-$TRAIN_WEIGHTS/wandb}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

export DENSE_BASE_WEIGHTS_DIR="$(resolve_dense_base_dir)"
export PROBE_STEP_OFFSET="${PROBE_STEP_OFFSET:-$(read_dense_train_iters)}"
export SECONDARY_PROBE_STEP_OFFSET="${SECONDARY_PROBE_STEP_OFFSET:-$PROBE_STEP_OFFSET}"
export WANDB_STEP_OFFSET="${WANDB_STEP_OFFSET:-$PROBE_STEP_OFFSET}"

if [ "$RESUME_ITERATION" != "latest" ]; then
    checkpoint_dir="$DENSE_BASE_WEIGHTS_DIR/iter_$(printf '%07d' "$RESUME_ITERATION")"
    if [ ! -d "$checkpoint_dir" ]; then
        echo "ERROR: missing checkpoint directory $checkpoint_dir"
        exit 1
    fi
fi

mkdir -p "$SSD_WIKI_TRAIN" "$SSD_CODE_TRAIN" "$SSD_WEIGHTS" "$TRAIN_WEIGHTS" "$LOG_DIR"
exec > >(
    tee -a "$RUN_LOG" | "$PYTHON_BIN" -u -c '
import re, sys
iter_re = re.compile(r"(\[[^]]+\]) iteration\s+(\d+)/\s*(\d+).*throughput per GPU \(TFLOP/s/GPU\):\s*([0-9.]+)")
val_re = re.compile(r"validation loss at iteration\s+(\d+).*lm loss value:\s*([^|]+)")
save_re = re.compile(r"saving checkpoint at iteration\s+(\d+)")
keep_re = re.compile(r"mixed qv lora|ERROR:|Traceback|failed \(exitcode|checkpoint at|probe ")
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

echo "mixed qv lora run log: $RUN_LOG"
echo "mixed qv lora GPU log: $GPU_LOG"
echo "mixed qv lora metadata: $RUN_METADATA"
echo "mixed qv lora source: $DENSE_BASE_WEIGHTS_DIR at iteration $RESUME_ITERATION"

rsync -rlptD \
    --exclude 'logs/' \
    --exclude 'wandb/' \
    --exclude 'events.out.tfevents*' \
    --exclude 'progress.txt' \
    "$DENSE_BASE_WEIGHTS_DIR/" "$SSD_WEIGHTS/"
if [ "$RESUME_ITERATION" != "latest" ]; then
    printf '%s\n' "$RESUME_ITERATION" > "$SSD_WEIGHTS/latest_checkpointed_iteration.txt"
fi
rsync -rlptD --info=progress2 "$TRAIN_DATASET_WIKI/" "$SSD_WIKI_TRAIN/"
rsync -rlptD --info=progress2 "$TRAIN_DATASET_CODE/" "$SSD_CODE_TRAIN/"

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
from megatron.core.datasets import indexed_dataset

def summarize(dataset_dir: Path):
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
    return {'path': str(dataset_dir), 'tokens': total_tokens, 'documents': total_documents, 'shards': shards}

metadata = {
    'stage': 'mixed_qv_lora_experts',
    'run_id': os.environ['RUN_ID'],
    'base_weights_dir': os.environ['DENSE_BASE_WEIGHTS_DIR'],
    'base_probe_step_offset': int(os.environ['PROBE_STEP_OFFSET']),
    'wiki_train': summarize(Path(os.environ['SSD_WIKI_TRAIN'])),
    'code_train': summarize(Path(os.environ['SSD_CODE_TRAIN'])),
    'train_iters': int(os.environ['TRAIN_ITERS']),
    'micro_batch_size': int(os.environ['MICRO_BATCH_SIZE']),
    'global_batch_size': int(os.environ['GLOBAL_BATCH_SIZE']),
    'num_layers': int(os.environ['NUM_LAYERS']),
    'hidden_size': int(os.environ['HIDDEN_SIZE']),
    'ffn_hidden_size': int(os.environ['FFN_HIDDEN_SIZE']),
    'attn_lora_num_experts': int(os.environ['ATTN_LORA_NUM_EXPERTS']),
    'attn_lora_rank': int(os.environ['ATTN_LORA_RANK']),
    'attn_lora_topk': int(os.environ['ATTN_LORA_TOPK']),
    'attn_lora_alpha': float(os.environ['ATTN_LORA_ALPHA']),
    'precision': os.environ['PRECISION'],
    'train_router_and_experts_only': True,
}
with open(os.environ['RUN_METADATA'], 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
PY

source "$MODEL_CONFIG_SCRIPT"

INFRA_ARGS=(
    --transformer-impl "$TRANSFORMER_IMPL"
    --pipeline-model-parallel-size "$PIPELINE_MODEL_PARALLEL_SIZE"
    --expert-model-parallel-size "$EXPERT_MODEL_PARALLEL_SIZE"
    --distributed-timeout-minutes 30
    --no-persist-layer-norm
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
    --dist-ckpt-strictness assume_ok_unexpected
    --finetune
    --no-load-optim
    --no-load-rng
    --attn-lora-train-router-and-experts-only
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
    --data-path $(build_data_path "$SSD_WIKI_TRAIN" "$SSD_CODE_TRAIN")
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
    --probe-step-offset "$PROBE_STEP_OFFSET"
    --probe-data-path $(build_data_path "$PROBE_DATASET")
    --secondary-probe-name "$SECONDARY_PROBE_NAME"
    --secondary-probe-eval-iters "$SECONDARY_PROBE_EVAL_ITERS"
    --secondary-probe-eval-interval "$SECONDARY_PROBE_EVAL_INTERVAL"
    --secondary-probe-step-offset "$SECONDARY_PROBE_STEP_OFFSET"
    --secondary-probe-data-path $(build_data_path "$SECONDARY_PROBE_DATASET")
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

wait "$TORCHRUN_PID"
kill "$GPU_LOG_PID" 2>/dev/null || true
rsync -rlptD "$SSD_WEIGHTS/" "$TRAIN_WEIGHTS/"

echo "mixed qv lora training complete. Checkpoint at: $TRAIN_WEIGHTS"
echo "mixed qv lora run log saved at: $RUN_LOG"
