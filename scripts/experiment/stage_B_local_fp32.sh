#!/bin/bash
set -euo pipefail

# Stage B (local, experimental fp32 path): expand a Stage A checkpoint and
# continue training with conservative fp32 defaults on 3090-class GPUs.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export RUN_ID="${RUN_ID:-stage-b-local-fp32-$(date -u +%Y%m%d-%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29511}"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_DATASET="${LOCAL_DATASET:-$LOCAL_BASE/dataset}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"

export SSD_MOUNT="${LOCAL_SSD_ROOT}/${RUN_ID}"
export SSD_DATASET="${SSD_MOUNT}/dataset"
export SSD_WEIGHTS="${SSD_MOUNT}/weights"

resolve_stage_a_dir() {
    python3 - <<'PY'
import json
import os
from pathlib import Path

stage_a_weights_dir = os.environ.get("STAGE_A_WEIGHTS_DIR", "").strip()
if stage_a_weights_dir:
    print(stage_a_weights_dir)
    raise SystemExit(0)

local_weights = Path(os.environ["LOCAL_WEIGHTS"]) / "continual-stage-A-local"
requested_run_id = os.environ.get("STAGE_A_RUN_ID", "").strip()
if requested_run_id:
    candidate = local_weights / requested_run_id
    if candidate.exists():
        print(candidate)
        raise SystemExit(0)
    raise SystemExit(f"ERROR: requested STAGE_A_RUN_ID not found: {candidate}")

best = None
best_mtime = -1.0
for candidate in local_weights.iterdir() if local_weights.exists() else []:
    if not candidate.is_dir():
        continue
    tracker = candidate / "latest_checkpointed_iteration.txt"
    metadata = candidate / "logs" / "run_metadata.json"
    if not tracker.exists() or not metadata.exists():
        continue
    try:
        tracker_step = int(tracker.read_text(encoding="utf-8").strip())
        train_iters = int(json.loads(metadata.read_text(encoding="utf-8"))["train_iters"])
    except Exception:
        continue
    if tracker_step < train_iters:
        continue
    candidate_mtime = tracker.stat().st_mtime
    if candidate_mtime > best_mtime:
        best = candidate
        best_mtime = candidate_mtime

if best is None:
    raise SystemExit(
        "ERROR: could not find a completed Stage A fp32 run. "
        "Set STAGE_A_RUN_ID or STAGE_A_WEIGHTS_DIR explicitly."
    )

print(best)
PY
}

read_stage_a_train_iters() {
    python3 - <<'PY'
import json
import os
from pathlib import Path

metadata_path = Path(os.environ["STAGE_A_WEIGHTS_DIR"]) / "logs" / "run_metadata.json"
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
print(int(metadata["train_iters"]))
PY
}

export NUM_LAYERS="${NUM_LAYERS:-9}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-5472}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-704}"
export MOE_LAYER_FREQ="${MOE_LAYER_FREQ:-[0]*1+[1]*8}"
export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-4}"
export NUM_EXPERTS="${NUM_EXPERTS:-7}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-2}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=1
export TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-local}"
export PRECISION="fp32"
export USE_DISTRIBUTED_OPTIMIZER="${USE_DISTRIBUTED_OPTIMIZER:-0}"
export CHECK_NAN_IN_LOSS_AND_GRAD="${CHECK_NAN_IN_LOSS_AND_GRAD:-1}"
export TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY="${TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY:-0}"

export TRAIN_ITERS="${TRAIN_ITERS:-}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-}"
export OLD_MODEL_KL_COEFF="${OLD_MODEL_KL_COEFF:-0.01}"
export OLD_MODEL_KL_TEMPERATURE="${OLD_MODEL_KL_TEMPERATURE:-1.0}"
export GPU_LOG_INTERVAL_SECONDS="${GPU_LOG_INTERVAL_SECONDS:-30}"
export LR="${LR:-3e-4}"
export MIN_LR="${MIN_LR:-3e-5}"
export LR_DECAY_STYLE="${LR_DECAY_STYLE:-WSD}"
export LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.01}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-}"
export LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-configs/model/flame-moe.sh}"

export TRAIN_DATASET="${TRAIN_DATASET:-$LOCAL_DATASET/python-code-full/tokenized/EleutherAI/pythia-12b}"
export STAGE_A_WEIGHTS_DIR="${STAGE_A_WEIGHTS_DIR:-}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$LOCAL_WEIGHTS/continual-stage-B-local/$RUN_ID}"
export LOG_DIR="${LOG_DIR:-$TRAIN_WEIGHTS/logs}"
export RUN_LOG="${RUN_LOG:-$LOG_DIR/stage_b.log}"
export GPU_LOG="${GPU_LOG:-$LOG_DIR/gpu_usage.csv}"
export RUN_METADATA="${RUN_METADATA:-$LOG_DIR/run_metadata.json}"
export DATASET_NAME="${DATASET_NAME:-python_code}"
export DATASET_SOURCE="${DATASET_SOURCE:-Hugging Face codeparrot/codeparrot-clean}"
export PROBE_DATASET="${PROBE_DATASET:-$LOCAL_DATASET/wikipedia-full/tokenized/EleutherAI/pythia-12b}"
export PROBE_NAME="${PROBE_NAME:-task_a_probe}"
export PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-25}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-10}"
export SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET:-$TRAIN_DATASET}"
export SECONDARY_PROBE_NAME="${SECONDARY_PROBE_NAME:-task_b_probe}"
export SECONDARY_PROBE_EVAL_ITERS="${SECONDARY_PROBE_EVAL_ITERS:-$PROBE_EVAL_ITERS}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-$PROBE_EVAL_INTERVAL}"
export SECONDARY_PROBE_STEP_OFFSET="${SECONDARY_PROBE_STEP_OFFSET:-}"
export STAGE_A_RUN_LOG="${STAGE_A_RUN_LOG:-}"
export PROBE_STEP_OFFSET="${PROBE_STEP_OFFSET:-}"
export WANDB_STEP_OFFSET="${WANDB_STEP_OFFSET:-}"
export CONTINUAL_PLOT_PREFIX="${CONTINUAL_PLOT_PREFIX:-$LOG_DIR/task_a_probe_continual}"
export WANDB_PROJECT="${WANDB_PROJECT:-}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-$RUN_ID}"
export WANDB_SAVE_DIR="${WANDB_SAVE_DIR:-$TRAIN_WEIGHTS/wandb}"

if [ "$TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY" = "1" ]; then
    export STAGE_B_LABEL="${STAGE_B_LABEL:-Stage B new-expert-router-only}"
    export STAGE_B_STAGE_NAME="${STAGE_B_STAGE_NAME:-B_new_expert_router_only}"
else
    export STAGE_B_LABEL="${STAGE_B_LABEL:-Stage B}"
    export STAGE_B_STAGE_NAME="${STAGE_B_STAGE_NAME:-B}"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-8}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"

if [ "$SOURCE_NUM_EXPERTS" -ge "$NUM_EXPERTS" ]; then
    echo "ERROR: SOURCE_NUM_EXPERTS must be smaller than NUM_EXPERTS."
    exit 1
fi

export STAGE_A_WEIGHTS_DIR="$(resolve_stage_a_dir)"
export STAGE_A_RUN_LOG="${STAGE_A_RUN_LOG:-$STAGE_A_WEIGHTS_DIR/logs/stage_a.log}"
export PROBE_STEP_OFFSET="${PROBE_STEP_OFFSET:-$(read_stage_a_train_iters)}"
export SECONDARY_PROBE_STEP_OFFSET="${SECONDARY_PROBE_STEP_OFFSET:-$PROBE_STEP_OFFSET}"
export WANDB_STEP_OFFSET="${WANDB_STEP_OFFSET:-$PROBE_STEP_OFFSET}"

mkdir -p "$SSD_DATASET" "$SSD_WEIGHTS" "$TRAIN_WEIGHTS" "$LOG_DIR"
exec > >(
    tee -a "$RUN_LOG" | python3 -u -c '
import os
import re
import sys

iter_re = re.compile(r"(\[[^]]+\]) iteration\s+(\d+)/\s*(\d+).*throughput per GPU \(TFLOP/s/GPU\):\s*([0-9.]+)")
val_re = re.compile(r"validation loss at iteration\s+(\d+).*lm loss value:\s*([^|]+)")
save_re = re.compile(r"saving checkpoint at iteration\s+(\d+)")
label = re.escape(os.environ["STAGE_B_LABEL"])
keep_re = re.compile(
    rf"{label} run log:|{label} GPU log:|{label} metadata:|{label} local fp32 complete|"
    rf"{label} run log saved at:|{label} GPU log saved at:|{label} metadata saved at:|"
    rf"{label} expansion audit dir:|Expanded MoE checkpoint|ERROR:|Traceback|failed \(exitcode|checkpoint at|probe "
)

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

echo "$STAGE_B_LABEL run log: $RUN_LOG"
echo "$STAGE_B_LABEL GPU log: $GPU_LOG"
echo "$STAGE_B_LABEL metadata: $RUN_METADATA"
echo "Stage A source checkpoint: $STAGE_A_WEIGHTS_DIR"
echo "Stage A probe log: $STAGE_A_RUN_LOG"
echo "$STAGE_B_LABEL probe step offset: $PROBE_STEP_OFFSET"
echo "$STAGE_B_LABEL output checkpoint: $TRAIN_WEIGHTS"
rsync -rlptD \
    --exclude 'logs/' \
    --exclude 'events.out.tfevents*' \
    --exclude 'progress.txt' \
    "$STAGE_A_WEIGHTS_DIR/" "$SSD_WEIGHTS/"
rsync -rlptD --info=progress2 "$TRAIN_DATASET/" "$SSD_DATASET/"

if [ -z "${TRAIN_ITERS:-}" ]; then
    export TRAIN_ITERS="$(python - <<'PY'
from pathlib import Path
from megatron.core.datasets import indexed_dataset
import os

dataset_dir = Path(os.environ["SSD_DATASET"])
seq_length = int(os.environ.get("SEQ_LENGTH", "512"))
global_batch_size = int(os.environ.get("GLOBAL_BATCH_SIZE", "16"))
train_split = float(os.environ.get("TRAIN_SPLIT_FRACTION", "0.95"))

total_tokens = 0
for idx_path in sorted(dataset_dir.glob("*.idx")):
    ds = indexed_dataset.IndexedDataset(str(idx_path.with_suffix("")), multimodal=False, mmap=True)
    total_tokens += int(ds.sequence_lengths.sum())

approx_train_sequences = int(total_tokens * train_split) // seq_length
approx_train_steps = max(1, approx_train_sequences // global_batch_size)
print(approx_train_steps)
PY
)"
fi

export SAVE_INTERVAL="${SAVE_INTERVAL:-$TRAIN_ITERS}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-$TRAIN_ITERS}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-$TRAIN_ITERS}"
export LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-$((TRAIN_ITERS / 10))}"

python - <<'PY'
import json
import os
from pathlib import Path
from megatron.core.datasets import indexed_dataset

dataset_dir = Path(os.environ["SSD_DATASET"])
seq_length = int(os.environ.get("SEQ_LENGTH", "512"))
global_batch_size = int(os.environ.get("GLOBAL_BATCH_SIZE", "16"))
train_split = float(os.environ.get("TRAIN_SPLIT_FRACTION", "0.95"))

total_tokens = 0
total_documents = 0
shards = []
for idx_path in sorted(dataset_dir.glob("*.idx")):
    prefix = idx_path.with_suffix("")
    ds = indexed_dataset.IndexedDataset(str(prefix), multimodal=False, mmap=True)
    shard_tokens = int(ds.sequence_lengths.sum())
    shard_docs = int(ds.document_indices.shape[0] - 1)
    total_tokens += shard_tokens
    total_documents += shard_docs
    shards.append({
        "prefix": prefix.name,
        "documents": shard_docs,
        "tokens": shard_tokens,
    })

metadata = {
    "stage": os.environ["STAGE_B_STAGE_NAME"],
    "run_id": os.environ["RUN_ID"],
    "dataset_name": os.environ["DATASET_NAME"],
    "dataset_source": os.environ["DATASET_SOURCE"],
    "dataset_path": str(dataset_dir),
    "stage_a_weights_dir": os.environ["STAGE_A_WEIGHTS_DIR"],
    "source_num_experts": int(os.environ["SOURCE_NUM_EXPERTS"]),
    "target_num_experts": int(os.environ["NUM_EXPERTS"]),
    "moe_router_topk": int(os.environ["MOE_ROUTER_TOPK"]),
    "num_shards": len(shards),
    "total_documents": total_documents,
    "total_tokens": total_tokens,
    "seq_length": seq_length,
    "global_batch_size": global_batch_size,
    "micro_batch_size": int(os.environ.get("MICRO_BATCH_SIZE", "1")),
    "train_split_fraction": train_split,
    "approx_train_sequences": int(total_tokens * train_split) // seq_length,
    "train_iters": int(os.environ["TRAIN_ITERS"]),
    "save_interval": int(os.environ["SAVE_INTERVAL"]),
    "eval_interval": int(os.environ["EVAL_INTERVAL"]),
    "old_model_kl_enabled": os.environ.get("TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY", "0") != "1",
    "old_model_kl_coeff": None if os.environ.get("TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY", "0") == "1" else float(os.environ["OLD_MODEL_KL_COEFF"]),
    "old_model_kl_temperature": None if os.environ.get("TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY", "0") == "1" else float(os.environ["OLD_MODEL_KL_TEMPERATURE"]),
    "probe_step_offset": int(os.environ["PROBE_STEP_OFFSET"]),
    "stage_a_run_log": os.environ["STAGE_A_RUN_LOG"],
    "hidden_size": int(os.environ["HIDDEN_SIZE"]),
    "ffn_hidden_size": int(os.environ["FFN_HIDDEN_SIZE"]),
    "moe_ffn_hidden_size": int(os.environ["MOE_FFN_HIDDEN_SIZE"]),
    "num_layers": int(os.environ["NUM_LAYERS"]),
    "precision": os.environ["PRECISION"],
    "shards": shards,
}

with open(os.environ["RUN_METADATA"], "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
PY

source "$MODEL_CONFIG_SCRIPT"

INFRA_ARGS=(
    --transformer-impl "$TRANSFORMER_IMPL"
    --pipeline-model-parallel-size "$PIPELINE_MODEL_PARALLEL_SIZE"
    --expert-model-parallel-size "$EXPERT_MODEL_PARALLEL_SIZE"
    --moe-token-dispatcher-type alltoall
    --distributed-timeout-minutes 30
    --no-persist-layer-norm
    --no-gradient-accumulation-fusion
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
    --accumulate-allreduce-grads-in-fp32
    --main-grads-dtype fp32
    --main-params-dtype fp32
    --exp-avg-dtype fp32
    --exp-avg-sq-dtype fp32
)

if [ "$USE_DISTRIBUTED_OPTIMIZER" = "1" ]; then
    INFRA_ARGS+=(--use-distributed-optimizer)
fi

if [ "$CHECK_NAN_IN_LOSS_AND_GRAD" != "1" ]; then
    INFRA_ARGS+=(--no-check-for-nan-in-loss-and-grad)
fi

TRAIN_ARGS=(
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
    --data-path $(find "$SSD_DATASET" -type f -name '*.bin' -exec sh -c 'printf "1.0 %s " "${1%.bin}"' _ {} \; | sed 's/ $//')
    --split 95,5,0
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
    --no-load-optim
    --no-load-rng
    --finetune
    --moe-expand-from-num-experts "$SOURCE_NUM_EXPERTS"
    --moe-freeze-existing-experts
    --moe-freeze-existing-router
)

if [ "$TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY" = "1" ]; then
    SAVE_ARGS+=(
        --moe-train-new-experts-and-router-only
    )
else
    SAVE_ARGS+=(
        --moe-old-model-kl-coeff "$OLD_MODEL_KL_COEFF"
        --moe-old-model-kl-temperature "$OLD_MODEL_KL_TEMPERATURE"
    )
fi

PROBE_ARGS=()
if [ -n "$PROBE_DATASET" ]; then
    PROBE_ARGS+=(
        --probe-name "$PROBE_NAME"
        --probe-eval-iters "$PROBE_EVAL_ITERS"
        --probe-eval-interval "$PROBE_EVAL_INTERVAL"
        --probe-step-offset "$PROBE_STEP_OFFSET"
        --probe-data-path $(find "$PROBE_DATASET" -type f -name '*.bin' -exec sh -c 'printf "1.0 %s " "${1%.bin}"' _ {} \; | sed 's/ $//')
    )
fi

if [ -n "$SECONDARY_PROBE_DATASET" ]; then
    PROBE_ARGS+=(
        --secondary-probe-name "$SECONDARY_PROBE_NAME"
        --secondary-probe-eval-iters "$SECONDARY_PROBE_EVAL_ITERS"
        --secondary-probe-eval-interval "$SECONDARY_PROBE_EVAL_INTERVAL"
        --secondary-probe-step-offset "$SECONDARY_PROBE_STEP_OFFSET"
        --secondary-probe-data-path $(find "$SECONDARY_PROBE_DATASET" -type f -name '*.bin' -exec sh -c 'printf "1.0 %s " "${1%.bin}"' _ {} \; | sed 's/ $//')
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

python -m torch.distributed.run \
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

if [ -f "$STAGE_A_RUN_LOG" ] && [ -f "$RUN_LOG" ]; then
    python3 "$PROJECT_ROOT/analysis/plot_continual_probe.py" \
        --stage-a-log "$STAGE_A_RUN_LOG" \
        --stage-b-log "$RUN_LOG" \
        --probe-name "$PROBE_NAME" \
        --output-prefix "$CONTINUAL_PLOT_PREFIX"
fi

echo "$STAGE_B_LABEL local fp32 complete. Checkpoint at: $TRAIN_WEIGHTS"
echo "$STAGE_B_LABEL run log saved at: $RUN_LOG"
echo "$STAGE_B_LABEL GPU log saved at: $GPU_LOG"
echo "$STAGE_B_LABEL metadata saved at: $RUN_METADATA"
echo "$STAGE_B_LABEL expansion audit dir: $TRAIN_WEIGHTS/expansion_audit"
echo "$STAGE_B_LABEL continual probe plots: ${CONTINUAL_PLOT_PREFIX}_accuracy.png ${CONTINUAL_PLOT_PREFIX}_ppl.png"
