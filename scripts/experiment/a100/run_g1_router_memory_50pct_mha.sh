#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-300}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1800}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export BASE_STAGE_DIR="${BASE_STAGE_DIR:-$PROJECT_ROOT/.local/weights/a100/mha/shared-router-granularity-qkvo}"

# G1 setting: top-2, 4 -> 8 experts, FFN 704, QKVO full-rank LoRA rank 512.
export RUN_ID="${RUN_ID:-g1-top2-e4to8-ffn704-r512-wiki-to-code-shared-router-qkvo-mha-a100-bf16-mb96-router-memory-fixed50-kl0p1-1800}"
export STAGE1_WEIGHTS_DIR="${STAGE1_WEIGHTS_DIR:-$BASE_STAGE_DIR/wiki/g1-top2-e4-ffn704-r512-wiki-shared-router-qkvo-mha-a100-bf16-mb72-1800}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$BASE_STAGE_DIR/code/$RUN_ID}"

# Router-memory 50% replay: one Wiki router-only KL update every two Code steps.
# The fixed memory pool is also 10x larger than the 5% replay pool, so the
# higher replay frequency is not silently reusing the smaller sample stream.
# No early stop: keep applying router-memory KL through the full Code budget.
export ROUTER_MEMORY_KL_COEFF="${ROUTER_MEMORY_KL_COEFF:-0.1}"
export ROUTER_MEMORY_FRACTION="${ROUTER_MEMORY_FRACTION:-0.5}"
export ROUTER_MEMORY_INTERVAL="${ROUTER_MEMORY_INTERVAL:-2}"
export ROUTER_MEMORY_DATASET="${ROUTER_MEMORY_DATASET:-$PROJECT_ROOT/data/wiki/router_memory_50pct}"
export ROUTER_MEMORY_EVAL_DATASET="${ROUTER_MEMORY_EVAL_DATASET:-$ROUTER_MEMORY_DATASET}"
# Diagnostic KL is measured on the same fixed mini-set every 100 Code steps.
export ROUTER_MEMORY_EVAL_INTERVAL="${ROUTER_MEMORY_EVAL_INTERVAL:-100}"
export ROUTER_MEMORY_EVAL_ITERS="${ROUTER_MEMORY_EVAL_ITERS:-1}"
export ROUTER_KL_EARLY_STOP_ENABLED=0
export ROUTER_KL_STOP_STEP=""

export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"
export ATTN_LORA_GROUPED_GEMM="${ATTN_LORA_GROUPED_GEMM:-1}"
export MOE_PERMUTE_FUSION="${MOE_PERMUTE_FUSION:-0}"
export MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}"

is_completed() {
    local run_dir="$1"
    local latest_file="${run_dir}/latest_checkpointed_iteration.txt"
    [ -f "$latest_file" ] && [ "$(tr -d '\n\r[:space:]' < "$latest_file")" = "$TRAIN_ITERS" ]
}

if ! is_completed "$STAGE1_WEIGHTS_DIR"; then
    echo "[ERROR] missing completed G1 wiki checkpoint ${TRAIN_ITERS}: ${STAGE1_WEIGHTS_DIR}" >&2
    exit 1
fi

if [ "$ROUTER_MEMORY_KL_COEFF" != "0" ] && [ "$ROUTER_MEMORY_KL_COEFF" != "0.0" ]; then
    if ! compgen -G "$ROUTER_MEMORY_DATASET/*.bin" >/dev/null; then
        echo "[ERROR] router memory dataset not found: $ROUTER_MEMORY_DATASET" >&2
        echo "[ERROR] create the 50pct fixed memory pool first, e.g. data/wiki/router_memory_50pct" >&2
        exit 1
    fi
    python - "$ROUTER_MEMORY_DATASET/router_memory_metadata.json" "$TRAIN_ITERS" "$GLOBAL_BATCH_SIZE" "$ROUTER_MEMORY_INTERVAL" <<'PY'
import json
import math
import sys
from pathlib import Path

metadata_path = Path(sys.argv[1])
train_iters = int(sys.argv[2])
global_batch_size = int(sys.argv[3])
router_interval = max(1, int(sys.argv[4]))
expected = math.ceil(train_iters / router_interval) * global_batch_size

if not metadata_path.exists():
    raise SystemExit(f"[ERROR] missing router memory metadata: {metadata_path}")

metadata = json.loads(metadata_path.read_text())
actual = int(metadata.get("memory", {}).get("samples", 0))
if actual < expected:
    raise SystemExit(
        "[ERROR] router memory pool is too small for 50pct replay: "
        f"actual={actual}, expected_at_least={expected}. "
        "Recreate data/wiki/router_memory_50pct with 2073600 samples for the default G1 run."
    )

print(f"[CONFIG] router memory samples={actual} expected_at_least={expected}")
PY
fi

if is_completed "$TRAIN_WEIGHTS"; then
    echo "[SKIP] already completed: $RUN_ID"
    exit 0
fi

echo "[CONFIG] G1 router-memory 50pct replay"
echo "[CONFIG] code steps=${TRAIN_ITERS}, router replay interval=${ROUTER_MEMORY_INTERVAL}, fraction=${ROUTER_MEMORY_FRACTION}"
echo "[CONFIG] early stop disabled: ROUTER_KL_EARLY_STOP_ENABLED=${ROUTER_KL_EARLY_STOP_ENABLED}"
echo "[CONFIG] wiki=${STAGE1_WEIGHTS_DIR}"
echo "[CONFIG] code=${TRAIN_WEIGHTS}"

env \
    WANDB_MODE="$WANDB_MODE" \
    DIRECT_LOCAL_SAVE=1 \
    RUN_ID="$RUN_ID" \
    TRAIN_WEIGHTS="$TRAIN_WEIGHTS" \
    STAGE1_WEIGHTS_DIR="$STAGE1_WEIGHTS_DIR" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_EXP_NAME="G1 - router replay 50pct kl0p1 - wiki to code" \
    LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND=0 \
    RUN_INITIAL_PROBE_EVAL=1 \
    PROBE_EVAL_INTERVAL=100 \
    SECONDARY_PROBE_EVAL_INTERVAL=100 \
    MICRO_BATCH_SIZE=96 \
    GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
    TRAIN_ITERS="$TRAIN_ITERS" \
    SAVE_INTERVAL="$SAVE_INTERVAL" \
    EVAL_INTERVAL="$EVAL_INTERVAL" \
    SOURCE_NUM_EXPERTS=4 \
    NUM_EXPERTS=8 \
    MOE_ROUTER_TOPK=2 \
    MOE_FFN_HIDDEN_SIZE=704 \
    ENABLE_OLD_MODEL_KL=0 \
    OLD_MODEL_KL_COEFF=0.0 \
    OLD_MODEL_KL_TEMPERATURE=1.0 \
    ATTN_LORA_RANK=512 \
    ATTN_LORA_ALPHA=512 \
    ATTN_FULL_RANK_LORA_RANK=512 \
    ATTN_FULL_RANK_LORA_ALPHA=512 \
    ATTN_FULL_RANK_LORA_TARGETS=qkvo \
    ATTN_FULL_RANK_LORA_ACTIVE_TARGETS= \
    ATTN_LORA_GROUPED_GEMM="$ATTN_LORA_GROUPED_GEMM" \
    MOE_GROUPED_GEMM="$MOE_GROUPED_GEMM" \
    MOE_PERMUTE_FUSION="$MOE_PERMUTE_FUSION" \
    MOE_ROUTER_DTYPE="$MOE_ROUTER_DTYPE" \
    ROUTER_MEMORY_KL_COEFF="$ROUTER_MEMORY_KL_COEFF" \
    ROUTER_MEMORY_FRACTION="$ROUTER_MEMORY_FRACTION" \
    ROUTER_MEMORY_INTERVAL="$ROUTER_MEMORY_INTERVAL" \
    ROUTER_MEMORY_DATASET="$ROUTER_MEMORY_DATASET" \
    ROUTER_MEMORY_EVAL_DATASET="$ROUTER_MEMORY_EVAL_DATASET" \
    ROUTER_MEMORY_EVAL_INTERVAL="$ROUTER_MEMORY_EVAL_INTERVAL" \
    ROUTER_MEMORY_EVAL_ITERS="$ROUTER_MEMORY_EVAL_ITERS" \
    ROUTER_KL_EARLY_STOP_ENABLED="$ROUTER_KL_EARLY_STOP_ENABLED" \
    ROUTER_KL_STOP_STEP="$ROUTER_KL_STOP_STEP" \
    MASTER_PORT="${MASTER_PORT:-29722}" \
    "$SCRIPT_DIR/run_guarded_training.sh" \
    bash "$SCRIPT_DIR/code_from_wiki_e6_fullrank_qkvo_expert_mha_a100_bf16.sh"

echo "[DONE] G1 router-memory 50pct replay $(date)"
