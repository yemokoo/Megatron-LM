#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

source "$SCRIPT_DIR/common.sh"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export FIXED_DATA_SEED="${FIXED_DATA_SEED:-20260607}"
export FINETUNE_FRACTION="${FINETUNE_FRACTION:-0.20}"
export FULL_SAMPLES_PER_TASK="${FULL_SAMPLES_PER_TASK:-4147200}"
export SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-512}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-EleutherAI/pythia-12b}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$LOCAL_BASE/datasets/router_finetune_20pct/seed${FIXED_DATA_SEED}}"
export CACHE_ROOT="${CACHE_ROOT:-$OUTPUT_ROOT/cache}"
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-train_text_document}"
export FORCE_REBUILD="${FORCE_REBUILD:-0}"

export SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-$(
    "$PYTHON_BIN" - <<'PY'
import os
print(int(round(int(os.environ["FULL_SAMPLES_PER_TASK"]) * float(os.environ["FINETUNE_FRACTION"]))))
PY
)}"

if [ "$SAMPLES_PER_TASK" -le 0 ]; then
    echo "[ERROR] SAMPLES_PER_TASK must be positive: $SAMPLES_PER_TASK" >&2
    exit 1
fi

materialize_task() {
    local task="$1"
    local input_dir
    local output_dir
    local cache_dir

    input_dir="$(dataset_dir_for_task "$task")"
    output_dir="$OUTPUT_ROOT/$task/train"
    cache_dir="$CACHE_ROOT/$task"

    if [ ! -d "$input_dir" ]; then
        echo "[ERROR] missing source dataset for $task: $input_dir" >&2
        exit 1
    fi

    if [ -f "$output_dir/${OUTPUT_PREFIX}.bin" ] && [ -f "$output_dir/${OUTPUT_PREFIX}.idx" ]; then
        if [ "$FORCE_REBUILD" != "1" ]; then
            echo "[SKIP] fixed ${task} 20pct dataset already exists: $output_dir"
            return
        fi
        echo "[REBUILD] removing existing fixed ${task} dataset: $output_dir"
        rm -rf "$output_dir"
    elif [ -e "$output_dir" ] && [ "$FORCE_REBUILD" != "1" ]; then
        echo "[ERROR] destination exists but is incomplete: $output_dir" >&2
        echo "[HINT] inspect it first, or rerun with FORCE_REBUILD=1." >&2
        exit 1
    fi

    mkdir -p "$output_dir" "$cache_dir"

    echo "[BUILD] task=$task"
    echo "[BUILD] source=$input_dir"
    echo "[BUILD] output=$output_dir"
    echo "[BUILD] samples=$SAMPLES_PER_TASK seed=$FIXED_DATA_SEED fraction=$FINETUNE_FRACTION"

    "$PYTHON_BIN" scripts/dataset/materialize_fixed_sample_stream.py \
        --input-dir "$input_dir" \
        --output-dir "$output_dir" \
        --samples "$SAMPLES_PER_TASK" \
        --tokenizer-model "$TOKENIZER_MODEL" \
        --dataset-split "100,0,0" \
        --dataset-split-name train \
        --sequence-length "$SEQUENCE_LENGTH" \
        --random-seed "$FIXED_DATA_SEED" \
        --cache-dir "$cache_dir" \
        --output-prefix "$OUTPUT_PREFIX"
}

echo "[CONFIG] fixed router-finetune 20pct data"
echo "[CONFIG] output_root=$OUTPUT_ROOT"
echo "[CONFIG] seed=$FIXED_DATA_SEED"
echo "[CONFIG] fraction=$FINETUNE_FRACTION"
echo "[CONFIG] samples_per_task=$SAMPLES_PER_TASK"
echo "[CONFIG] tokenizer=$TOKENIZER_MODEL"

materialize_task wiki
materialize_task code

cat > "$OUTPUT_ROOT/env.sh" <<EOF
export ROUTER_FINETUNE_DATASET_ROOT="$OUTPUT_ROOT"
export TRAIN_DATASET_WIKI="$OUTPUT_ROOT/wiki/train"
export TRAIN_DATASET_CODE="$OUTPUT_ROOT/code/train"
export RETUNE_ITERS="\${RETUNE_ITERS:-720}"
export DATASET_NAME="\${DATASET_NAME:-wiki_code_fixed_20pct_seed${FIXED_DATA_SEED}}"
export DATASET_SOURCE="\${DATASET_SOURCE:-Fixed 20pct deterministic random GPT sample stream from wiki train + code train; seed=${FIXED_DATA_SEED}; samples_per_task=${SAMPLES_PER_TASK}; sequence_length=${SEQUENCE_LENGTH}}"
EOF

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT"])
metadata = {
    "purpose": "fixed_router_finetune_20pct",
    "output_root": str(root),
    "seed": int(os.environ["FIXED_DATA_SEED"]),
    "fraction": float(os.environ["FINETUNE_FRACTION"]),
    "full_samples_per_task": int(os.environ["FULL_SAMPLES_PER_TASK"]),
    "samples_per_task": int(os.environ["SAMPLES_PER_TASK"]),
    "sequence_length": int(os.environ["SEQUENCE_LENGTH"]),
    "tokenizer_model": os.environ["TOKENIZER_MODEL"],
    "wiki_train": str(root / "wiki" / "train"),
    "code_train": str(root / "code" / "train"),
    "env_file": str(root / "env.sh"),
}
(root / "fixed_20pct_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(json.dumps(metadata, indent=2))
PY

echo "[DONE] fixed 20pct finetune data is ready"
echo "[NEXT] source $OUTPUT_ROOT/env.sh before router finetune runs"
