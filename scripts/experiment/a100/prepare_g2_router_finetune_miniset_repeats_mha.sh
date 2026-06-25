#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

source "$SCRIPT_DIR/common.sh"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export FIXED_DATA_SEED="${FIXED_DATA_SEED:-1234}"
export FULL_SAMPLES_PER_TASK="${FULL_SAMPLES_PER_TASK:-4147200}"
export SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-512}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-EleutherAI/pythia-12b}"
export OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-$LOCAL_BASE/datasets/router_finetune_miniset_repeats/seed${FIXED_DATA_SEED}}"
export CACHE_ROOT_BASE="${CACHE_ROOT_BASE:-$OUTPUT_ROOT_BASE/cache}"
export OUTPUT_PREFIX="${OUTPUT_PREFIX:-train_text_document}"
export FORCE_REBUILD="${FORCE_REBUILD:-0}"

# label:fraction:repeat_epochs. All defaults equal 20pct total budget.
export MINISET_SPECS="${MINISET_SPECS:-0p01pctx2000:0.0001:2000 0p1pctx200:0.001:200 1pctx20:0.01:20 5pctx4:0.05:4 10pctx2:0.10:2}"
export FULL_100PCT_RETUNE_ITERS="${FULL_100PCT_RETUNE_ITERS:-3600}"
export TARGET_TOTAL_FRACTION="${TARGET_TOTAL_FRACTION:-0.20}"
export TARGET_RETUNE_ITERS="${TARGET_RETUNE_ITERS:-720}"

samples_for_fraction() {
    local fraction="$1"
    "$PYTHON_BIN" - "$fraction" <<'PY'
import os
import sys

fraction = float(sys.argv[1])
full_samples = int(os.environ["FULL_SAMPLES_PER_TASK"])
print(int(round(full_samples * fraction)))
PY
}

materialize_task() {
    local label="$1"
    local fraction="$2"
    local repeat_epochs="$3"
    local task="$4"
    local samples_per_task="$5"

    local input_dir
    local output_dir
    local cache_dir

    input_dir="$(dataset_dir_for_task "$task")"
    output_dir="$OUTPUT_ROOT_BASE/$label/$task/train"
    cache_dir="$CACHE_ROOT_BASE/$label/$task"

    if [ ! -d "$input_dir" ]; then
        echo "[ERROR] missing source dataset for $task: $input_dir" >&2
        exit 1
    fi

    if [ -f "$output_dir/${OUTPUT_PREFIX}.bin" ] && [ -f "$output_dir/${OUTPUT_PREFIX}.idx" ]; then
        if [ "$FORCE_REBUILD" != "1" ]; then
            echo "[SKIP] fixed ${label} ${task} dataset already exists: $output_dir"
            return
        fi
        echo "[REBUILD] removing existing fixed ${label} ${task} dataset: $output_dir"
        rm -rf "$output_dir"
    elif [ -e "$output_dir" ] && [ "$FORCE_REBUILD" != "1" ]; then
        echo "[ERROR] destination exists but is incomplete: $output_dir" >&2
        echo "[HINT] inspect it first, or rerun with FORCE_REBUILD=1." >&2
        exit 1
    fi

    mkdir -p "$output_dir" "$cache_dir"

    echo "[BUILD] label=$label task=$task"
    echo "[BUILD] source=$input_dir"
    echo "[BUILD] output=$output_dir"
    echo "[BUILD] fraction=$fraction repeat_epochs=$repeat_epochs samples=$samples_per_task seed=$FIXED_DATA_SEED"

    "$PYTHON_BIN" scripts/dataset/materialize_fixed_sample_stream.py \
        --input-dir "$input_dir" \
        --output-dir "$output_dir" \
        --samples "$samples_per_task" \
        --tokenizer-model "$TOKENIZER_MODEL" \
        --dataset-split "100,0,0" \
        --dataset-split-name train \
        --sequence-length "$SEQUENCE_LENGTH" \
        --random-seed "$FIXED_DATA_SEED" \
        --cache-dir "$cache_dir" \
        --output-prefix "$OUTPUT_PREFIX"
}

write_spec_metadata() {
    local label="$1"
    local fraction="$2"
    local repeat_epochs="$3"
    local samples_per_task="$4"
    local spec_root="$OUTPUT_ROOT_BASE/$label"

    cat > "$spec_root/env.sh" <<EOF
export ROUTER_FINETUNE_DATASET_ROOT="$spec_root"
export TRAIN_DATASET_WIKI="$spec_root/wiki/train"
export TRAIN_DATASET_CODE="$spec_root/code/train"
export RETUNE_ITERS="\${RETUNE_ITERS:-$TARGET_RETUNE_ITERS}"
export SAVE_INTERVAL="\${SAVE_INTERVAL:-36}"
export PROBE_EVAL_INTERVAL="\${PROBE_EVAL_INTERVAL:-36}"
export SECONDARY_PROBE_EVAL_INTERVAL="\${SECONDARY_PROBE_EVAL_INTERVAL:-36}"
export DATASET_NAME="\${DATASET_NAME:-wiki_code_fixed_${label}_seed${FIXED_DATA_SEED}}"
export DATASET_SOURCE="\${DATASET_SOURCE:-Fixed ${label} deterministic wiki/code train subsets; seed=${FIXED_DATA_SEED}; fraction=${fraction}; repeat_epochs=${repeat_epochs}; samples_per_task=${samples_per_task}; sequence_length=${SEQUENCE_LENGTH}}"
EOF

    "$PYTHON_BIN" - "$label" "$fraction" "$repeat_epochs" "$samples_per_task" <<'PY'
import json
import os
import sys
from pathlib import Path

label, fraction, repeat_epochs, samples_per_task = sys.argv[1:]
root = Path(os.environ["OUTPUT_ROOT_BASE"]) / label
metadata = {
    "purpose": "fixed_router_finetune_miniset_repeat",
    "label": label,
    "output_root": str(root),
    "seed": int(os.environ["FIXED_DATA_SEED"]),
    "fraction": float(fraction),
    "repeat_epochs": int(repeat_epochs),
    "target_total_fraction": float(os.environ["TARGET_TOTAL_FRACTION"]),
    "full_100pct_retune_iters": int(os.environ["FULL_100PCT_RETUNE_ITERS"]),
    "target_retune_iters": int(os.environ["TARGET_RETUNE_ITERS"]),
    "full_samples_per_task": int(os.environ["FULL_SAMPLES_PER_TASK"]),
    "samples_per_task": int(samples_per_task),
    "sequence_length": int(os.environ["SEQUENCE_LENGTH"]),
    "tokenizer_model": os.environ["TOKENIZER_MODEL"],
    "wiki_train": str(root / "wiki" / "train"),
    "code_train": str(root / "code" / "train"),
    "env_file": str(root / "env.sh"),
}
(root / "fixed_miniset_repeat_metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)
print(json.dumps(metadata, indent=2))
PY
}

echo "[CONFIG] fixed router-finetune miniset repeats"
echo "[CONFIG] output_root_base=$OUTPUT_ROOT_BASE"
echo "[CONFIG] seed=$FIXED_DATA_SEED"
echo "[CONFIG] specs=$MINISET_SPECS"
echo "[CONFIG] target_retune_iters=$TARGET_RETUNE_ITERS"
echo "[CONFIG] tokenizer=$TOKENIZER_MODEL"

for spec in $MINISET_SPECS; do
    IFS=: read -r label fraction repeat_epochs <<< "$spec"
    if [ -z "$label" ] || [ -z "$fraction" ] || [ -z "$repeat_epochs" ]; then
        echo "[ERROR] bad MINISET_SPECS entry: $spec" >&2
        exit 1
    fi

    samples_per_task="$(samples_for_fraction "$fraction")"
    if [ "$samples_per_task" -le 0 ]; then
        echo "[ERROR] samples_per_task must be positive for $spec: $samples_per_task" >&2
        exit 1
    fi

    materialize_task "$label" "$fraction" "$repeat_epochs" wiki "$samples_per_task"
    materialize_task "$label" "$fraction" "$repeat_epochs" code "$samples_per_task"
    write_spec_metadata "$label" "$fraction" "$repeat_epochs" "$samples_per_task"
done

echo "[DONE] fixed miniset-repeat router-finetune data is ready"
echo "[NEXT] source one of:"
for spec in $MINISET_SPECS; do
    IFS=: read -r label _fraction _repeat_epochs <<< "$spec"
    echo "  source $OUTPUT_ROOT_BASE/$label/env.sh"
done
