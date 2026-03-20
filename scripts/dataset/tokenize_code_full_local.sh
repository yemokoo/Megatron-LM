#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

source /workspace/FLAME-MoE/.conda/etc/profile.d/conda.sh
conda activate flame3090

RAW_DIR="${RAW_DIR:-/workspace/FLAME-MoE/.local/dataset/python-code-full/raw}"
TOK_DIR="${TOK_DIR:-/workspace/FLAME-MoE/.local/dataset/python-code-full/tokenized/EleutherAI/pythia-12b}"
LOG_PATH="${LOG_PATH:-/workspace/FLAME-MoE/.logs/code_full_tokenize.log}"

mkdir -p "$TOK_DIR" "$(dirname "$LOG_PATH")"

for shard in "$RAW_DIR"/shard_*.jsonl; do
    name="$(basename "$shard" .jsonl)"
    if [ -f "$TOK_DIR/${name}_text_document.bin" ] && [ -f "$TOK_DIR/${name}_text_document.idx" ]; then
        echo "SKIP $name"
        continue
    fi

    echo "TOKENIZING $name"
    python Megatron-LM/tools/preprocess_data.py \
        --input "$shard" \
        --output-prefix "$TOK_DIR/$name" \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model EleutherAI/pythia-12b \
        --json-keys text \
        --workers 8 \
        --append-eod
done 2>&1 | tee -a "$LOG_PATH"
