#!/bin/bash
# Download and tokenize Python code for continual learning experiment (Task B).
# Uses codeparrot/codeparrot-clean (deduplicated Python code, ~50GB).
# Output: $LOCAL_BASE/dataset/python-code/tokenized/EleutherAI/pythia-12b/

#SBATCH --job-name=download-code
#SBATCH --output=logs/%x/%j.log

#SBATCH --partition=gpu
#SBATCH --account=cis251382-gpu
#SBATCH --qos=gpu
#SBATCH --time=4:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:0

source scripts/config.sh

LOCAL_BASE="/anvil/scratch/x-dlee18/LLM-continual-learning"
RAW_DIR="$LOCAL_BASE/dataset/python-code/raw"
TOK_DIR="$LOCAL_BASE/dataset/python-code/tokenized/EleutherAI/pythia-12b"
mkdir -p "$RAW_DIR" "$TOK_DIR"

# Step 1: Download Python code from HuggingFace and write to JSONL
# Using a small subset (~5B tokens) to keep experiment manageable
python3 - <<'EOF'
import os, json
from datasets import load_dataset

raw_dir = os.environ.get("RAW_DIR", "/anvil/scratch/x-dlee18/LLM-continual-learning/dataset/python-code/raw")
os.makedirs(raw_dir, exist_ok=True)

# codeparrot-clean: ~50GB deduplicated Python code
# For a small-scale experiment, take the first 200k examples (~2-3B tokens)
print("Loading codeparrot/codeparrot-clean from HuggingFace...")
ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)

shard_size = 50_000
max_examples = 200_000  # adjust to control data size
shard_idx = 0
buf = []

for i, ex in enumerate(ds):
    if i >= max_examples:
        break
    buf.append({"text": ex["content"]})
    if len(buf) >= shard_size:
        out_path = os.path.join(raw_dir, f"shard_{shard_idx:04d}.jsonl")
        with open(out_path, "w") as f:
            for item in buf:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Wrote {out_path} ({i+1} examples total)")
        shard_idx += 1
        buf = []

if buf:
    out_path = os.path.join(raw_dir, f"shard_{shard_idx:04d}.jsonl")
    with open(out_path, "w") as f:
        for item in buf:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path}")

print("Download complete.")
EOF

export RAW_DIR="$RAW_DIR"

# Step 2: Tokenize each shard
for shard in "$RAW_DIR"/shard_*.jsonl; do
    name=$(basename "$shard" .jsonl)
    python Megatron-LM/tools/preprocess_data.py \
        --input "$shard" \
        --output-prefix "$TOK_DIR/$name" \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model EleutherAI/pythia-12b \
        --json-key text \
        --workers 16 \
        --chunk-size 512 \
        --append-eod
done

echo "Tokenization complete. Output at: $TOK_DIR"
