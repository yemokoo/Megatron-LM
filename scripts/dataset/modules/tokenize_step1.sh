#!/bin/bash

tokenize() {
    task=$1

    # Skip if task file is empty (already processed).
    [ ! -s "$task" ] && return 0

    # Read source path and local temp path.
    src=$(sed -n '1p' $task)
    file=$(sed -n '2p' $task)

    # Copy from Anvil scratch to local node storage (max 3 attempts).
    for i in {1..3}; do
        echo "Copying $src (Attempt $i of 3)"
        cp "$src" "$file" > /dev/null 2>&1 && break
        echo "Failed to copy $src, retrying..." && sleep 5
        if [ $i -eq 3 ]; then
            echo "ERROR: Failed to copy $src after 3 attempts." >&2
            return 1
        fi
    done

    # Tokenize the file with Megatron-LM (max 3 attempts).
    cd Megatron-LM
    for i in {1..3}; do
        echo "Tokenizing $file (Attempt $i of 3)"
        python tools/preprocess_data.py \
            --input $file \
            --output-prefix ${file%.jsonl} \
            --tokenizer-type HuggingFaceTokenizer \
            --tokenizer-model $TOKENIZER \
            --append-eod \
            --workers $SLURM_CPUS_PER_TASK > /dev/null 2>&1 && break
        echo "Failed to tokenize $file, retrying..." && sleep 5
        if [ $i -eq 3 ]; then
            echo "ERROR: Failed to tokenize $file after 3 attempts." >&2
            return 1
        fi
    done

    # Copy tokenized files to output directory on Anvil scratch (max 3 attempts).
    for i in {1..3}; do
        echo "Saving tokenized files (Attempt $i of 3)"
        mkdir -p "$OUTPUT_DIR" && \
        cp ${file%.jsonl}_text_document.bin ${file%.jsonl}_text_document.idx "$OUTPUT_DIR/" > /dev/null 2>&1 && break
        echo "Failed to save tokenized files, retrying..." && sleep 5
        if [ $i -eq 3 ]; then
            echo "ERROR: Failed to save tokenized files after 3 attempts." >&2
            return 1
        fi
    done

    # Mark task as completed.
    > $task
}

export -f tokenize

# Process task files with file locking to avoid conflicts.
find $NFS_MOUNT -type f -name "*.task" | while read -r line; do
    flock -n $line -c "tokenize $line" || true
done
