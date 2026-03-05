#!/bin/bash

#SBATCH --job-name=tokenize
#SBATCH --output=logs/%x/%j.log

#SBATCH --partition=gpu
#SBATCH --account=cis251382-gpu
#SBATCH --qos=gpu
#SBATCH --time=00-04:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=240G
#SBATCH --cpus-per-task=24

# Setup the runtime environment.
source scripts/config.sh

# Paths
LOCAL_BASE="/anvil/scratch/x-dlee18/LLM-continual-learning"
NFS_MOUNT="$LOCAL_BASE/tokenize-tasks-$SLURM_JOB_ID"
SSD_MOUNT="/tmp/slurm-$SLURM_JOB_ID"

# TODO: set DATASET and RAW_DATA_DIR to your raw .jsonl text files
# DATASET="my-dataset"
# RAW_DATA_DIR="$LOCAL_BASE/dataset/$DATASET/textfiles"
# OUTPUT_DIR="$LOCAL_BASE/dataset/$DATASET/tokenized/$TOKENIZER"
export TOKENIZER="EleutherAI/pythia-12b"

trap "rm -rf $NFS_MOUNT $SSD_MOUNT" EXIT
mkdir -p $NFS_MOUNT $SSD_MOUNT

# Build a task queue from the local raw text files.
find "$RAW_DATA_DIR" -type f -name "*.jsonl" | while read -r filepath; do
    name=$(basename $filepath)
    file=$SSD_MOUNT/$name
    task=$NFS_MOUNT/$name.task
    echo $filepath >> $task
    echo $file >> $task
done

# Dispatch the tokenization across tasks.
srun -W 0 scripts/dataset/modules/tokenize_step1.sh
