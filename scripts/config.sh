#!/bin/bash
# Anvil (Purdue) cluster configuration

# --- Local paths on Anvil scratch ---
export LOCAL_BASE="/anvil/scratch/x-dlee18/LLM-continual-learning"
export SSD_MOUNT="/tmp/slurm-$SLURM_JOB_ID"

export LOCAL_DATASET="$LOCAL_BASE/dataset"
export SSD_DATASET="$SSD_MOUNT/dataset"

export LOCAL_WEIGHTS="$LOCAL_BASE/weights"
export SSD_WEIGHTS="$SSD_MOUNT/weights"

# --- Module loading ---
module purge
module load modtree/gpu
module load gcc/11.2.0
module load cuda/12.0.1

# --- Conda environment ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate MoE
