#!/bin/bash
# Download the desired dataset.
# Usage: scripts/dataset/download.sh

# Author: Hao Kang
# Date: March 9, 2025

#SBATCH --job-name=download
#SBATCH --output=logs/%x-%j.log
#SBATCH --partition=gpu
#SBATCH --account=cis251382-gpu
#SBATCH --qos=gpu
#SBATCH --time=2-00:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

# ============================================================
# TODO: Replace this section with your dataset download logic.
#
# Set DEST_DIR to where you want raw data saved on Anvil scratch.
# Example for DCLM from S3 (requires AWS CLI configured):
#
#   DEST_DIR="/anvil/scratch/x-dlee18/LLM-continual-learning/dataset/raw"
#   mkdir -p $DEST_DIR
#   prefix=s3://commoncrawl/contrib/datacomp/DCLM-baseline/global-shard_03_of_10/local-shard_1_of_10/
#   aws s3 sync $prefix $DEST_DIR/ --no-sign-request
#
# For a HuggingFace dataset, use:
#   python -c "from datasets import load_dataset; ds = load_dataset('...'); ds.save_to_disk('$DEST_DIR')"
# ============================================================

echo "TODO: configure dataset download in scripts/dataset/download.sh"
exit 1
