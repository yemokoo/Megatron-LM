#!/bin/bash
# Compute the expert coactivation for FLAME-MoE-290M-1.3B

#SBATCH --job-name=expert-coactivation-290m
#SBATCH --output=logs/%x/%j.log

#SBATCH --partition=gpu
#SBATCH --account=cis251382-gpu
#SBATCH --qos=gpu
#SBATCH --time=4-00:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=480G
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4

source scripts/config.sh

# load the captured actives via google cloud
export TRAIN_JOB_ID=31066
export TRAIN_JOB_NAME=flame-moe-290m
bash scripts/empirical_analysis/modules/expert_coactivation_step1.sh

# process each layer inside each checkpoint for FLAME-MoE-290M-1.3B
find $SSD_MOUNT/actives -mindepth 2 -maxdepth 2 -type d | while read -r actives_path; do
    results_path=results/expert-coactivation/flame-moe-290m/$(basename $(dirname $actives_path))/$(basename $actives_path).pkl
    python3 scripts/empirical_analysis/modules/expert_coactivation_step2.py --actives-path $actives_path --results-path $results_path
done
