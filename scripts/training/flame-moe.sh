#!/bin/bash

#SBATCH --job-name=flame-moe
#SBATCH --output=logs/%x/%j.log

#SBATCH --partition=gpu
#SBATCH --account=cis251382-gpu
#SBATCH --qos=gpu
#SBATCH --time=4-00:00:00

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=480G
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4

source scripts/config.sh

export RDZV_BACKEND="c10d"
export RDZV_ENDPOINT="${RDZV_ENDPOINT:-$(hostname):8000}"
export WANDB_ENTITY="${WANDB_ENTITY:-$USER}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-moe}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$SLURM_JOB_NAME}"
export WANDB_NAME="${WANDB_NAME:-$SLURM_JOB_ID}"
export TRAIN_DATASET="${TRAIN_DATASET:-$LOCAL_DATASET/dclm-138b/tokenized/EleutherAI/pythia-12b}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$LOCAL_WEIGHTS/$SLURM_JOB_NAME/$SLURM_JOB_ID}"

srun -W 0 scripts/training/modules/flame-moe_step1.sh
srun -W 0 scripts/training/modules/flame-moe_step2.sh
