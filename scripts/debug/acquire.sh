#!/bin/bash

#SBATCH --job-name=acquire
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

srun -W 0 scripts/debug/modules/acquire_step1.sh
srun -W 0 scripts/debug/modules/acquire_step2.sh
