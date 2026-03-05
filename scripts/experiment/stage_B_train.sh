#!/bin/bash
# Stage B: Continue training FLAME-MoE-290M on Python code, starting from Stage A checkpoint.
#
# Usage:
#   sbatch --export=STAGE_A_JOB_ID=<job_id_from_stage_A> scripts/experiment/stage_B_train.sh
#
# This loads the final Stage A weights and continues training on Python code.
# All experts are unfrozen — this lets us measure how much B training disrupts
# the routing patterns learned during Stage A.

#SBATCH --job-name=continual-stage-B
#SBATCH --output=logs/%x/%j.log

#SBATCH --partition=gpu
#SBATCH --account=cis251382-gpu
#SBATCH --qos=gpu
#SBATCH --time=12:00:00

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=480G
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4

source scripts/config.sh

if [ -z "$STAGE_A_JOB_ID" ]; then
    echo "ERROR: STAGE_A_JOB_ID is not set."
    echo "Usage: sbatch --export=STAGE_A_JOB_ID=<job_id> $0"
    exit 1
fi

# FLAME-MoE-290M architecture (must match Stage A exactly)
export NUM_LAYERS=9
export HIDDEN_SIZE=1024
export FFN_HIDDEN_SIZE=5472
export MOE_FFN_HIDDEN_SIZE=704
export MOE_LAYER_FREQ="[0]*1+[1]*8"
export MICRO_BATCH_SIZE=4
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=8

# ~500 iters on Python code
export TRAIN_ITERS=500
export SAVE_INTERVAL=50
export EVAL_INTERVAL=50

export RDZV_BACKEND="c10d"
export RDZV_ENDPOINT="${RDZV_ENDPOINT:-$(hostname):8000}"

# Dataset: tokenized Python code
export TRAIN_DATASET="$LOCAL_DATASET/python-code/tokenized/EleutherAI/pythia-12b"

# Load Stage A weights, save Stage B weights separately
STAGE_A_WEIGHTS="$LOCAL_WEIGHTS/continual-stage-A/$STAGE_A_JOB_ID"
export TRAIN_WEIGHTS="$LOCAL_WEIGHTS/continual-stage-B/$SLURM_JOB_ID"

echo "Loading Stage A weights from: $STAGE_A_WEIGHTS"
echo "Saving Stage B weights to:    $TRAIN_WEIGHTS"

# Copy Stage A checkpoint to SSD (as starting point)
mkdir -p $SSD_WEIGHTS
rsync -a "$STAGE_A_WEIGHTS/" "$SSD_WEIGHTS/"

# Copy dataset to local SSD
mkdir -p $SSD_DATASET
rsync -a --info=progress2 "$TRAIN_DATASET/" "$SSD_DATASET/"

# Run training (continues from the loaded checkpoint)
source configs/model/flame-moe.sh
source configs/train/flame-moe.sh

DATA_ARGS=(
    --seq-length 2048
    --data-path $(find $SSD_DATASET -type f -name '*.bin' -exec sh -c 'printf "1.0 %s " "${1%.bin}"' _ {} \; | sed 's/ $//')
    --split 95,5,0
)

SAVE_ARGS=(
    --log-interval 10
    --log-throughput
    --save $SSD_WEIGHTS
    --save-interval $SAVE_INTERVAL
    --load $SSD_WEIGHTS
    --eval-interval $EVAL_INTERVAL
    --tensorboard-dir $SSD_WEIGHTS
    # Reset iteration counter so B starts from 0 relative to itself
    --no-load-optim
    --no-load-rng
    --finetune
)

mkdir -p $SSD_WEIGHTS $TRAIN_WEIGHTS

cd Megatron-LM && torchrun \
    --nnodes $SLURM_NNODES \
    --node_rank $SLURM_NODEID \
    --nproc_per_node $SLURM_GPUS_ON_NODE \
    --rdzv-id $SLURM_JOB_ID \
    --rdzv-backend $RDZV_BACKEND \
    --rdzv-endpoint $RDZV_ENDPOINT \
    pretrain_gpt.py \
    "${MODEL_ARGS[@]}" "${INFRA_ARGS[@]}" "${TRAIN_ARGS[@]}" \
    "${DATA_ARGS[@]}" "${SAVE_ARGS[@]}" &
TORCHRUN_PID=$!

(
    while kill -0 $TORCHRUN_PID 2>/dev/null; do
        rsync -a "$SSD_WEIGHTS/" "$TRAIN_WEIGHTS/"
        sleep 15m
    done
) &

wait $TORCHRUN_PID
rsync -a "$SSD_WEIGHTS/" "$TRAIN_WEIGHTS/"

echo "Stage B complete. Checkpoint at: $TRAIN_WEIGHTS"
