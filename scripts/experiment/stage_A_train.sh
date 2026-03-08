#!/bin/bash
# Stage A: Pretrain FLAME-MoE-290M on Wikipedia.
# After this job, run stage_B_train.sh using STAGE_A_JOB_ID=<this job id>.

#SBATCH --job-name=continual-stage-A
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

# FLAME-MoE-290M architecture
export NUM_LAYERS=9
export HIDDEN_SIZE=1024
export FFN_HIDDEN_SIZE=5472
export MOE_FFN_HIDDEN_SIZE=704
export MOE_LAYER_FREQ="[0]*1+[1]*8"
export NUM_EXPERTS=4
export MOE_ROUTER_TOPK=2
export MICRO_BATCH_SIZE=4
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=1

# ~500 iters × 1024 global batch × 2048 tokens = ~1B tokens from Wikipedia
export TRAIN_ITERS=500
export SAVE_INTERVAL=100
export EVAL_INTERVAL=100

export RDZV_BACKEND="c10d"
export RDZV_ENDPOINT="${RDZV_ENDPOINT:-$(hostname):8000}"

# Dataset: tokenized Wikipedia
export TRAIN_DATASET="$LOCAL_DATASET/wikipedia/tokenized/EleutherAI/pythia-12b"

# Weights saved under job name/id for stage B to pick up
export TRAIN_WEIGHTS="$LOCAL_WEIGHTS/continual-stage-A/$SLURM_JOB_ID"

# Copy dataset to local SSD
mkdir -p $SSD_DATASET
rsync -a --info=progress2 "$TRAIN_DATASET/" "$SSD_DATASET/"

# Run training
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

# Sync weights to scratch every 15 minutes
(
    while kill -0 $TORCHRUN_PID 2>/dev/null; do
        rsync -a "$SSD_WEIGHTS/" "$TRAIN_WEIGHTS/"
        sleep 15m
    done
) &

wait $TORCHRUN_PID
rsync -a "$SSD_WEIGHTS/" "$TRAIN_WEIGHTS/"

echo "Stage A complete. Checkpoint at: $TRAIN_WEIGHTS"
echo "To run Stage B: sbatch --export=STAGE_A_JOB_ID=$SLURM_JOB_ID scripts/experiment/stage_B_train.sh"
