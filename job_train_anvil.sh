#!/bin/bash
# FLAME-MoE training job for Anvil (Purdue)
# Submit with: sbatch job_train_anvil.sh

#SBATCH --job-name=flame-moe
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

#SBATCH --partition=gpu
#SBATCH --account=cis251382-gpu
#SBATCH --qos=gpu
#SBATCH --time=4-00:00:00

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=480G
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4

# ============================================================
# DATASET  — set the path to your tokenized data (.bin/.idx)
# ============================================================
# TODO: set this to your tokenized dataset directory
export TRAIN_DATASET=""   # e.g. "/anvil/scratch/x-dlee18/LLM-continual-learning/dataset/my-data/tokenized/EleutherAI/pythia-12b"

# ============================================================
# MODEL CONFIG — edit these to change architecture/size
#   For a quick test, the 38M defaults below are fine.
#   Swap in values from scripts/release/flame-moe-*.sh for
#   larger models.
# ============================================================
export NUM_LAYERS=9                     # total transformer layers
export HIDDEN_SIZE=256                  # d_model
export FFN_HIDDEN_SIZE=1368             # dense FFN dim (layer 0 only)
export MOE_FFN_HIDDEN_SIZE=176          # per-expert FFN dim
export MOE_LAYER_FREQ="[0]*1+[1]*8"    # layer 0=dense, rest=MoE

# ============================================================
# TRAINING CONFIG
# ============================================================
export MICRO_BATCH_SIZE=32
export TRAIN_ITERS=2121
export SAVE_INTERVAL=212
export EVAL_INTERVAL=212

# Parallelism — with 4 GPUs/node on Anvil:
#   EXPERT_MODEL_PARALLEL_SIZE=4  → 1 node (4 GPUs)
#   EXPERT_MODEL_PARALLEL_SIZE=8  → 2+ nodes (8+ GPUs)
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=8    # adjust to match --nodes × 4 if needed

# ============================================================
# PATHS  (derived from config.sh — usually no need to change)
# ============================================================
LOCAL_BASE="/anvil/scratch/x-dlee18/LLM-continual-learning"
SSD_MOUNT="/tmp/slurm-$SLURM_JOB_ID"
SSD_DATASET="$SSD_MOUNT/dataset"
SSD_WEIGHTS="$SSD_MOUNT/weights"
TRAIN_WEIGHTS="$LOCAL_BASE/weights/$SLURM_JOB_NAME/$SLURM_JOB_ID"

# ============================================================
# ENVIRONMENT SETUP
# ============================================================
module purge
module load modtree/gpu
module load gcc/11.2.0
module load cuda/12.0.1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate MoE

# ============================================================
# VALIDATION
# ============================================================
if [[ -z "$TRAIN_DATASET" ]]; then
    echo "ERROR: TRAIN_DATASET is not set. Edit job_train_anvil.sh and set it."
    exit 1
fi

echo "========================================"
echo "Job ID:         $SLURM_JOB_ID"
echo "Nodes:          $SLURM_NNODES"
echo "Node:           $SLURMD_NODENAME"
echo "Start time:     $(date)"
echo "Dataset:        $TRAIN_DATASET"
echo "Weights out:    $TRAIN_WEIGHTS"
echo "Model:          ${NUM_LAYERS}L h${HIDDEN_SIZE} experts=${MOE_FFN_HIDDEN_SIZE}"
echo "========================================"

mkdir -p logs
mkdir -p "$TRAIN_WEIGHTS"

# ============================================================
# STEP 1 — copy dataset to local node storage
# ============================================================
srun -W 0 bash -c "
    mkdir -p $SSD_DATASET
    echo \"[\$(hostname)] Copying dataset ...\"
    rsync -a '$TRAIN_DATASET/' '$SSD_DATASET/'
    echo \"[\$(hostname)] Dataset ready.\"
"

# ============================================================
# STEP 2 — launch training
# ============================================================
export RDZV_BACKEND="c10d"
export RDZV_ENDPOINT="${RDZV_ENDPOINT:-$(hostname):8000}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-moe}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$SLURM_JOB_NAME}"
export WANDB_NAME="${WANDB_NAME:-$SLURM_JOB_ID}"

srun -W 0 bash -c "
    export OMP_NUM_THREADS=16
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
    export TORCH_NCCL_TRACE_BUFFER_SIZE=8
    export TORCH_NCCL_DUMP_ON_TIMEOUT=1

    source $LOCAL_BASE/configs/model/flame-moe.sh
    source $LOCAL_BASE/configs/train/flame-moe.sh

    TORCH_ARGS=(
        --nnodes $SLURM_NNODES
        --node_rank \$SLURM_NODEID
        --nproc_per_node \$SLURM_GPUS_ON_NODE
        --rdzv-id $SLURM_JOB_ID
        --rdzv-backend $RDZV_BACKEND
        --rdzv-endpoint $RDZV_ENDPOINT
    )

    DATA_ARGS=(
        --seq-length 2048
        --data-path \$(find $SSD_DATASET -type f -name '*.bin' -exec sh -c 'printf \"1.0 %s \" \"\${1%.bin}\"' _ {} \; | sed 's/ \$//')
        --split 90,5,5
    )

    SAVE_ARGS=(
        --log-interval 5
        --log-throughput
        --save $SSD_WEIGHTS
        --save-interval $SAVE_INTERVAL
        --load $SSD_WEIGHTS
        --eval-interval $EVAL_INTERVAL
        --wandb-save-dir $SSD_WEIGHTS
        --wandb-project $WANDB_PROJECT
        --wandb-exp-name $SLURM_JOB_ID
        --tensorboard-dir $SSD_WEIGHTS
    )

    mkdir -p $SSD_WEIGHTS
    cd $LOCAL_BASE/Megatron-LM && torchrun \"\${TORCH_ARGS[@]}\" pretrain_gpt.py \
        \"\${MODEL_ARGS[@]}\" \"\${INFRA_ARGS[@]}\" \"\${TRAIN_ARGS[@]}\" \"\${DATA_ARGS[@]}\" \"\${SAVE_ARGS[@]}\"
" &
SRUN_PID=$!

# Sync weights to Anvil scratch every 15 minutes while training runs
(
    while kill -0 $SRUN_PID 2>/dev/null; do
        rsync -a "$SSD_WEIGHTS/" "$TRAIN_WEIGHTS/"
        sleep 15m
    done
) &

wait $SRUN_PID
rsync -a "$SSD_WEIGHTS/" "$TRAIN_WEIGHTS/"

echo "========================================"
echo "Training complete: $(date)"
echo "Weights saved to: $TRAIN_WEIGHTS"
echo "========================================"
