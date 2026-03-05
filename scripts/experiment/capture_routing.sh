#!/bin/bash
# Capture router traces for a given checkpoint, evaluated on Wikipedia samples.
# Run this after both Stage A and Stage B to compare routing patterns.
#
# Usage:
#   # After Stage A:
#   sbatch --export=CAPTURE_JOB_ID=<stage_A_job_id>,CAPTURE_STAGE=A scripts/experiment/capture_routing.sh
#   # After Stage B:
#   sbatch --export=CAPTURE_JOB_ID=<stage_B_job_id>,CAPTURE_STAGE=B scripts/experiment/capture_routing.sh

#SBATCH --job-name=capture-routing
#SBATCH --output=logs/%x/%j.log

#SBATCH --partition=gpu
#SBATCH --account=cis251382-gpu
#SBATCH --qos=gpu
#SBATCH --time=2:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=480G
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4

source scripts/config.sh

if [ -z "$CAPTURE_JOB_ID" ] || [ -z "$CAPTURE_STAGE" ]; then
    echo "ERROR: Must set CAPTURE_JOB_ID and CAPTURE_STAGE (A or B)"
    echo "Usage: sbatch --export=CAPTURE_JOB_ID=<id>,CAPTURE_STAGE=A $0"
    exit 1
fi

# FLAME-MoE-290M architecture
export NUM_LAYERS=9
export HIDDEN_SIZE=1024
export FFN_HIDDEN_SIZE=5472
export MOE_FFN_HIDDEN_SIZE=704
export MOE_LAYER_FREQ="[0]*1+[1]*8"
export MICRO_BATCH_SIZE=8
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=4
export TRAIN_ITERS=500
export RDZV_BACKEND="c10d"
export RDZV_ENDPOINT="localhost:8000"

# Always evaluate on Wikipedia (A-domain) regardless of which stage checkpoint we use
# This lets us see how A-routing changes after B training
EVAL_DATASET="$LOCAL_DATASET/wikipedia/tokenized/EleutherAI/pythia-12b"

if [ "$CAPTURE_STAGE" = "A" ]; then
    WEIGHTS_DIR="$LOCAL_WEIGHTS/continual-stage-A/$CAPTURE_JOB_ID"
else
    WEIGHTS_DIR="$LOCAL_WEIGHTS/continual-stage-B/$CAPTURE_JOB_ID"
fi

OUTPUT_DIR="$LOCAL_BASE/actives/continual-stage-$CAPTURE_STAGE/$CAPTURE_JOB_ID"
mkdir -p "$OUTPUT_DIR"

echo "Capturing routing for Stage $CAPTURE_STAGE (job $CAPTURE_JOB_ID)"
echo "Weights: $WEIGHTS_DIR"
echo "Eval dataset: $EVAL_DATASET (Wikipedia = A-domain)"
echo "Output: $OUTPUT_DIR"

# Copy weights and dataset to SSD
mkdir -p $SSD_WEIGHTS $SSD_DATASET
rsync -a "$WEIGHTS_DIR/" "$SSD_WEIGHTS/"
rsync -a "$EVAL_DATASET/" "$SSD_DATASET/"

source configs/model/flame-moe.sh
source configs/train/flame-moe.sh

DATA_ARGS=(
    --seq-length 2048
    --data-path $(find $SSD_DATASET -type f -name '*.bin' -exec sh -c 'printf "1.0 %s " "${1%.bin}"' _ {} \; | sed 's/ $//')
    --split 95,0,5
)

# Capture routing for each saved checkpoint
for item in $(ls -d $SSD_WEIGHTS/iter_* | sort); do
    name=$(basename $item)
    step=$((10#${name#iter_}))
    echo "Capturing step $step ..."
    export EACT_SAVE="$SSD_MOUNT/actives/$step"
    export TIDS_SAVE="$SSD_MOUNT/samples"
    echo $step > $SSD_WEIGHTS/latest_checkpointed_iteration.txt

    SAVE_ARGS=(
        --test-mode
        --skip-train
        --load $SSD_WEIGHTS
        --eval-iters 25
    )

    TORCH_ARGS=(
        --nnodes 1
        --node_rank 0
        --nproc_per_node $SLURM_GPUS_ON_NODE
        --rdzv-id $SLURM_JOB_ID
        --rdzv-backend $RDZV_BACKEND
        --rdzv-endpoint $RDZV_ENDPOINT
    )

    cd Megatron-LM && torchrun "${TORCH_ARGS[@]}" pretrain_gpt.py \
        "${MODEL_ARGS[@]}" "${INFRA_ARGS[@]}" "${TRAIN_ARGS[@]}" \
        "${DATA_ARGS[@]}" "${SAVE_ARGS[@]}"
    cd ..
done

# Save results
rsync -a "$SSD_MOUNT/actives/" "$OUTPUT_DIR/"
rsync -a "$SSD_MOUNT/samples/" "$LOCAL_BASE/samples/continual-stage-$CAPTURE_STAGE/$CAPTURE_JOB_ID/"

echo "Routing capture complete. Results at: $OUTPUT_DIR"
