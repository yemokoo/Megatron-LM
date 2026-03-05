#!/bin/bash
# Download the pretrained weights and dataset.

mkdir -p $SSD_WEIGHTS
echo "[$(hostname)] Copying weights from $TRAIN_WEIGHTS ..."
rsync -a "$TRAIN_WEIGHTS/" "$SSD_WEIGHTS/"

mkdir -p $SSD_DATASET
echo "[$(hostname)] Copying dataset from $TRAIN_DATASET ..."
rsync -a "$TRAIN_DATASET/" "$SSD_DATASET/"
