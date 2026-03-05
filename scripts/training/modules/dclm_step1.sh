#!/bin/bash
# Copy tokenized dataset from Anvil scratch to local node storage.

mkdir -p $SSD_DATASET
echo "[$(hostname)] Copying dataset from $TRAIN_DATASET to $SSD_DATASET ..."
rsync -a --info=progress2 "$TRAIN_DATASET/" "$SSD_DATASET/"
echo "[$(hostname)] Dataset copy complete."
