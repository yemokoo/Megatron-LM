#!/bin/bash

LOCAL_BASE="/anvil/scratch/x-dlee18/LLM-continual-learning"
mkdir -p $SSD_MOUNT/actives
rsync -a "$LOCAL_BASE/actives/$TRAIN_JOB_NAME/$TRAIN_JOB_ID/" "$SSD_MOUNT/actives/"
