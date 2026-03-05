#!/bin/bash
# Upload the router traces and the associated samples.

LOCAL_BASE="/anvil/scratch/x-dlee18/LLM-continual-learning"
rsync -a "$SSD_MOUNT/actives/" "$LOCAL_BASE/actives/$TRAIN_JOB_NAME/$TRAIN_JOB_ID/"
rsync -a "$SSD_MOUNT/samples/" "$LOCAL_BASE/samples/$TRAIN_JOB_NAME/$TRAIN_JOB_ID/"
