#!/usr/bin/env bash
set -uo pipefail
TRACE=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace
ROOT=/data2/seonghyeonnoh/paper/bos_token/cstance_1phase_kd_rep
D=/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace/C-STANCE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
job(){ local lr=$1 g=$2; local o=$ROOT/lr$lr; mkdir -p $o
  CUDA_VISIBLE_DEVICES=$g $TRACE/.venv-runtime/bin/python $TRACE/scripts/bos_token/train_bos_token.py \
    --checkpoint /data2/seonghyeonnoh/paper/ablation/1phase_kd_rep/0 \
    --train-json $D/train.json --eval-json $D/eval.json --eval-num 200 \
    --lr $lr --epochs 1 --micro-batch 4 --grad-accum 8 --eval-every 20 --out-dir $o > $o/train.log 2>&1
  echo "[$(date '+%F %T')] lr=$lr exit=$?" >> $ROOT/progress.log; }
job 1e-3 0 & sleep 10; job 1e-2 1 & sleep 10; job 3e-4 2 &
wait; echo "[$(date '+%F %T')] ALL DONE" >> $ROOT/progress.log
