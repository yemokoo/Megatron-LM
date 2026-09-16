#!/usr/bin/env bash
# One arm: wait for data -> router-only tuning (+/- residual expert) -> full MMLU.
#   usage: ARM=<rep_res|rep_ctl|gen_res|gen_ctl> GPU=<g> bash run_arm3.sh
set -uo pipefail
ARM=${ARM:?}; GPU=${GPU:?}
R=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/residual_router_20260915
T=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace
TRACE=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace
PY=$TRACE/.venv-runtime/bin/python
W=/data2/seonghyeonnoh/LLM-continual-learning-runtime/lmeval
OWN=$T/tab3_20260913/.gpu_owner
BB=$R/data/backbone_bos/records.jsonl
G20=$R/data/oursgen_20Minuten/records.jsonl
say(){ printf '[RES %s] %s\n' "$(date '+%F %T')" "$*" >> $R/progress.log; }
echo $$ > $OWN/$GPU

REP_SRC=$T/v3_replay1to1/v3_new_replay1to1_st_top1
GEN_SRC=$T/selfgen_cl_frozen_20260901/model
case $ARM in
  rep_*) SRC=$REP_SRC/7; REPLAY="real:$REP_SRC/fixed_replay_memory"; NEED="$BB" ;;
  gen_*) SRC=$GEN_SRC/7
         GR=$T/selfgen_cl_frozen_20260901/gen/round_7
         REPLAY="gen:C-STANCE=$GR/C-STANCE/records.jsonl,FOMC=$GR/FOMC/records.jsonl,MeetingBank=$GR/MeetingBank/records.jsonl,Py150=$GR/Py150/records.jsonl,ScienceQA=$GR/ScienceQA/records.jsonl,NumGLUE-cm=$GR/NumGLUE-cm/records.jsonl,NumGLUE-ds=$GR/NumGLUE-ds/records.jsonl,20Minuten=$G20"
         NEED="$BB $G20" ;;
esac
case $ARM in *_res|*_resonly) NRES=1 ;; *_ctl) NRES=0 ;; esac

for f in $NEED; do
  until [ -f $R/logs/$(basename $(dirname $f)).done ] || { [ "$f" = "$BB" ] && grep -q "^exit=0" $R/logs/gen_backbone_bos.log 2>/dev/null; } \
        || { [ "$f" = "$G20" ] && grep -q "^exit=0" $R/logs/gen20min.log 2>/dev/null; }; do
    if grep -q "^exit=[1-9]" $R/logs/gen_backbone_bos.log $R/logs/gen20min.log 2>/dev/null; then say "$ARM: data generation failed"; exit 1; fi
    sleep 30
  done
done

OUT=$R/$ARM
if [ ! -f $OUT/router_state.pt ]; then
  say "$ARM train start gpu$GPU (n_residual=$NRES)"
  cd $TRACE && CUDA_VISIBLE_DEVICES=$GPU $PY scripts/residual/train_router_residual.py \
    --source $SRC --replay "$REPLAY" --backbone $BB --out $OUT --n_residual $NRES ${XARGS:-} \
    --replay_per_task 500 --backbone_n ${BB_N:-500} --batch 8 --accum 2 --lr 2e-4 --epochs 1 \
    > $R/logs/train_$ARM.log 2>&1
  [ -f $OUT/router_state.pt ] || { say "$ARM TRAIN FAILED (logs/train_$ARM.log)"; exit 1; }
  say "$ARM train done: $(tail -1 $OUT/train_log.jsonl)"
fi

export HF_HOME=/data2/seonghyeonnoh/huggingface HF_DATASETS_CACHE=/data2/seonghyeonnoh/LLM-continual-learning-data/lmeval_datasets
export HF_DATASETS_OFFLINE=1 HF_DATASETS_TRUST_REMOTE_CODE=1 TOKENIZERS_PARALLELISM=false
say "$ARM MMLU start"
CUDA_VISIBLE_DEVICES=$GPU $W/../lmeval-venv/bin/python $W/run_lmeval_trace.py --ckpt $OUT --out $OUT/mmlu \
  --tasks mmlu --batch_size 16 > $R/logs/mmlu_$ARM.log 2>&1
say "$ARM MMLU exit=$? $(grep -m1 '^|mmlu ' $R/logs/mmlu_$ARM.log | tr -s ' ')"
[ "$(cat $OWN/$GPU 2>/dev/null)" = "$$" ] && rm -f $OWN/$GPU
