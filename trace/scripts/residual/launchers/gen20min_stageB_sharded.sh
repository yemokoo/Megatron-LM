#!/usr/bin/env bash
# Take over 20Minuten stage B (answer pass) from gen_20min_oursgen.sh and split it over GPUs.
# Waits for stage A to finish, stops the single-GPU driver before its stage B loads,
# shards stageA/text.jsonl, runs answer_pass_v3_fix.py per shard, concatenates.
set -uo pipefail
R=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/residual_router_20260915
TRACE=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace
PY=$TRACE/.venv-runtime/bin/python
CKPT=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/selfgen_cl_frozen_20260901/model/7
OWN=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/tab3_20260913/.gpu_owner
D=$R/data/oursgen_20Minuten
GPUS=(${GPUS:-6 7})
DRIVER_PGID=${DRIVER_PGID:?}
export OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false

until [ -f $D/stageA/stats.json ]; do sleep 5; done
sleep 8; kill -TERM -$DRIVER_PGID 2>/dev/null; sleep 3
for g in "${GPUS[@]}"; do echo $$ > $OWN/$g; done
N=${#GPUS[@]}
for k in $(seq 0 $((N-1))); do
  mkdir -p $D/stageA_shard$k
  awk -v n=$N -v k=$k 'NR % n == k' $D/stageA/text.jsonl > $D/stageA_shard$k/text.jsonl
  cp $D/stageA/stats.json $D/stageA_shard$k/ 2>/dev/null
done
pids=()
for k in $(seq 0 $((N-1))); do
  ( cd $TRACE && CUDA_VISIBLE_DEVICES=${GPUS[$k]} $PY scripts/analysis/answer_pass_v3_fix.py --checkpoint $CKPT \
      --stage-a $D/stageA_shard$k --out $D/records.shard$k.jsonl --prompt-cue $'\n\nSimplification:' \
      --max-answer-tokens 512 --batch 64 > $R/logs/gen20min_stageB_shard$k.log 2>&1 ) &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
cat $D/records.shard*.jsonl > $D/records.jsonl
echo "records: $(wc -l < $D/records.jsonl) (sharded x$N)" >> $R/logs/gen20min.log
echo "exit=$rc" >> $R/logs/gen20min.log
for g in "${GPUS[@]}"; do [ "$(cat $OWN/$g 2>/dev/null)" = "$$" ] && rm -f $OWN/$g; done
