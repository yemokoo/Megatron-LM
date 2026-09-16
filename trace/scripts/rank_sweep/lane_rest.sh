#!/usr/bin/env bash
# Continue a rank-sweep lane after its running first step: wait for PID (the running
# run_slora_rank.sh) to exit, then run STEPS="method:rank method:rank ..." on GPUS.
#   usage: WAIT_PID=123 GPUS=0,1,2,3 PORT_BASE=29621 STEPS="ewc:256 seq_lora:16 lifelong_moe_attn:256" bash lane_rest.sh
set -uo pipefail
GPUS=${GPUS:?}; STEPS=${STEPS:?}; PORT_BASE=${PORT_BASE:-29641}; WAIT_PID=${WAIT_PID:-}
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SWEEP=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/rank_sweep_20260916
say(){ printf '[LANE %s %s] %s\n' "$GPUS" "$(date '+%F %T')" "$*" | tee -a "$SWEEP/progress.log"; }
if [ -n "$WAIT_PID" ]; then say "waiting for pid $WAIT_PID, then: $STEPS"; while kill -0 $WAIT_PID 2>/dev/null; do sleep 30; done; fi
i=0
for st in $STEPS; do
  m=${st%%:*}; r=${st##*:}; i=$((i+1))
  micro=$([ "$r" -ge 256 ] && echo 8 || echo 16)
  export TAB1_LIFELONG_KD=1.5 TAB1_LIFELONG_SHARED=attn TAB1_LIFELONG_TRAIN_SHARED=1   # Table-1 Lifelong-MoE values
  say "step $i: $m r$r"
  if [ "$m" = slora ]; then RANK=$r GPUS=$GPUS PORT=$((PORT_BASE+i)) MICRO=$micro bash $D/run_slora_rank.sh || say "$m r$r failed; continuing"
  elif [ "$m" = ewc_lamfix ]; then METHOD=ewc RANK=$r GPUS=$GPUS PORT=$((PORT_BASE+i)) MICRO=$micro NAME_SUFFIX=_lamfix bash $D/run_tab1_rank.sh || say "$m r$r failed; continuing"
  else METHOD=$m RANK=$r GPUS=$GPUS PORT=$((PORT_BASE+i)) MICRO=$micro bash $D/run_tab1_rank.sh || say "$m r$r failed; continuing"; fi
done
say "lane done"
