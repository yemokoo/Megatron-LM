#!/usr/bin/env bash
# Rerun EWC r16/r256 with the rank-anchored lambda (400*64/rank -> 1600 / 100)
# instead of the fixed lambda=400 the first pass used, once r32/r128 are done
# freeing the GPUs. Output under *_lamfix so the original (fixed-lambda) rows
# stay on disk for comparison.
set -uo pipefail
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SWEEP=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/rank_sweep_20260916
CHAIN_PID=${CHAIN_R32_R128_PID:?export CHAIN_R32_R128_PID=<pid of chain_r32_r128.sh>}
say(){ printf '[EWC-LAMFIX %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$SWEEP/progress.log"; }
say "waiting for chain_r32_r128.sh (pid $CHAIN_PID) to launch its lanes"
while kill -0 $CHAIN_PID 2>/dev/null; do sleep 30; done
sleep 10  # let its two lane_rest.sh backgrounds actually spawn
say "r32/r128 lanes launched; waiting for them to finish (lane_rest.sh pids)"
for p in $(pgrep -f "[l]ane_rest.sh"); do
  say "  waiting on pid $p ($(tr '\0' ' ' </proc/$p/cmdline 2>/dev/null | cut -c1-80))"
  while kill -0 $p 2>/dev/null; do sleep 60; done
done
say "r32/r128 lanes done; launching EWC lambda-fix reruns (r16 lambda=1600, r256 lambda=100)"
METHOD=ewc RANK=16 GPUS=0,1,2,3 PORT=29681 NAME_SUFFIX=_lamfix bash $D/run_tab1_rank.sh &
p1=$!
METHOD=ewc RANK=256 GPUS=4,5,6,7 PORT=29691 NAME_SUFFIX=_lamfix bash $D/run_tab1_rank.sh &
p2=$!
wait $p1; wait $p2
say "EWC lambda-fix reruns done"
