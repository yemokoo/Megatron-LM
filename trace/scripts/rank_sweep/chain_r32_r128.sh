#!/usr/bin/env bash
# Wait for the r16/r256 lanes (slora->ewc->seq_lora->lifelong_moe_attn each) to fully
# finish, then launch the same 4-model interleaved pattern for r32/r128 on the same
# GPU groups: lane A (0-3) = slora r32 -> ewc r128 -> seq_lora r32 -> lifelong r128,
# lane B (4-7) = slora r128 -> ewc r32 -> seq_lora r128 -> lifelong r32.
set -uo pipefail
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SWEEP=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/rank_sweep_20260916
say(){ printf '[CHAIN-32-128 %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$SWEEP/progress.log"; }
say "waiting for r16/r256 lanes to finish (lane_rest.sh pids)"
for p in $(pgrep -f "[l]ane_rest.sh"); do
  say "  waiting on pid $p ($(tr '\0' ' ' </proc/$p/cmdline 2>/dev/null | cut -c1-80))"
  while kill -0 $p 2>/dev/null; do sleep 60; done
done
say "r16/r256 lanes done; launching r32/r128"
GPUS=0,1,2,3 STEPS="slora:32 ewc:128 seq_lora:32 lifelong_moe_attn:128" PORT_BASE=29661 setsid nohup bash $D/lane_rest.sh >/dev/null 2>&1 &
GPUS=4,5,6,7 STEPS="slora:128 ewc:32 seq_lora:128 lifelong_moe_attn:32" PORT_BASE=29671 setsid nohup bash $D/lane_rest.sh >/dev/null 2>&1 &
say "launched: lane 0,1,2,3 = slora32->ewc128->seq32->lifelong128 ; lane 4,5,6,7 = slora128->ewc32->seq128->lifelong32"
