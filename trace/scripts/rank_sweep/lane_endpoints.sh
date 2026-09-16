#!/usr/bin/env bash
# One lane of the TRACE rank-endpoint sweep: S-LoRA -> EWC -> Seq-LoRA at one rank,
# each step train then eval, on a fixed 4-GPU set.
#   usage: RANK=16 GPUS=0,1,2,3 PORT_BASE=29621 bash lane_endpoints.sh
set -uo pipefail
RANK=${RANK:?}; GPUS=${GPUS:?}; PORT_BASE=${PORT_BASE:-29621}
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SWEEP=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/rank_sweep_20260916
say(){ printf '[LANE r%s %s] %s\n' "$RANK" "$(date '+%F %T')" "$*" | tee -a "$SWEEP/progress.log"; }
say "lane start on $GPUS: slora -> ewc -> seq_lora"
MICRO_SLORA=$([ "$RANK" -ge 256 ] && echo 8 || echo 16)
RANK=$RANK GPUS=$GPUS PORT=$PORT_BASE MICRO=$MICRO_SLORA bash $D/run_slora_rank.sh || say "slora r$RANK failed; continuing"
METHOD=ewc      RANK=$RANK GPUS=$GPUS PORT=$((PORT_BASE+1)) MICRO=$MICRO_SLORA bash $D/run_tab1_rank.sh || say "ewc r$RANK failed; continuing"
METHOD=seq_lora RANK=$RANK GPUS=$GPUS PORT=$((PORT_BASE+2)) MICRO=$MICRO_SLORA bash $D/run_tab1_rank.sh || say "seq_lora r$RANK failed"
say "lane done"
