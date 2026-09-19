#!/usr/bin/env bash
# Serial queue of the four HP-sensitivity cells, each on all 8 GPUs / global
# batch 64 (the final model's condition).  Resumable: a cell whose round 7
# checkpoint exists is skipped, and inside a cell each round is skipped if its
# checkpoint exists.
#
#   bash run_host_hp.sh                 # all four
#   CELLS="e4_r16 e8_r8" bash run_host_hp.sh
set -uo pipefail
H=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace/scripts/hp_sensitivity
ROOT=${ROOT:-/data2/seonghyeonnoh/paper/hp_sens}
GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}
PHASE=${PHASE:-1phase}
KD=${KD:-on}
DISPATCH=${DISPATCH:-loop}
KD_FRACTION=${KD_FRACTION:-0.5}
CELLS=${CELLS:-"order_reverse e2_r32 e4_r16 e8_r8"}
mkdir -p "$ROOT"
say(){ printf '[HOST-HP %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$ROOT/progress.log"; }

run_cell() {   # name
  local mode=forward E=1 R=64 K=1
  case $1 in
    order_reverse) mode=reverse; E=1; R=64; K=1 ;;
    e2_r32)        E=2; R=32; K=2 ;;
    e4_r16)        E=4; R=16; K=4 ;;
    e8_r8)         E=8; R=8;  K=8 ;;
    *) say "unknown cell $1"; return 2 ;;
  esac
  env ORDER_MODE="$mode" EXPERTS_PER_TASK="$E" RANK="$R" TOPK="$K" NAME="$1" \
      ROOT="$ROOT" GPU_LIST="$GPU_LIST" PHASE="$PHASE" KD="$KD" DISPATCH="$DISPATCH" KD_FRACTION="$KD_FRACTION" \
      bash "$H/run_hp_cell.sh"
}

say "queue start: $CELLS (gpus $GPU_LIST, phase=$PHASE kd=$KD kd_frac=$KD_FRACTION dispatch=$DISPATCH)"
rc=0
for c in $CELLS; do
  if [ -f "$ROOT/$c/model/7/lora_moe_meta.json" ]; then
    say "$c already complete -> $ROOT/$c"; continue
  fi
  run_cell "$c" || { rc=1; say "$c FAILED, continuing to next cell"; }
done
say "queue done rc=$rc"
exit $rc
