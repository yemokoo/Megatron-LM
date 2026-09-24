#!/usr/bin/env bash
# pct10 initialization ablation, sequential on the same GPUs:
#   rrzb:       original-Ours expansion -- new router row random, LoRA A
#               random, LoRA B zero; no expansion KD-init.
#   rrzb_kd20:  the same expansion, then output-logit KL init for 20% of
#               normal task optimizer updates. The pre-expansion function is
#               the teacher; only the new expert/row are updated.
# All data, replay, optimizer, main training and evaluation settings match
# residual_real_20260923/pct10. Creating this file does not launch anything.
set -uo pipefail

TRACE=${TRACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ROOT=${RUN_ROOT:-/data2/seonghyeonnoh/paper/residual_real_pct10_init_ablation_zerob_20260924}
SOURCE_ROOT=${SOURCE_ROOT:-/data2/seonghyeonnoh/paper/residual_real_20260923}
MEM=$SOURCE_ROOT/replay_subsets/pct10/fixed_replay_memory
BOS=$SOURCE_ROOT/backbone_bos/backbone_bos_500.jsonl
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
LOG=$ROOT/init_ablation_chain.log
mkdir -p "$ROOT"
say(){ printf '[INIT-ABLATION %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

[ -d "$MEM" ] || { say "missing pct10 replay memory: $MEM"; exit 2; }
[ -f "$BOS" ] || { say "missing pct10 backbone BoS file: $BOS"; exit 2; }

run_arm(){  # name kd fraction
  local name=$1 kd=$2 fraction=$3 run=$ROOT/$1
  if [ -f "$run/model/sparse15_summary.json" ]; then
    say "$name already evaluated, skip"
    return 0
  fi
  mkdir -p "$run/model/fixed_replay_memory"
  cp -a "$MEM/." "$run/model/fixed_replay_memory/"
  say "$name START init=random_router_zero_b kd_init=$kd fraction=$fraction"
  REPLAY_SOURCE=real RUN_DIR="$run" GPUS="$GPUS" \
    PERSIST_PER_TASK=500 POOL_CAP=4000 EXPOSURE_CAP=5000 \
    RESIDUAL_BOS_JSONL="$BOS" RESIDUAL_NEW_EXPERT_INIT=random_router_zero_b \
    KD_INIT="$kd" KD_INIT_STEP_FRACTION="$fraction" \
    bash "$TRACE/scripts/residual/run_residual_chain.sh" \
    >> "$run/init_ablation_driver.log" 2>&1
  local rc=$?
  if [ -f "$run/model/sparse15_summary.json" ]; then
    say "$name DONE"
  else
    say "$name FAILED rc=$rc (see $run/logs and $run/init_ablation_driver.log)"
    exit 1
  fi
}

run_arm rrzb off 1.0
run_arm rrzb_kd20 on 0.2
say "EVENT: INIT_ABLATION_CHAIN_DONE"
