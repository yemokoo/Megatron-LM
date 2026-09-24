#!/usr/bin/env bash
# Real-data replay size ablation: the residual split recipe run at 10/20/1/15/5 % of each
# task's train set, one ratio after another on the same GPUs.
#
# Only the replay *diversity* changes between cells:
#   * records per source   = PERSIST (1%=50, 5%=250, 10%=500, 15%=750, 20%=1000), a nested
#     prefix of the seed-2025 20% pool built by build_real_replay_subsets.py, so the smaller
#     ratios are subsets of the larger ones;
#   * pool cap             = sources x PERSIST (8 sources: 7 past tasks + the backbone-BoS
#     pseudo task), i.e. every stored record is reachable;
#   * backbone BoS records = PERSIST as well, a prefix of backbone_bos_2000.jsonl, so the
#     pseudo task scales with the ratio;
#   * replay forwards      = EXPOSURE_CAP x epochs, FIXED at 5000 for every ratio, so replay
#     compute, primary compute and the optimizer-step count are identical across cells and a
#     record is simply repeated fewer times as the pool grows.
# Everything else (seed 2025, 1-phase split-gradient residual recipe, header guard, no KD-init,
# no generation) is held constant.
#
#   RUN_ROOT=/data2/.../residual_real_20260923 bash scripts/residual/run_real_ratio_chain.sh
#   RATIOS="10 20 1 15 5"   order to run (default)
#   SUBSETS=<dir>           replay_subsets root (pct<r>/fixed_replay_memory)
#   BOS_DIR=<dir>           backbone_bos_<n>.jsonl files
#   SEED_CKPT=<model dir>   optional shared round-0 checkpoint; omitted = each ratio trains its
#                           own round 0 (correct here: round 0's BoS pool size is ratio-specific)
#   GPUS, TRACE_ROOT, TRACE_PYTHON, SLORA_LLAMA31_PATH, TRACE_DATA_ROOT, TOKEN_CACHE pass through.
set -uo pipefail
TRACE=${TRACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ROOT=${RUN_ROOT:-/data2/seonghyeonnoh/paper/residual_real_20260923}
SUBSETS=${SUBSETS:-$ROOT/replay_subsets}
BOS_DIR=${BOS_DIR:-$ROOT/backbone_bos}
RATIOS=${RATIOS:-"10 20 1 15 5"}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
LOG=$ROOT/ratio_chain.log
mkdir -p "$ROOT"
say(){ printf '[RATIO %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

declare -A N=( [1]=50 [5]=250 [10]=500 [15]=750 [20]=1000 )

for r in $RATIOS; do
  n=${N[$r]:-}
  [ -n "$n" ] || { say "unknown ratio $r (use 1/5/10/15/20)"; exit 2; }
  run=$ROOT/pct$r
  bos=$BOS_DIR/backbone_bos_$n.jsonl
  mem=$SUBSETS/pct$r/fixed_replay_memory
  [ -f "$bos" ] || { say "missing backbone BoS file $bos"; exit 2; }
  [ -d "$mem" ] || { say "missing replay subset $mem"; exit 2; }
  if [ -f "$run/model/sparse15_summary.json" ]; then say "pct$r already evaluated, skip"; continue; fi

  mkdir -p "$run/model"
  # the trainer validates these index files against unique_samples == PERSIST, so the copy and
  # --v2_new_persistent_samples_per_task must agree; copying is idempotent across restarts
  cp -a "$mem/." "$run/model/fixed_replay_memory/" 2>/dev/null || {
    mkdir -p "$run/model/fixed_replay_memory"; cp -a "$mem/." "$run/model/fixed_replay_memory/"; }
  say "pct$r START  records/source=$n  pool_cap=$((8 * n))  exposure=5000xepochs  bos=$(basename "$bos")"

  REPLAY_SOURCE=real RUN_DIR="$run" GPUS="$GPUS" \
    PERSIST_PER_TASK="$n" POOL_CAP=$((8 * n)) EXPOSURE_CAP=5000 \
    RESIDUAL_BOS_JSONL="$bos" METHOD_NAME="residual_real_pct$r" \
    ${SEED_CKPT:+SEED_CKPT="$SEED_CKPT"} \
    bash "$TRACE/scripts/residual/run_residual_chain.sh" >> "$run/ratio_driver.log" 2>&1
  rc=$?
  if [ -f "$run/model/sparse15_summary.json" ]; then
    say "pct$r DONE  $(python3 - "$run/model/sparse15_summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print("AA %.2f  F %.2f" % (s["final_average"], -s["BWT"]))
PY
)"
  else
    say "pct$r FAILED rc=$rc (see $run/logs, $run/ratio_driver.log)"
    exit 1
  fi
done
say "EVENT: RATIO_CHAIN_DONE  ratios=$RATIOS"
