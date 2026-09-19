#!/usr/bin/env bash
# sparse-15 evaluation for the HP-sensitivity cells, serial, each on all 8 GPUs.
# Waits for run_host_hp.sh to exit first (training and eval must not share GPUs).
#
# The reversed-order cell is scored with --task-order taken from its cell.json,
# so round r's diagonal cell is that order's r-th task.  2-phase cells score
# their diagonal from the pre-router-retune checkpoint.
#
#   bash eval_hp.sh                          # wait for the training queue
#   SKIP_WAIT=1 CELLS="e4_r16" bash eval_hp.sh
set -uo pipefail
TRACE=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace
PY=$TRACE/.venv-runtime/bin/python
ROOT=${ROOT:-/data2/seonghyeonnoh/paper/hp_sens}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
CELLS=${CELLS:-"order_reverse e2_r32 e4_r16 e8_r8"}
say(){ printf '[EVAL-HP %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$ROOT/progress.log"; }

wait_for_training_queue() {
  pgrep -f "[r]un_host_hp.sh" >/dev/null || return 0
  say "waiting for run_host_hp.sh to finish before evaluating"
  while pgrep -f "[r]un_host_hp.sh" >/dev/null; do sleep 60; done
  say "training queue done"
}

eval_cell() {
  local cell=$1 run=$ROOT/$1/model spec=$ROOT/$1/cell.json
  [ -f "$run/7/lora_moe_meta.json" ] || { say "$cell: SKIP, no round 7 checkpoint"; return 1; }
  [ -f "$run/sparse15_summary.json" ] && { say "$cell: already scored"; return 0; }
  local order phase suffix=""
  order=$($PY -c "import json;print(json.load(open('$spec'))['task_order'])")
  phase=$($PY -c "import json;print(json.load(open('$spec'))['phase'])")
  [ "$phase" = 2phase ] && suffix="_prephase2"
  say "$cell: eval start order=$order diagonal_suffix='${suffix:-none}'"
  ( cd "$TRACE" && $PY scripts/run_ours_sparse15_optimized.py \
      --run-dir "$run" --method "$cell" --gpus "$GPUS" --task-order "$order" \
      ${suffix:+--diagonal-checkpoint-suffix "$suffix"} ) > "$run/eval.log" 2>&1
  local rc=$?
  [ -f "$run/sparse15_summary.json" ] || { say "$cell: EVAL FAILED rc=$rc ($run/eval.log)"; return 1; }
  say "$cell: $($PY - "$run/sparse15_summary.json" <<'PYEOF'
import json,sys
d=json.load(open(sys.argv[1]))
print(f"AA {d.get('final_average', float('nan')):.2f} F {-d.get('BWT', float('nan')):.2f}")
PYEOF
)"
}

say "queue: $CELLS"
[ "${SKIP_WAIT:-0}" = 1 ] || wait_for_training_queue
rc=0
for c in $CELLS; do eval_cell "$c" || { rc=1; say "$c failed/skipped, continuing"; }; done
say "eval queue done rc=$rc"
exit $rc
