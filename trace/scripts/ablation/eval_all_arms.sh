#!/usr/bin/env bash
# Evaluate this host's three ablation arms, one after another, each using all
# 8 GPUs. Waits for ALL THREE arms to finish TRAINING first (i.e. for
# run_host_a.sh itself to exit) before evaluating any of them -- evaluating
# arm 1 while run_host_a.sh has already moved on to training arm 2 would put
# eval and training on the same 8 GPUs at once.
#
# MeetingBank/Py150/ScienceQA (the slow tasks) are auto-sharded across every
# configured GPU by run_ours_sparse15_optimized.py's work-queue scheduler
# (default --num-shards 0 = one shard per GPU) -- no extra flags needed for
# that, it already fills all 8 GPUs continuously as jobs finish.
#
# 2-phase arms score their diagonal (acquisition) cells from the
# pre-router-retune checkpoint (--diagonal-checkpoint-suffix _prephase2); the
# final row still comes from the continued model. 1-phase arms need no
# suffix.
#
#   bash eval_all_arms.sh                       # wait for training queue, then all three
#   SKIP_WAIT=1 bash eval_all_arms.sh            # training already done; eval now
#   ARMS="2phase_kd_rep" bash eval_all_arms.sh   # just one
set -uo pipefail
TRACE=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace
PY=$TRACE/.venv-runtime/bin/python
ROOT=${ROOT:-/data2/seonghyeonnoh/paper/ablation}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
export SLORA_LLAMA31_PATH=${SLORA_LLAMA31_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}
ARMS=${ARMS:-"1phase_kd_rep 2phase_kd_rep 2phase_kd_gen"}
mkdir -p "$ROOT"
say() { printf '[EVAL %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$ROOT/progress.log"; }

run_dir_for() {   # arm -> run-dir passed to the evaluator
  case $1 in
    *_gen) echo "$ROOT/$1/model" ;;
    *)     echo "$ROOT/$1" ;;
  esac
}

wait_for_training_queue() {   # block until run_host_a.sh (all 3 arms) has exited
  if ! pgrep -f "[r]un_host_a.sh" >/dev/null; then return 0; fi
  say "waiting for run_host_a.sh (all 3 arms' training) to finish before evaluating any of them"
  while pgrep -f "[r]un_host_a.sh" >/dev/null; do sleep 60; done
  say "run_host_a.sh has exited; training queue done"
}

check_round7() {   # arm run_dir -> 0 if round 7 checkpoint exists
  local arm=$1 run=$2
  if [ -f "$run/7/lora_moe_meta.json" ]; then return 0; fi
  say "$arm: SKIPPING eval -- no round 7 checkpoint at $run (training queue finished without completing this arm)"
  return 1
}

eval_arm() {   # arm
  local arm=$1
  local run; run=$(run_dir_for "$arm")
  check_round7 "$arm" "$run" || return 1
  if [ -f "$run/sparse15_summary.json" ]; then
    say "$arm: sparse15_summary.json already present, skipping eval"
    return 0
  fi
  local suffix=""
  [[ $arm == 2phase_* ]] && suffix="_prephase2"
  say "$arm: eval start (gpus=$GPUS diagonal_suffix='${suffix:-none}')"
  ( cd "$TRACE" && $PY scripts/run_ours_sparse15_optimized.py \
      --run-dir "$run" --method "$arm" --gpus "$GPUS" \
      ${suffix:+--diagonal-checkpoint-suffix "$suffix"} \
    ) > "$run/eval.log" 2>&1
  local rc=$?
  if [ ! -f "$run/sparse15_summary.json" ]; then
    say "$arm: EVAL FAILED rc=$rc ($run/eval.log)"; return 1
  fi
  local score
  score=$($PY - "$run/sparse15_summary.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"AA {d.get('final_average', float('nan')):.2f} F {-d.get('BWT', float('nan')):.2f}")
PYEOF
)
  say "$arm: eval done -> $score"
}

say "queue: ${ARMS} (serial, each uses all of $GPUS)"
[ "${SKIP_WAIT:-0}" = "1" ] || wait_for_training_queue
rc=0
for arm in $ARMS; do
  eval_arm "$arm" || { rc=1; say "$arm failed or skipped; continuing to next arm"; }
done
say "eval queue done rc=$rc"
exit $rc
