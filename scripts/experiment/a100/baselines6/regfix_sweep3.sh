#!/bin/bash
# Lambda sweep, this time with the penalty gradient actually reaching the
# optimizer.
#
# Root cause fixed here: with gradient_accumulation_fusion on, the fused
# linear backward writes the LM wgrad straight into main_grad, flags the
# param, and returns a dummy tensor; the DDP hook then discards param.grad --
# including the EWC penalty's contribution, which travels through ordinary
# autograd.  --no-gradient-accumulation-fusion sends every gradient through
# param.grad, so the hook sums LM and penalty together.  Only EWC needs the
# flag; GEM edits main_grad directly and never had this problem.
#
# lambda=400 runs first at the user's request: it is the original
# configuration, now measured for the first time with a live penalty.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/train_stage.sh"
R="$BASELINES6_OUTPUT_ROOT"
SW="${SWEEP_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/baselines6_regfix_sweep_20260818}"
WIKI="$R/common_dense/wiki"

port=29941
for LAMBDA in 400 12000 200000 600000; do
    run_baselines6_stage ewc code "$SW/ewc_fix_l${LAMBDA}_code" "$WIKI" "$(state_dir "$WIKI" ewc)" 64 "$port" \
        --continual-ewc-lambda "$LAMBDA" --exit-interval 400 \
        --no-gradient-accumulation-fusion || true
    port=$(( port + 1 ))
done

# GEM last, per the agreed order.  Clear the placeholder that steered round 1
# past this stage.
rm -rf "$SW/gem_m0_code"
run_baselines6_stage trace_gem code "$SW/gem_m0_code" "$WIKI" "$(state_dir "$WIKI" trace_gem)" 64 29945 \
    --continual-trace-gem-margin 0 --exit-interval 400 || true
echo "[SWEEP3 DONE] $(date '+%F %T')"
