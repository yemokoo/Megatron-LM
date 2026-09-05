#!/bin/bash
# Short validation runs that answer the two retraining questions with data:
#   1. EWC: does a recalibrated lambda actually hold wiki accuracy?  The
#      shipped lambda=400 leaves the penalty at 0.3% of the loss because the
#      Fisher is built from minibatch-mean gradients; 12k/48k put it at ~10%
#      and ~40%.
#   2. TRACE-GEM: does margin=0 (the canonical GEM dual, v>=0) stop the
#      projection from distorting training the way margin=0.5 did?
#
# Each run branches off the finished common Wiki checkpoint exactly like the
# real Code stage, but exits at iteration 400 -- enough for eight probe points
# against the originals' early trajectory at a quarter of the cost.  Probes and
# LR schedule are identical to the real runs (TRAIN_ITERS stays 1800).
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/train_stage.sh"

R="$BASELINES6_OUTPUT_ROOT"
SW="${SWEEP_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/baselines6_regfix_sweep_20260818}"
WIKI="$R/common_dense/wiki"
mkdir -p "$SW"

run_baselines6_stage ewc code "$SW/ewc_l12000_code" "$WIKI" "$(state_dir "$WIKI" ewc)" 64 29931 \
    --continual-ewc-lambda 12000 --exit-interval 400 || true
run_baselines6_stage ewc code "$SW/ewc_l48000_code" "$WIKI" "$(state_dir "$WIKI" ewc)" 64 29932 \
    --continual-ewc-lambda 48000 --exit-interval 400 || true
run_baselines6_stage trace_gem code "$SW/gem_m0_code" "$WIKI" "$(state_dir "$WIKI" trace_gem)" 64 29933 \
    --continual-trace-gem-margin 0 --exit-interval 400 || true
echo "[SWEEP DONE] $(date '+%F %T')"
