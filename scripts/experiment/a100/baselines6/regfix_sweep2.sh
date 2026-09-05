#!/bin/bash
# Round 2 of the EWC lambda search, gradient-calibrated this time.
#
# Round 1 showed lambda=12000 is trajectory-identical to lambda=400: the
# value-based calibration was wrong because drift is resisted by the penalty
# *gradient* lambda*F*(theta-theta*), not the penalty value.  Back-calculating
# from the lambda=400 run's final drift gives ||F.delta|| = 6.03e-7 against an
# LM grad norm of 0.35, so gradient parity needs lambda ~ 5.8e5 and a 30%
# restraint ~ 1.7e5.  These two runs bracket that range; if even 6e5 leaves the
# trajectory unchanged, the penalty gradient is not reaching the optimizer at
# all and the implementation needs a debugger, not a bigger lambda.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/train_stage.sh"
R="$BASELINES6_OUTPUT_ROOT"
SW="${SWEEP_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/baselines6_regfix_sweep_20260818}"
WIKI="$R/common_dense/wiki"
run_baselines6_stage ewc code "$SW/ewc_l200000_code" "$WIKI" "$(state_dir "$WIKI" ewc)" 64 29934 \
    --continual-ewc-lambda 200000 --exit-interval 400 || true
run_baselines6_stage ewc code "$SW/ewc_l600000_code" "$WIKI" "$(state_dir "$WIKI" ewc)" 64 29935 \
    --continual-ewc-lambda 600000 --exit-interval 400 || true

# GEM runs last, after every EWC point, per the agreed order.  Round 1 was
# steered past its gem stage with a placeholder completion marker; clear it so
# this run starts clean.
rm -rf "$SW/gem_m0_code"
run_baselines6_stage trace_gem code "$SW/gem_m0_code" "$WIKI" "$(state_dir "$WIKI" trace_gem)" 64 29933 \
    --continual-trace-gem-margin 0 --exit-interval 400 || true
echo "[SWEEP2 DONE] $(date '+%F %T')"
