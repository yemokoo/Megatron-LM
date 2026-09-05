#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/train_stage.sh"
bash "$DIR/train_dense_wiki.sh"

WIKI="$BASELINES6_OUTPUT_ROOT/common_dense/wiki"
CODE="$BASELINES6_OUTPUT_ROOT/ewc/code"
CONV="$BASELINES6_OUTPUT_ROOT/ewc/conversation"
MB="$(method_micro_batch ewc)"

# The EWC penalty travels through ordinary autograd into param.grad.  With
# gradient-accumulation fusion on, the fused linear backward writes the LM
# wgrad straight into main_grad, flags the param, and the DDP hook then throws
# param.grad away -- penalty included.  Disabling fusion routes every gradient
# through param.grad so LM and penalty are summed.  Verified 2026-08-18: with
# fusion on, lambda 400 -> 48000 produced identical trajectories.
#
# lambda is calibrated to this branch's Fisher, which is the squared
# *minibatch-mean* gradient (128 samples) and therefore ~1e2 smaller than the
# per-example Fisher behind the paper's 400.  Sweep at 400 iters, wiki acc:
# seq-dense .329 | 400 .329 | 12k .336 | 200k .380 | 600k .400 (code -4pt).
EWC_ARGS=(--continual-ewc-lambda "${EWC_LAMBDA:-600000}" --no-gradient-accumulation-fusion)
run_baselines6_stage ewc code "$CODE" "$WIKI" "$(state_dir "$WIKI" ewc)" "$MB" 29711 "${EWC_ARGS[@]}"
run_baselines6_stage ewc conversation "$CONV" "$CODE" "$(state_dir "$CODE" ewc)" "$MB" 29712 "${EWC_ARGS[@]}"
