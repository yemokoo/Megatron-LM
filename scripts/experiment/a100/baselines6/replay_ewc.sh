#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/train_stage.sh"
source "$DIR/replay_common.sh"
R="$BASELINES6_OUTPUT_ROOT"
# lambda and fusion fix identical to the retrained no-replay EWC.
EWC_ARGS=(--continual-ewc-lambda "${EWC_LAMBDA:-600000}" --no-gradient-accumulation-fusion)
replay_pair ewc "$NOREPLAY_ROOT/common_dense/wiki" "$R/ewc/code" "$R/ewc/conversation" \
    "$(method_micro_batch ewc)" 29911 29912 ewc "${EWC_ARGS[@]}"
