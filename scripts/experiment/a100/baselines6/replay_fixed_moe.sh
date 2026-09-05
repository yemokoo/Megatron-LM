#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/train_stage.sh"
source "$DIR/replay_common.sh"
R="$BASELINES6_OUTPUT_ROOT"
# Fixed MoE has its own Wiki; reuse the no-replay one.
replay_pair fixed_moe "$NOREPLAY_ROOT/fixed_moe/wiki" "$R/fixed_moe/code" "$R/fixed_moe/conversation" \
    "$(method_micro_batch fixed_moe)" 29961 29962 ""
