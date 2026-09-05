#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/train_stage.sh"
source "$DIR/replay_common.sh"
R="$BASELINES6_OUTPUT_ROOT"
RANK="${SLORA_RANK:-64}"
SL_ARGS=(--continual-slora-rank "$RANK" --continual-slora-conversation-rank 64 \
         --continual-slora-max-rank 256 --continual-slora-alpha 128)
replay_pair slora_pre "$NOREPLAY_ROOT/common_dense/wiki" \
    "$R/slora_pre/rank$RANK/code" "$R/slora_pre/rank$RANK/conversation" \
    "$(method_micro_batch slora_pre)" 29931 29932 slora_pre "${SL_ARGS[@]}"
