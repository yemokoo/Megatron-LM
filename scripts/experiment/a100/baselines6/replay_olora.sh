#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/train_stage.sh"
source "$DIR/replay_common.sh"
R="$BASELINES6_OUTPUT_ROOT"
OL_ARGS=(--continual-olora-rank 352 --continual-olora-alpha 352 \
         --continual-olora-dropout 0.1 --continual-olora-orth-lambda 0.5)
# O-LoRA has its own Wiki (base + slot0); reuse the no-replay one.
replay_pair olora "$NOREPLAY_ROOT/olora/wiki" "$R/olora/code" "$R/olora/conversation" \
    "$(method_micro_batch olora)" 29941 29942 "" "${OL_ARGS[@]}"
