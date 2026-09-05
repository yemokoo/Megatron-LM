#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/train_stage.sh"
source "$DIR/replay_common.sh"
R="$BASELINES6_OUTPUT_ROOT"
replay_pair sequential_dense "$NOREPLAY_ROOT/common_dense/wiki" \
    "$R/sequential_dense/code" "$R/sequential_dense/conversation" \
    "$(method_micro_batch sequential_dense)" 29951 29952 ""
