#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_stage.sh"

COMMON_WIKI="$BASELINES6_OUTPUT_ROOT/common_dense/wiki"
run_baselines6_stage \
    ewc wiki "$COMMON_WIKI" "" "" "$(method_micro_batch ewc)" 29701 \
    --continual-ewc-lambda 400 \
    --continual-capture-trace-terminal
