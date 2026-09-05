#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/train_stage.sh"

WIKI="$BASELINES6_OUTPUT_ROOT/olora/wiki"
CODE="$BASELINES6_OUTPUT_ROOT/olora/code"
CONV="$BASELINES6_OUTPUT_ROOT/olora/conversation"
MB="$(method_micro_batch olora)"
OLORA_ARGS=(--continual-olora-rank 352 --continual-olora-alpha 352 --continual-olora-dropout 0.1 --continual-olora-orth-lambda 0.5)
run_baselines6_stage olora wiki "$WIKI" "" "" "$MB" 29741 "${OLORA_ARGS[@]}"
run_baselines6_stage olora code "$CODE" "$WIKI" "" "$MB" 29742 "${OLORA_ARGS[@]}"
run_baselines6_stage olora conversation "$CONV" "$CODE" "" "$MB" 29743 "${OLORA_ARGS[@]}"
