#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/train_stage.sh"

WIKI="$BASELINES6_OUTPUT_ROOT/fixed_moe/wiki"
CODE="$BASELINES6_OUTPUT_ROOT/fixed_moe/code"
CONV="$BASELINES6_OUTPUT_ROOT/fixed_moe/conversation"
MB="$(method_micro_batch fixed_moe)"
run_baselines6_stage fixed_moe wiki "$WIKI" "" "" "$MB" 29761
run_baselines6_stage fixed_moe code "$CODE" "$WIKI" "" "$MB" 29762
run_baselines6_stage fixed_moe conversation "$CONV" "$CODE" "" "$MB" 29763
