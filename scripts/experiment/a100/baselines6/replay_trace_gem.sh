#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/train_stage.sh"
source "$DIR/replay_common.sh"
R="$BASELINES6_OUTPUT_ROOT"
# margin 0 identical to the retrained no-replay GEM.  Memory stays the frozen
# terminal gradient; replay changes only the data, not the mechanism.
GEM_ARGS=(--continual-trace-gem-margin "${GEM_MARGIN:-0}")
replay_pair trace_gem "$NOREPLAY_ROOT/common_dense/wiki" "$R/trace_gem/code" "$R/trace_gem/conversation" \
    "$(method_micro_batch trace_gem)" 29921 29922 trace_gem "${GEM_ARGS[@]}"
