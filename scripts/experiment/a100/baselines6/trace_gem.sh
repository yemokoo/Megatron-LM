#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/train_stage.sh"
bash "$DIR/train_dense_wiki.sh"

WIKI="$BASELINES6_OUTPUT_ROOT/common_dense/wiki"
CODE="$BASELINES6_OUTPUT_ROOT/trace_gem/code"
CONV="$BASELINES6_OUTPUT_ROOT/trace_gem/conversation"
MB="$(method_micro_batch trace_gem)"
# margin is the dual lower bound v >= margin.  The GEM paper's projection
# (Eq. 11) is v >= 0; the 0.5 default is the "memory strength" the official code
# adds to nudge old-task loss downward, and it is only self-limiting when the
# memory gradient is recomputed every step from replayed examples.  TRACE
# freezes one terminal gradient per task instead, so with margin 0.5 the same
# vector is injected into every step; Adam integrates that fixed bias into
# straight-line drift 6-19x the natural update, aimed against the memory --
# which is why the shipped run destroyed Code (acc 0.63 -> 0.02).  Verified
# 2026-08-18: margin 0 removes the drift entirely.
GEM_ARGS=(--continual-trace-gem-margin "${GEM_MARGIN:-0}")
run_baselines6_stage trace_gem code "$CODE" "$WIKI" "$(state_dir "$WIKI" trace_gem)" "$MB" 29721 "${GEM_ARGS[@]}"
run_baselines6_stage trace_gem conversation "$CONV" "$CODE" "$(state_dir "$CODE" trace_gem)" "$MB" 29722 "${GEM_ARGS[@]}"
