#!/bin/bash
# Paper-faithful GEM (Lopez-Paz & Ranzato 2017) on Wiki -> Code -> Conversation.
#
# Differences from the existing `trace_gem` baseline, which reproduces the
# TRACE benchmark's variant and is left untouched:
#   * the constraint gradient of each past task is recomputed EVERY step from a
#     stored episodic memory of examples, instead of one stale terminal
#     gradient saved at the end of that task;
#   * the inequality-constrained QP is solved once on the concatenated global
#     gradient vector, instead of independently per parameter tensor.
#
# Episodic memory budget is the same 0.1% subset our own method is credited
# with: Code sees 0.1% of Wiki; Conversation sees 0.05% Wiki + 0.05% Code, so
# the stored total stays at 0.1% of one task's corpus.  A fresh global batch is
# drawn cyclically from that stored subset at every optimizer step, which costs
# one extra forward/backward per past task (2x at Code, 3x at Conversation).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/train_stage.sh"

MINISET="${BASELINES6_REPLAY_MINISET_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/baselines6_replay_miniset_seed1234}"
WIKI_MEM_0P1="$MINISET/0p1pct/wiki/train/train_text_document"
WIKI_MEM_0P05="$MINISET/0p05pct/wiki/train/train_text_document"
CODE_MEM_0P05="$MINISET/0p05pct/code/train/train_text_document"
for prefix in "$WIKI_MEM_0P1" "$WIKI_MEM_0P05" "$CODE_MEM_0P05"; do
    [ -f "${prefix}.bin" ] && [ -f "${prefix}.idx" ] || {
        echo "ERROR: episodic memory subset missing: ${prefix}.bin/.idx" >&2; exit 1; }
done

# The Wiki backbone is shared with the other dense baselines; GEM only starts
# constraining from Code onwards, so Wiki itself is reused, never retrained.
WIKI="${GEM_EPISODIC_WIKI:-$BASELINES6_OUTPUT_ROOT/common_dense/wiki}"
CODE="$BASELINES6_OUTPUT_ROOT/gem_episodic/code"
CONV="$BASELINES6_OUTPUT_ROOT/gem_episodic/conversation"
MB="$(method_micro_batch gem_episodic)"
MARGIN="${GEM_MARGIN:-0.5}"
EPS="${GEM_EPS:-1e-3}"

run_baselines6_stage gem_episodic code "$CODE" "$WIKI" "" "$MB" "${PORT_CODE:-29771}" \
    --continual-gem-margin "$MARGIN" --continual-gem-eps "$EPS" \
    --continual-gem-memory-data-path 1.0 "$WIKI_MEM_0P1"

run_baselines6_stage gem_episodic conversation "$CONV" "$CODE" "" "$MB" "${PORT_CONV:-29772}" \
    --continual-gem-margin "$MARGIN" --continual-gem-eps "$EPS" \
    --continual-gem-memory-data-path 1.0 "$WIKI_MEM_0P05" \
    --continual-gem-memory-data-path 1.0 "$CODE_MEM_0P05"
