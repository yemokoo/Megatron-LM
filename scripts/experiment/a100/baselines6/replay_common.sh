#!/bin/bash
# Replay-matched variant of the baselines: every method sees the same fixed
# old-task subsets our own lm-loss method saw (0.1% x 200 epochs), blended
# into the primary stream.  Code replays wiki 0.1%; Conversation replays
# wiki 0.05% + code 0.05% (0.1% total).  Steps grow 1800 -> 2160.
#
# The shared Wiki checkpoint is reused from the no-replay run.  Method-specific
# regularizer state (EWC Fisher, GEM memory, SLoRA reference) is likewise taken
# from that Wiki stage, so the only thing that changes is the training data.
REPLAY_ROOT="${BASELINES6_REPLAY_MINISET_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/baselines6_replay_miniset_seed1234}"
export BASELINES6_REPLAY_EPOCHS="${BASELINES6_REPLAY_EPOCHS:-200}"
REPLAY_CODE_DIRS="$REPLAY_ROOT/0p1pct/wiki/train"
REPLAY_CONV_DIRS="$REPLAY_ROOT/0p05pct/wiki/train $REPLAY_ROOT/0p05pct/code/train"

# The no-replay run root: its common_dense/wiki, olora/wiki, fixed_moe/wiki are
# the shared starting points.  Output goes to BASELINES6_OUTPUT_ROOT (set by
# the caller to the replay root).
NOREPLAY_ROOT="${BASELINES6_NOREPLAY_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/baselines6_wiki_code_conversation_20260816}"

for d in $REPLAY_CODE_DIRS $REPLAY_CONV_DIRS; do
    [ -f "$d/train_text_document.bin" ] || { echo "ERROR: replay subset missing: $d" >&2; exit 1; }
done

# Run a Code+Conversation pair for one method with replay wired in.
#   replay_pair METHOD WIKI_DIR CODE_OUT CONV_OUT MB PORT_CODE PORT_CONV [STATE_METHOD] [method args...]
replay_pair() {
    local method="$1" wiki="$2" code_out="$3" conv_out="$4" mb="$5" pc="$6" pv="$7"; shift 7
    local state_method="${1:-}"; [ $# -gt 0 ] && shift
    # State hand-off mirrors the original method scripts exactly:
    #   ewc / trace_gem : Code loads the Wiki sidecar (Fisher / terminal grad),
    #                     Conversation loads the Code sidecar.
    #   slora_pre       : Code takes NO state -- it snapshots the immutable Wiki
    #                     reference itself; only Conversation loads Code's sidecar.
    #   olora / dense / moe : no sidecar at all.
    local code_state="" conv_state=""
    case "$state_method" in
        ewc|trace_gem)
            code_state="$(state_dir "$wiki" "$state_method")"
            conv_state="$(state_dir "$code_out" "$state_method")" ;;
        slora_pre)
            conv_state="$(state_dir "$code_out" "$state_method")" ;;
    esac
    BASELINES6_REPLAY_DIRS="$REPLAY_CODE_DIRS" \
        run_baselines6_stage "$method" code "$code_out" "$wiki" "$code_state" "$mb" "$pc" "$@"
    BASELINES6_REPLAY_DIRS="$REPLAY_CONV_DIRS" \
        run_baselines6_stage "$method" conversation "$conv_out" "$code_out" "$conv_state" "$mb" "$pv" "$@"
}
