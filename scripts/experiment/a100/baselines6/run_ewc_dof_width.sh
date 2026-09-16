#!/usr/bin/env bash
# Run a genuine width-specific EWC chain using the baseline-native layer spec.
# The ordinary DoF Wiki checkpoints use a different layer-spec key layout, so
# this chain trains its own width-specific Wiki stage before code/conversation.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/train_stage_dof.sh"

W="${W:?set W to 352, 704, 2816, or 5632}"
MB="${MB:?set a width-appropriate micro batch size}"
PORT="${MASTER_PORT:-44000}"
SWEEP_ROOT="${DOF_SWEEP_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/dof_sweep_20260908}"
OUT_ROOT="$SWEEP_ROOT/b6_ewc_dof_fixed"
OUT="$OUT_ROOT/w$W"
WIKI="$OUT/wiki"

case "$W" in
    352|704|2816|5632) ;;
    *) echo "unsupported DoF width: $W" >&2; exit 2 ;;
esac

export BASELINES6_OUTPUT_ROOT="$OUT_ROOT"
export TRAIN_ITERS=1800 SAVE_INTERVAL=1800 PAUSE_SECONDS=0
export DENSE_FFN_HIDDEN_SIZE="$W"
export CONTINUAL_DENSE_FFN_HIDDEN_SIZE="$W"

# Train the width-specific Wiki baseline and compute its empirical Fisher.
# A scheduler may preserve this stage while changing the DP world size for the
# more expensive Code/Conversation stages.
if [ "${SKIP_WIKI_STAGE:-0}" != 1 ]; then
    run_baselines6_stage ewc wiki "$WIKI" "" "" "$MB" "$PORT" \
        --continual-ewc-lambda 1800000 --no-gradient-accumulation-fusion
else
    [ "$(tr -d '[:space:]' < "$WIKI/latest_checkpointed_iteration.txt" 2>/dev/null || true)" = 1800 ] || {
        echo "cannot skip incomplete Wiki stage: $WIKI" >&2; exit 3;
    }
    [ -f "$WIKI/continual_state_ewc/manifest.json" ] || {
        echo "cannot skip Wiki stage without EWC Fisher: $WIKI" >&2; exit 3;
    }
fi

run_baselines6_stage ewc code "$OUT/code" "$WIKI" "$WIKI/continual_state_ewc" "$MB" "$((PORT + 1))" \
    --continual-ewc-lambda 1800000 --no-gradient-accumulation-fusion
run_baselines6_stage ewc conversation "$OUT/conversation" "$OUT/code" "$OUT/code/continual_state_ewc" "$MB" "$((PORT + 2))" \
    --continual-ewc-lambda 1800000 --no-gradient-accumulation-fusion

echo "EWC DoF width $W complete: $OUT"
