#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

usage() {
    cat >&2 <<'EOF'
usage: prepare_g2_phase3_copy.sh <exp1|exp2> [no-reinit|router-reinit]

Creates a Phase 3 router-retuning copy under:
  .local/weights/a100/mha/g2-checkpoints/code/phase3/<run-id>

The source checkpoint is not modified.
EOF
}

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    usage
    exit 1
fi

EXPERIMENT="$1"
ROUTER_INIT="${2:-no-reinit}"

case "$ROUTER_INIT" in
    no-reinit|router-reinit)
        ;;
    *)
        echo "[ERROR] router init must be no-reinit or router-reinit: $ROUTER_INIT" >&2
        exit 1
        ;;
esac

G2_ROOT="${G2_ROOT:-$PROJECT_ROOT/.local/weights/a100/mha/g2-checkpoints}"
PHASE3_ROOT="${PHASE3_ROOT:-$G2_ROOT/code/phase3}"
mkdir -p "$PHASE3_ROOT"

case "$EXPERIMENT" in
    exp1)
        SRC="$G2_ROOT/code/phase1/g2-top4-e8to16-ffn352-r256-wiki-to-code-shared-router-qkvo-all-experts-router-mha-a100-bf16-mb72-1800"
        RUN_ID="g2-exp1-phase3-router-only-retune-wikicode-from-all-experts-router-${ROUTER_INIT}-mb72-1800"
        SOURCE_LABEL="G2 - experiment 1 Phase 1"
        ;;
    exp2)
        SRC="$G2_ROOT/code/phase1/g2-exp2-top4-e8to16-ffn352-r256-wiki-to-code-new-experts-all-router-mha-a100-bf16-mb72-1800"
        RUN_ID="g2-exp2-phase3-router-only-retune-wikicode-from-new-experts-all-router-${ROUTER_INIT}-mb72-1800"
        SOURCE_LABEL="G2 - experiment 2 Phase 1"
        ;;
    *)
        echo "[ERROR] experiment must be exp1 or exp2: $EXPERIMENT" >&2
        exit 1
        ;;
esac

DST="$PHASE3_ROOT/$RUN_ID"

if [ ! -f "$SRC/latest_checkpointed_iteration.txt" ]; then
    echo "[ERROR] source checkpoint tracker not found: $SRC/latest_checkpointed_iteration.txt" >&2
    exit 1
fi

SOURCE_STEP="$(tr -d '\n\r[:space:]' < "$SRC/latest_checkpointed_iteration.txt")"
if [ "$SOURCE_STEP" != "1800" ]; then
    echo "[ERROR] source is not complete. expected 1800, got $SOURCE_STEP: $SRC" >&2
    exit 1
fi

if [ -e "$DST" ]; then
    echo "[ERROR] destination already exists: $DST" >&2
    exit 1
fi

echo "[COPY] source:      $SRC"
echo "[COPY] destination: $DST"
mkdir -p "$DST"
rsync -aH --info=progress2 \
    --exclude 'wandb/' \
    "$SRC/" "$DST/"

{
    echo "phase=3"
    echo "purpose=router-only retuning copy"
    echo "experiment=$EXPERIMENT"
    echo "source_label=$SOURCE_LABEL"
    echo "source=$SRC"
    echo "source_step=$SOURCE_STEP"
    echo "router_init=$ROUTER_INIT"
    echo "copied_at=$(date -Is)"
} > "$DST/PHASE3_SOURCE.txt"

echo "[DONE] $DST"
echo "[NEXT] use this as RESUME_FROM_WEIGHTS or LOAD source for the router-only retuning run."
