#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

G2_2R_ROOT="${G2_2R_ROOT:-$PROJECT_ROOT/.local/weights/a100/mha/g2-2router}"
PHASE1_RUN_ID="${PHASE1_RUN_ID:-g2-2router-exp2-top4-e8to16-ffn352-r256-wiki-to-code-new-experts-new-router-rows-qkvo-mha-a100-bf16-mb72-1800}"
PHASE2_RUN_ID="${PHASE2_RUN_ID:-g2-2router-exp2-phase2-router-only-retune-wikicode-from-new-experts-new-router-rows-no-reinit-mb72-1800}"

SRC="${SRC:-$G2_2R_ROOT/code/phase1/$PHASE1_RUN_ID}"
DST="${DST:-$G2_2R_ROOT/code/phase2/$PHASE2_RUN_ID}"

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
    echo "[SKIP] destination already exists: $DST"
    exit 0
fi

echo "[COPY] source:      $SRC"
echo "[COPY] destination: $DST"
mkdir -p "$DST"
rsync -aH --info=progress2 \
    --exclude 'wandb/' \
    "$SRC/" "$DST/"

{
    echo "phase=2"
    echo "purpose=two-router router-only retuning copy"
    echo "experiment=g2-2router-exp2"
    echo "source_label=G2-2router exp2 Phase 1"
    echo "source=$SRC"
    echo "source_step=$SOURCE_STEP"
    echo "router_init=no-reinit"
    echo "trainable=attention router and FFN router only"
    echo "loss=single mixed wiki+code LM loss"
    echo "copied_at=$(date -Is)"
} > "$DST/PHASE2_SOURCE.txt"
cp "$DST/PHASE2_SOURCE.txt" "$DST/PHASE3_SOURCE.txt"

echo "[DONE] $DST"
