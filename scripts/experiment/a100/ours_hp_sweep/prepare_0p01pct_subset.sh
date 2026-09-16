#!/usr/bin/env bash
# Build the one replay miniset the sweep is missing: 0.01% x 2000 epochs.
#
# seed1234 already holds 0p1pctx200, 1pctx20 and 10pctx2, all at the same total
# exposure (~829,4xx sequences = 20% of one epoch) with only the unique-data
# count differing -- that is the convention the sweep's replay axis follows.
# 0p01pctx2000 is already declared in the builder's default MINISET_SPECS; this
# wrapper just runs that one spec into the existing seed1234 tree.
#
# Writes only the new label's directory. Existing subsets are skipped by the
# builder unless FORCE_REBUILD=1, which this script never sets.
set -euo pipefail

P=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning
ROOT_BASE=/data2/seonghyeonnoh/LLM-continual-learning-data/router_finetune_miniset_repeats/seed1234

if [ -d "$ROOT_BASE/0p01pctx2000/wiki/train" ]; then
    echo "[SKIP] already built: $ROOT_BASE/0p01pctx2000"
    exit 0
fi

OUTPUT_ROOT_BASE="$ROOT_BASE" \
CACHE_ROOT_BASE="$ROOT_BASE/cache" \
MINISET_SPECS="0p01pctx2000:0.0001:2000" \
FIXED_DATA_SEED=1234 \
FORCE_REBUILD=0 \
    bash "$P/scripts/experiment/a100/prepare_g2_router_finetune_miniset_repeats_mha.sh"

echo "[OK] $ROOT_BASE/0p01pctx2000"
