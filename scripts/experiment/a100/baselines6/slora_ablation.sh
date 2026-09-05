#!/bin/bash
# SLoRA-Pre rank sweep, deferred until every baseline has finished.
#
# Rank 64 is the primary configuration and runs inside the main sequence via
# slora_pre.sh, so it is deliberately absent here.  Completed stages are
# skipped, so this is safe to re-run.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SLORA_RANKS="${SLORA_ABLATION_RANKS:-16 32 128 256}" bash "$DIR/slora_pre.sh"
