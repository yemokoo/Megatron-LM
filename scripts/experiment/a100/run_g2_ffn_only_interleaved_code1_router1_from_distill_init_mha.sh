#!/bin/bash
set -euo pipefail

# Extreme-granularity variant of the interleaved Code/router experiment.
# Keep this as a separate launcher so the original 50/50 baseline and its
# output naming remain unchanged.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MOE_INTERLEAVE_CODE_STEPS=1
export MOE_INTERLEAVE_ROUTER_STEPS=1

exec bash "$SCRIPT_DIR/run_g2_ffn_only_interleaved_code_router_from_distill_init_mha.sh" logits
