#!/bin/bash
set -euo pipefail

# Experiment 3: FFN-only expandable MoE, conversation stage with ATTENTION UNFREEZE.
#
# Continues from the wiki->code (router-retuned, 16-expert) checkpoint, expands
# experts 16->24, and trains conversation with:
#   - old experts / old router  : FROZEN  (only new experts + new router rows train)
#   - attention / shared trunk  : TRAINED (unfrozen)  <-- the difference vs the
#                                 already-existing FFN-only "freeze" conversation.
#
# The freeze variant already exists and is a baseline; this run produces only the
# attention-unfreeze counterpart. It is a single conversation run (1800 steps),
# not a full wiki->code->conv retrain.
#
# Thin wrapper over run_g2_phase4_conversation_after_router_finetune_offline_chain_mha.sh
# with RUN_ONLY_STAGE=ffn_only_attn_unfreeze.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec env \
    RUN_ONLY_STAGE=ffn_only_attn_unfreeze \
    "$SCRIPT_DIR/run_g2_phase4_conversation_after_router_finetune_offline_chain_mha.sh"
