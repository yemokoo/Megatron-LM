#!/bin/bash
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$D/common.sh"
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
G2_ROOT="${G2_ROOT:-$PROJECT_ROOT/.local/weights/a100/mha/g2-checkpoints}"
ROOT="${LPR_ROOT:-$G2_ROOT/lpr_chain}"
LPR_COEFF="${LPR_COEFF:-0.1}"
CODE_MB="${CODE_MB:-96}"
CODE_LPR_MB="${CODE_LPR_MB:-96}"
CONV_MB="${CONV_MB:-64}"
CONV_LPR_MB="${CONV_LPR_MB:-64}"
CODE_ID="g2-ffn-only-e8to16-code-lm-aux-z-mb${CODE_MB}-1800-lpr-chain"
CODE_OUT="$ROOT/code_task/$CODE_ID"
CODE_LPR_ID="g2-ffn-only-code-router-lpr-gamma${LPR_COEFF}-equal-token-mb${CODE_LPR_MB}-360"
CODE_LPR_OUT="$ROOT/code_router/$CODE_LPR_ID"
CONV_ID="g2-ffn-only-e16to24-conversation-lm-aux-z-mb${CONV_MB}-1800-from-lpr"
CONV_OUT="$ROOT/conversation_task/$CONV_ID"
CONV_LPR_ID="g2-ffn-only-conversation-router-lpr-gamma${LPR_COEFF}-equal-token-mb${CONV_LPR_MB}-360"
CONV_LPR_OUT="$ROOT/conversation_router/$CONV_LPR_ID"

if [ ! -f "$CODE_OUT/latest_checkpointed_iteration.txt" ] || [ "$(tr -d '[:space:]' < "$CODE_OUT/latest_checkpointed_iteration.txt")" != 1800 ]; then
  RUN_ID="$CODE_ID" TRAIN_WEIGHTS="$CODE_OUT" TRAIN_ITERS=1800 MICRO_BATCH_SIZE="$CODE_MB" MASTER_PORT=29930 \
    bash "$D/code_from_wiki_ffn_moe_g2matched_attn_freeze_mha_a100_bf16.sh"
fi
RUN_ID="$CODE_LPR_ID" LPR_COEFF="$LPR_COEFF" RETUNE_ITERS=360 MICRO_BATCH_SIZE="$CODE_LPR_MB" MASTER_PORT=29931 \
  bash "$D/run_g2_ffn_only_task_group_lpr_router_retune_mha.sh" code "$CODE_OUT" "$CODE_LPR_OUT"

if [ ! -f "$CONV_OUT/latest_checkpointed_iteration.txt" ] || [ "$(tr -d '[:space:]' < "$CONV_OUT/latest_checkpointed_iteration.txt")" != 1800 ]; then
  RUN_ONLY_STAGE=ffn_only FFN_SOURCE="$CODE_LPR_OUT" FFN_SOURCE_REQUIRED_ITERS=2160 \
  FFN_RUN_ID="$CONV_ID" FFN_TRAIN_WEIGHTS="$CONV_OUT" TRAIN_ITERS=1800 FFN_MICRO_BATCH_SIZE="$CONV_MB" \
  SOURCE_LOGICAL_STEP=2160 PAUSE_SECONDS=0 FFN_MASTER_PORT=29932 \
    bash "$D/run_g2_phase4_conversation_after_router_finetune_offline_chain_mha.sh" ffn_only
fi
RUN_ID="$CONV_LPR_ID" LPR_COEFF="$LPR_COEFF" RETUNE_ITERS=360 MICRO_BATCH_SIZE="$CONV_LPR_MB" MASTER_PORT=29933 \
  bash "$D/run_g2_ffn_only_task_group_lpr_router_retune_mha.sh" conversation "$CONV_OUT" "$CONV_LPR_OUT"

echo "[DONE] LPR chain: 1800 code -> 360 router -> 1800 conversation -> 360 router"
echo "[FINAL] $CONV_LPR_OUT"
