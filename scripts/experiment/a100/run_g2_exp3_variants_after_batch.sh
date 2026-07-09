#!/bin/bash
# Queue the two extra exp3 (FFN-only expand) conversation variants to run AFTER the
# main exp2..7 batch (run_g2_exp2345_sequential_mha.sh) finishes and frees the GPUs.
#
#   RUN2  conv-from-code (NO router finetune) : expand 16->24 conv, attn-unfreeze,
#         starting from the raw code checkpoint (phase2) instead of the router-
#         retuned one. This is the "router-FT-free" variant.
#   RUN1  phase5 router finetune              : router-only retune on full
#         wiki+code+conv, starting from the CURRENT exp3 attn-unfreeze conv (phase4)
#         checkpoint. Gives the properly-consolidated "after conv" state.
#
# Both are attn-unfreeze. No GPU is touched until the batch's orchestrator process
# is gone. Safe to launch now (it just waits). Run:
#   nohup bash scripts/experiment/a100/run_g2_exp3_variants_after_batch.sh > \
#         logs/exp3_variants_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

CODE_CKPT="$PROJECT_ROOT/.local/weights/a100/mha/g2matched-ffn-moe-attn-freeze-bf16/g2matched-top4-e8to16-ffn352-wiki-to-code-ffn-moe-attn-freeze-mha-a100-bf16-mb96-1800"
CONV_ATTN_UNFR="$PROJECT_ROOT/.local/weights/a100/mha/g2-checkpoints/conversation/phase4/g2-ffn-only-attn-unfreeze-phase4-conversation-from-router-retuned-e16to24-mb72-1800"

# --- wait for the main batch to finish (its orchestrator process disappears) ---
echo "[waiter] $(date) waiting for run_g2_exp2345_sequential batch to finish..."
while pgrep -f "run_g2_exp2345_sequential_mha.sh" >/dev/null 2>&1; do sleep 300; done
# secondary guard: let any lingering training process clear, then settle.
while pgrep -f "pretrain_gpt.py" >/dev/null 2>&1; do sleep 120; done
sleep 60
echo "[waiter] $(date) batch finished, GPUs free -> starting exp3 variants"

rc_run2=SKIPPED; rc_run1=SKIPPED

# ============================ RUN 2: conv from CODE (no router-FT) ============================
echo "=========================================================================="
echo "[RUN2] conv-from-code (no router-FT), attn-unfreeze  $(date)"
echo "       source(code, 16e): $CODE_CKPT"
if [ -f "$CODE_CKPT/latest_checkpointed_iteration.txt" ]; then
    env \
        FFN_SOURCE="$CODE_CKPT" \
        FFN_SOURCE_REQUIRED_ITERS=1800 \
        SOURCE_LOGICAL_STEP=3600 \
        FFN_MICRO_BATCH_SIZE=72 \
        FFN_RUN_ID="g2-ffn-only-attn-unfreeze-phase4-conversation-from-CODE-noRouterFT-e16to24-mb72-1800" \
        HF_HUB_OFFLINE="$HF_HUB_OFFLINE" TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE" \
        WANDB_MODE="$WANDB_MODE" \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" NPROC_PER_NODE="$NPROC_PER_NODE" \
        bash "$SCRIPT_DIR/run_g2_exp3_ffn_only_attn_unfreeze_conversation_mha.sh"
    rc_run2=$?
else
    echo "[RUN2][ERROR] code checkpoint missing; skipping"; rc_run2=MISSING
fi
echo "[RUN2] done rc=$rc_run2  $(date)"
sleep 180

# ============================ RUN 1: phase5 router-FT (wiki+code+conv) ============================
echo "=========================================================================="
echo "[RUN1] phase5 router-FT fullmix from attn-unfreeze conv  $(date)"
echo "       source(conv, 24e): $CONV_ATTN_UNFR"
if [ -f "$CONV_ATTN_UNFR/latest_checkpointed_iteration.txt" ]; then
    env \
        FFN_SOURCE="$CONV_ATTN_UNFR" \
        FFN_RUN_ID="g2-ffn-only-attn-unfreeze-phase5-router-retune-wikicodeconv-fullmix-from-attnunfr-conv-mb48" \
        FFN_WANDB_EXP_NAME="G2 FFN-only attn-unfreeze - phase5 router-only full wiki+code+conv" \
        HF_HUB_OFFLINE="$HF_HUB_OFFLINE" TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE" \
        WANDB_MODE="$WANDB_MODE" \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" NPROC_PER_NODE="$NPROC_PER_NODE" \
        bash "$SCRIPT_DIR/run_g2_phase5_router_only_retune_after_conversation_fullmix_offline_chain_mha.sh" ffn_only
    rc_run1=$?
else
    echo "[RUN1][ERROR] attn-unfreeze conv checkpoint missing; skipping"; rc_run1=MISSING
fi
echo "[RUN1] done rc=$rc_run1  $(date)"

echo "=========================================================================="
echo "[exp3-variants] SUMMARY  RUN2(conv-from-code)=$rc_run2  RUN1(phase5)=$rc_run1  $(date)"
echo "=========================================================================="
