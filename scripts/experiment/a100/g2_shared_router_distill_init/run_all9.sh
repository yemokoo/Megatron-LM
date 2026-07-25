#!/bin/bash
# =============================================================================
# G2 shared-router (FFN + QKVO attention experts) pre-Code expert-init
# distillation — 9-run sequential driver
#
#   Stage A (x3): expand 8->16 + KL distill on WIKI      (mb48, teacher resident)
#   Stage B (x3): code training from the distill-init    (mb72, no teacher)
#   Stage C (x3): router-only retune on WIKI+CODE mix    (mb72)
#
# Modes: logits | logits_hidden | logits_hidden_router
# Order: stage-major (all A -> all B -> all C). Each run uses all 4 GPUs (DP=4),
# gbs=2304. Runs are sequential. Completed stages are auto-skipped, so this is
# safe to re-launch to resume after a failure.
#
# Launch it detached with:  bash launch.sh   (see that script)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
A100_DIR="$PROJECT_ROOT/scripts/experiment/a100"
cd "$PROJECT_ROOT"

# ---- shared run environment -------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export WANDB_MODE="${WANDB_MODE:-offline}"     # no login required; sync later
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # use local pythia tokenizer snapshot
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MODES="${MODES:-logits logits_hidden logits_hidden_router}"
MB_A="${MB_A:-48}"        # Stage A micro-batch (teacher resident -> conservative)
MB_BC="${MB_BC:-72}"      # Stage B/C micro-batch (attention experts -> heavier)
ITERS="${ITERS:-1800}"    # Stage A/B train iters
RETUNE="${RETUNE:-1800}"  # Stage C router-retune iters (target step = 1800+1800)

G2_ROOT="$PROJECT_ROOT/.local/weights/a100/mha/g2-checkpoints"
LOGDIR="${LOGDIR:-$PROJECT_ROOT/.local/logs/g2_shared_router_distill_init}"
mkdir -p "$LOGDIR"

safe(){ echo "${1//_/-}"; }
A_dir(){ echo "$G2_ROOT/code/shared_router_expansion_distill_init/g2-shared-router-e8to16-code-expert-init-$(safe "$1")-wiki-distill-qkvo-mha-a100-bf16-mb${MB_A}-${ITERS}"; }
B_dir(){ echo "$G2_ROOT/code/shared_router_from_distill_init/g2-shared-router-e8to16-code-from-distill-init-$(safe "$1")-qkvo-mha-a100-bf16-mb${MB_BC}-${ITERS}"; }
C_dir(){ echo "$G2_ROOT/code/shared_router_from_distill_init_phase3/g2-shared-router-code-from-distill-init-$(safe "$1")-phase3-router-retune-wikicode-mb${MB_BC}-${RETUNE}"; }
done_at(){ [ -f "$1/latest_checkpointed_iteration.txt" ] && [ "$(tr -d '[:space:]' < "$1/latest_checkpointed_iteration.txt")" = "$2" ]; }

banner(){ echo ""; echo "############################################################"; echo "### $*"; echo "### $(date -Is)"; echo "############################################################"; }

# ---- Stage A: expansion + KL distill on wiki --------------------------------
for m in $MODES; do
  if done_at "$(A_dir "$m")" "$ITERS"; then banner "SKIP  Stage A / $m (already at $ITERS)"; continue; fi
  banner "RUN   Stage A / $m   (mb=$MB_A, wiki distill)"
  MICRO_BATCH_SIZE="$MB_A" MASTER_PORT=29921 \
    bash "$A100_DIR/run_g2_shared_router_code_expert_distill_init_mha.sh" "$m" 2>&1 | tee "$LOGDIR/A_$m.log"
done

# ---- Stage B: code training from distill-init (no re-expansion) --------------
for m in $MODES; do
  if done_at "$(B_dir "$m")" "$ITERS"; then banner "SKIP  Stage B / $m (already at $ITERS)"; continue; fi
  banner "RUN   Stage B / $m   (mb=$MB_BC, code from distill-init)"
  # DISTILL_SOURCE_MB must match Stage A's mb so Stage B finds the checkpoint.
  MICRO_BATCH_SIZE="$MB_BC" DISTILL_SOURCE_MB="$MB_A" MASTER_PORT=29922 \
    bash "$A100_DIR/run_g2_shared_router_code_from_distill_init_mha.sh" "$m" 2>&1 | tee "$LOGDIR/B_$m.log"
done

# ---- Stage C: router-only retune on wiki+code -------------------------------
for m in $MODES; do
  if done_at "$(C_dir "$m")" "$((ITERS + RETUNE))"; then banner "SKIP  Stage C / $m (already at $((ITERS + RETUNE)))"; continue; fi
  banner "RUN   Stage C / $m   (mb=$MB_BC, router retune wiki+code)"
  # STAGE_B_MB must match Stage B's mb so Stage C finds the checkpoint.
  MICRO_BATCH_SIZE="$MB_BC" STAGE_B_MB="$MB_BC" \
    bash "$A100_DIR/run_g2_shared_router_code_distill_init_phase3_router_retune_mha.sh" "$m" 2>&1 | tee "$LOGDIR/C_$m.log"
done

banner "ALL 9 RUNS COMPLETE"
