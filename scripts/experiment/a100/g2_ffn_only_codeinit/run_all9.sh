#!/bin/bash
# =============================================================================
# G2 FFN-only pre-Code expert-init distillation — CODE-DATA init 9-run driver
#
# Ablation of the wiki-init lineage: Stage A distills the expanded student on
# CODE data (vs wiki). Stages B and C are IDENTICAL to the wiki-init pipeline
# and REUSE the exact same runners — only the source checkpoint (Stage A output)
# and the output folders differ, set here via env overrides. No B/C logic is
# duplicated.
#
#   Stage A (x3): expand 8->16 + KL distill on CODE   (mb48, teacher=wiki 8e)
#   Stage B (x3): code training from the code-init     (mb96, no teacher)
#   Stage C (x3): router-only retune on WIKI+CODE mix  (mb96)
#
# Modes: logits | logits_hidden | logits_hidden_router
# Order: stage-major (all A -> all B -> all C). Sequential; completed stages
# auto-skip, so re-launching resumes after a failure.
#
# Outputs (kept SEPARATE from the wiki-init lineage):
#   code/expansion_distill_init/   ...-code-distill-...   <- Stage A (code-init)
#   code/from_code_distill_init/                          <- Stage B
#   code/from_code_distill_init_phase3/                   <- Stage C
#
# Launch detached with:  bash launch.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
A100_DIR="$PROJECT_ROOT/scripts/experiment/a100"
cd "$PROJECT_ROOT"

# ---- shared run environment -------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MODES="${MODES:-logits logits_hidden logits_hidden_router}"
MB_A="${MB_A:-48}"        # Stage A micro-batch (matches wiki-init lineage)
MB_BC="${MB_BC:-96}"      # Stage B/C micro-batch (matches wiki-init lineage)
ITERS="${ITERS:-1800}"
RETUNE="${RETUNE:-300}"    # Stage C router-retune iters (short finetune; target = 1800+300)

G2_ROOT="$PROJECT_ROOT/.local/weights/a100/mha/g2-checkpoints"
A_ROOT="$G2_ROOT/code/expansion_distill_init"          # shared dir; -code-distill- tag
B_ROOT="$G2_ROOT/code/from_code_distill_init"           # separate from wiki-init from_distill_init
C_ROOT="$G2_ROOT/code/from_code_distill_init_phase3"
LOGDIR="${LOGDIR:-$PROJECT_ROOT/.local/logs/g2_ffn_only_codeinit}"
mkdir -p "$LOGDIR"

safe(){ echo "${1//_/-}"; }
A_id(){ echo "g2-ffn-only-e8to16-code-expert-init-$(safe "$1")-code-distill-mha-a100-bf16-mb${MB_A}-${ITERS}"; }
B_id(){ echo "g2-ffn-only-e8to16-code-from-distill-init-$(safe "$1")-mha-a100-bf16-mb${MB_BC}-${ITERS}"; }
C_id(){ echo "g2-ffn-only-code-from-distill-init-$(safe "$1")-phase3-router-retune-wikicode-mb${MB_BC}-${RETUNE}"; }
done_at(){ [ -f "$1/latest_checkpointed_iteration.txt" ] && [ "$(tr -d '[:space:]' < "$1/latest_checkpointed_iteration.txt")" = "$2" ]; }

banner(){ echo ""; echo "############################################################"; echo "### $*"; echo "### $(date -Is)"; echo "############################################################"; }

# ---- Stage A: expansion + KL distill on CODE --------------------------------
for m in $MODES; do
  if done_at "$A_ROOT/$(A_id "$m")" "$ITERS"; then banner "SKIP  Stage A / $m (already at $ITERS)"; continue; fi
  banner "RUN   Stage A / $m   (mb=$MB_A, CODE distill)"
  MICRO_BATCH_SIZE="$MB_A" MASTER_PORT=29951 \
    bash "$A100_DIR/run_g2_ffn_only_code_expert_distill_init_codedata_mha.sh" "$m" 2>&1 | tee "$LOGDIR/A_$m.log" || true
  # Tolerate post-checkpoint teardown segfaults: verify success via the tracker,
  # not the exit code.
  done_at "$A_ROOT/$(A_id "$m")" "$ITERS" || { banner "FAIL  Stage A / $m (checkpoint not at $ITERS)"; exit 1; }
done

# ---- Stage B: code training from code-init (reuses wiki-init Stage B runner) --
for m in $MODES; do
  if done_at "$B_ROOT/$(B_id "$m")" "$ITERS"; then banner "SKIP  Stage B / $m (already at $ITERS)"; continue; fi
  banner "RUN   Stage B / $m   (mb=$MB_BC, code from code-init)"
  # Point the source at the code-init Stage A checkpoint and write the output to
  # the separate from_code_distill_init/ root (RUN_ID pattern kept identical so
  # Stage C's fixed source pattern still matches).
  MICRO_BATCH_SIZE="$MB_BC" MASTER_PORT=29952 \
    SOURCE_RUN_ID="$(A_id "$m")" \
    RUN_ID="$(B_id "$m")" \
    TRAIN_WEIGHTS="$B_ROOT/$(B_id "$m")" \
    WANDB_EXP_NAME="G2 FFN-only - code from ${m} CODE-init" \
    bash "$A100_DIR/run_g2_ffn_only_code_from_distill_init_mha.sh" "$m" 2>&1 | tee "$LOGDIR/B_$m.log" || true
  done_at "$B_ROOT/$(B_id "$m")" "$ITERS" || { banner "FAIL  Stage B / $m (checkpoint not at $ITERS)"; exit 1; }
done

# ---- Stage C: router-only retune (reuses wiki-init Stage C runner) -----------
for m in $MODES; do
  if done_at "$C_ROOT/$(C_id "$m")" "$((ITERS + RETUNE))"; then banner "SKIP  Stage C / $m (already at $((ITERS + RETUNE)))"; continue; fi
  banner "RUN   Stage C / $m   (mb=$MB_BC, router retune wiki+code)"
  # Source Stage B from the code-init root; write phase3 output to its own root.
  MICRO_BATCH_SIZE="$MB_BC" STAGE_B_MB="$MB_BC" RETUNE_ITERS="$RETUNE" \
    STAGE_B_ROOT="$B_ROOT" \
    PHASE3_ROOT="$C_ROOT" \
    bash "$A100_DIR/run_g2_ffn_only_code_distill_init_phase3_router_retune_mha.sh" "$m" 2>&1 | tee "$LOGDIR/C_$m.log" || true
  done_at "$C_ROOT/$(C_id "$m")" "$((ITERS + RETUNE))" || { banner "FAIL  Stage C / $m (checkpoint not at $((ITERS + RETUNE)))"; exit 1; }
done

banner "ALL 9 RUNS COMPLETE (code-init)"
