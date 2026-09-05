#!/usr/bin/env bash
set -euo pipefail

# End-to-end, checkpoint-gated Conversation old-like feasibility pipeline.
# Nothing after the oracle stage can start until its 1800-step checkpoint is
# complete.  Every child stage is restartable and refuses mismatched outputs.

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/../../.." && pwd)"
PIPELINE_ROOT="${PIPELINE_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/conversation_old_like_gt_pipeline_20260812}"
PLAN_ONLY="${PLAN_ONLY:-0}"
FLAME_ENV="${FLAME_ENV:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100}"
PYTHON_BIN="${PYTHON_BIN:-$FLAME_ENV/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "[ERROR] missing Python: $PYTHON_BIN" >&2; exit 1; }
# Megatron's dataset-helper Makefile invokes `python3` directly.  Pin PATH as
# well as PYTHON_BIN so it cannot accidentally compile against the base conda
# Python (currently 3.14) while torchrun itself uses the flame Python (3.10).
export PIPELINE_ROOT FLAME_ENV PYTHON_BIN PYTHONNOUSERSITE=1
export PATH="$FLAME_ENV/bin:/usr/bin:/bin"

cat <<EOF
[PIPELINE]
  1. Code hidden-MSE result -> 16E-to-24E expansion output-KD init, 600 steps
  2. Conversation LM + Wiki/Code replay LM oracle, mandatory 1800 steps
  3. paired Conversation forward: KD-init reference vs 1800-step oracle
  4. Conversation layerwise cosine top-1% x Layers 2-9 intersection GT
  5. GT-positive token-only indexed miniset, repeated to 20% exposure
  6. method-matched five-run comparison: completed oracle + MSE/KL/vocab-KL/LM
Root: $PIPELINE_ROOT
EOF
if [[ "$PLAN_ONLY" == 1 ]]; then
    PLAN_ONLY=1 bash "$D/run_g2_conversation_old_like_oracle_chain_mha.sh"
    echo "[PLAN] downstream stages are checkpoint-gated and intentionally not preflighted before stage 2 exists"
    exit 0
fi

mkdir -p "$PIPELINE_ROOT/logs"
exec 9>"$PIPELINE_ROOT/.full_pipeline.lock"
flock -n 9 || { echo "[ERROR] full pipeline already active" >&2; exit 1; }

bash "$D/run_g2_conversation_old_like_oracle_chain_mha.sh" \
    2>&1 | tee -a "$PIPELINE_ROOT/logs/full_pipeline_stage_1_2.log"

bash "$R/scripts/analysis/run_conversation_old_like_gt_build_chain.sh" \
    2>&1 | tee -a "$PIPELINE_ROOT/logs/full_pipeline_stage_3_5.log"

bash "$D/run_g2_conversation_old_like_method_matched_5run_chain_mha.sh" \
    2>&1 | tee -a "$PIPELINE_ROOT/logs/full_pipeline_stage_6.log"

echo "$(date -Is) COMPLETE Conversation old-like full pipeline" | tee -a "$PIPELINE_ROOT/logs/status.tsv"
