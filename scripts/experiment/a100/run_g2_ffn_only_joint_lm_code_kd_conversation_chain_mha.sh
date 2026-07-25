#!/bin/bash
set -euo pipefail
export WANDB_MODE="${WANDB_MODE:-offline}"
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/../../.." && pwd)"
W="${G2_ROOT:-$R/.local/weights/a100/mha/g2-checkpoints}"
L="${LOG_DIR:-$R/.local/logs/g2_joint_lm_code_kd600_conversation_chain}"
mkdir -p "$L"

CODE_STEPS="${CODE_STEPS:-1800}"
KD_STEPS="${KD_STEPS:-600}"
CONV_STEPS="${CONV_STEPS:-1800}"
CODE_MB="${CODE_MB:-64}"
KD_MB="${KD_MB:-32}"
CONV_MB="${CONV_MB:-64}"

# Source: Wiki 8->16 expansion initialized with output-logits KD only.
INIT_ID="${INIT_RUN_ID:-g2-ffn-only-e8to16-code-expert-init-logits-wiki-distill-mha-a100-bf16-mb48-1800}"
INIT_OUT="${INIT_WEIGHTS:-$W/code/expansion_distill_init/$INIT_ID}"
INIT_EXPECTED="${INIT_EXPECTED_STEP:-1800}"

CODE_ID="${CODE_RUN_ID:-g2-ffn-only-code-wiki-joint-lm-allrouter-mb${CODE_MB}-${CODE_STEPS}}"
CODE_OUT="$W/code/joint_lm_replay/$CODE_ID"

# Run 2 expands 16->24 experts and performs Wiki+Code output-logits-only KD.
KD_ID="${KD_RUN_ID:-g2-ffn-only-e16to24-conv-init-from-joint-code-logits-wikicode-distill-mb${KD_MB}-${KD_STEPS}}"
KD_OUT="$W/conversation/expansion_distill_init_joint_code/$KD_ID"

CONV_ID="${CONV_RUN_ID:-g2-ffn-only-conv-wikicode-joint-lm-allrouter-112-mb${CONV_MB}-${CONV_STEPS}}"
CONV_OUT="$W/conversation/joint_lm_replay/$CONV_ID"

done_at(){
  [ -f "$1/latest_checkpointed_iteration.txt" ] &&
    [ "$(tr -d '[:space:]' < "$1/latest_checkpointed_iteration.txt")" = "$2" ]
}
run(){
  local name="$1" output="$2" expected="$3" log="$4"; shift 4
  if done_at "$output" "$expected"; then
    echo "[SKIP] $name already complete at $expected"
    return
  fi
  "$@" 2>&1 | tee "$log"
  done_at "$output" "$expected" || {
    echo "[ERROR] $name incomplete: $output (expected $expected)" >&2
    exit 1
  }
}

done_at "$INIT_OUT" "$INIT_EXPECTED" || {
  echo "[ERROR] output-only Wiki KD init checkpoint is missing/incomplete: $INIT_OUT" >&2
  exit 1
}

echo "[CHAIN] exactly 3 runs"
echo "[RUN 1] Code+Wiki joint LM: 8:16 experts + all router, mb=$CODE_MB, steps=$CODE_STEPS"
echo "[RUN 2] expand 16->24 + Wiki/Code output-only KD, mb=$KD_MB, steps=$KD_STEPS"
echo "[RUN 3] Conversation + Wiki/Code joint LM (1:1:2), mb=$CONV_MB, steps=$CONV_STEPS"
echo "[SOURCE] $INIT_OUT"

if [ "${PLAN_ONLY:-0}" = "1" ]; then
  echo "[OUTPUT 1] $CODE_OUT"
  echo "[OUTPUT 2] $KD_OUT"
  echo "[OUTPUT 3] $CONV_OUT"
  exit 0
fi

run code_joint "$CODE_OUT" "$CODE_STEPS" "$L/stage1_code_joint.log" env   SOURCE_WEIGHTS_DIR="$INIT_OUT" SOURCE_REQUIRED_ITERS="$INIT_EXPECTED"   TRAIN_ITERS="$CODE_STEPS" MICRO_BATCH_SIZE="$CODE_MB"   RUN_ID="$CODE_ID" TRAIN_WEIGHTS="$CODE_OUT" MASTER_PORT=29981   bash "$D/run_g2_ffn_only_code_wiki_joint_lm_allrouter_mha.sh"

run conversation_expansion_kd "$KD_OUT" "$KD_STEPS" "$L/stage2_wikicode_output_kd600.log" env   TRAIN_ITERS="$KD_STEPS" MICRO_BATCH_SIZE="$KD_MB"   SOURCE_WEIGHTS_DIR="$CODE_OUT" SOURCE_REQUIRED_ITERS="$CODE_STEPS"   STAGE_DIR_NAME=a100/mha/g2-checkpoints/conversation/expansion_distill_init_joint_code   RUN_ID="$KD_ID" TRAIN_WEIGHTS="$KD_OUT" MASTER_PORT=29982   bash "$D/run_g2_ffn_only_conversation_expert_distill_init_wikicode_mha.sh"

run conversation_joint "$CONV_OUT" "$CONV_STEPS" "$L/stage3_conversation_joint.log" env   TRAIN_ITERS="$CONV_STEPS" MICRO_BATCH_SIZE="$CONV_MB"   SOURCE_WEIGHTS_DIR="$KD_OUT" SOURCE_REQUIRED_ITERS="$KD_STEPS"   RUN_ID="$CONV_ID" TRAIN_WEIGHTS="$CONV_OUT" MASTER_PORT=29983   bash "$D/run_g2_ffn_only_conversation_wikicode_joint_lm_allrouter_mha.sh"

echo "[DONE] all 3 runs complete"
echo "[FINAL] $CONV_OUT"
