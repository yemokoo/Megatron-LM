#!/bin/bash
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/../../.." && pwd)"
W="$R/.local/weights/a100/mha/g2-checkpoints"
: "${WANDB_PROJECT:?WANDB_PROJECT must be set}"
WIKI_SOURCE="$W/wiki/g2matched-top4-e8-ffn352-wiki-ffn-moe-mha-a100-bf16-mb128-1800"
CODE_ID=g2-ffn-only-no-kd-random-e8to16-code-wiki-joint-allrouter-mb96-1800
CODE_OUT="$W/code/no_kd_joint_lm_replay/$CODE_ID"
CONV_ID=g2-ffn-only-no-kd-random-e16to24-conv-wikicode-joint-allrouter-112-mb96-1800
CONV_OUT="$W/conversation/no_kd_joint_lm_replay/$CONV_ID"
require_step() {
  local dir="$1" expected="$2" actual
  [ -f "$dir/latest_checkpointed_iteration.txt" ] || { echo "ERROR: missing tracker: $dir" >&2; exit 1; }
  actual="$(tr -d '[:space:]' < "$dir/latest_checkpointed_iteration.txt")"
  [ "$actual" = "$expected" ] || { echo "ERROR: expected $expected at $dir, found $actual" >&2; exit 1; }
}
require_step "$WIKI_SOURCE" 1800
if [ ! -f "$CODE_OUT/latest_checkpointed_iteration.txt" ]; then
  env CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 WANDB_MODE=offline WANDB_RESUME=never LOAD_EXPANDED_SOURCE=0 \
    SOURCE_WEIGHTS_DIR="$WIKI_SOURCE" SOURCE_REQUIRED_ITERS=1800 RUN_ID="$CODE_ID" TRAIN_WEIGHTS="$CODE_OUT" \
    TRAIN_ITERS=1800 MICRO_BATCH_SIZE=96 GLOBAL_BATCH_SIZE=2304 PROBE_EVAL_INTERVAL=50 SECONDARY_PROBE_EVAL_INTERVAL=50 \
    PROBE_EVAL_ITERS=25 SECONDARY_PROBE_EVAL_ITERS=25 PROBE_STEP_OFFSET=1800 SECONDARY_PROBE_STEP_OFFSET=1800 \
    WANDB_STEP_OFFSET=1800 RUN_INITIAL_PROBE_EVAL=1 RUN_INITIAL_VALID_EVAL=0 LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND=0 \
    SAVE_INTERVAL=600 LOG_INTERVAL=50 MASTER_PORT=29986 WANDB_RUN_ID="$CODE_ID" WANDB_EXP_NAME="$CODE_ID" \
    bash "$D/run_g2_ffn_only_code_wiki_joint_lm_allrouter_mha.sh"
fi
require_step "$CODE_OUT" 1800
if [ ! -f "$CONV_OUT/latest_checkpointed_iteration.txt" ]; then
  env CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 WANDB_MODE=offline WANDB_RESUME=never LOAD_EXPANDED_SOURCE=0 \
    SOURCE_WEIGHTS_DIR="$CODE_OUT" SOURCE_REQUIRED_ITERS=1800 RUN_ID="$CONV_ID" TRAIN_WEIGHTS="$CONV_OUT" \
    TRAIN_ITERS=1800 MICRO_BATCH_SIZE=96 GLOBAL_BATCH_SIZE=2304 PROBE_EVAL_INTERVAL=50 SECONDARY_PROBE_EVAL_INTERVAL=50 \
    TERTIARY_PROBE_DATASET="$R/data/wiki/test" TERTIARY_PROBE_NAME=wiki_probe TERTIARY_PROBE_EVAL_INTERVAL=50 \
    PROBE_EVAL_ITERS=25 SECONDARY_PROBE_EVAL_ITERS=25 TERTIARY_PROBE_EVAL_ITERS=25 \
    PROBE_STEP_OFFSET=3600 SECONDARY_PROBE_STEP_OFFSET=3600 TERTIARY_PROBE_STEP_OFFSET=3600 WANDB_STEP_OFFSET=3600 \
    RUN_INITIAL_PROBE_EVAL=1 RUN_INITIAL_VALID_EVAL=0 LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND=0 \
    SAVE_INTERVAL=600 LOG_INTERVAL=50 MASTER_PORT=29987 WANDB_RUN_ID="$CONV_ID" WANDB_EXP_NAME="$CONV_ID" \
    bash "$D/run_g2_ffn_only_conversation_wikicode_joint_lm_allrouter_mha.sh"
fi
require_step "$CONV_OUT" 1800
echo "[DONE] no-KD random-expansion joint-LM Code -> Conversation chain"
