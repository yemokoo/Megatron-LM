#!/bin/bash
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/../../.." && pwd)"
W="${G2_ROOT:-$R/.local/weights/a100/mha/g2-checkpoints}"
L="${LOG_DIR:-$R/.local/logs/g2_kd_repeat_1pct_20ep_joint_chain}"
SEED="${KD_SUBSET_SEED:-20260725}"
SUBSETS="${KD_SUBSET_ROOT:-$R/data/kd_subsets/g2_repeat_1pct_seed${SEED}}"
WIKI_SUBSET="$SUBSETS/wiki"
WIKI_HALF_SUBSET="$SUBSETS/wiki_half"
CODE_SUBSET="$SUBSETS/code"
CODE_HALF_SUBSET="$SUBSETS/code_half"
mkdir -p "$L"

SUBSET_SAMPLES="${KD_SUBSET_SAMPLES:-41472}"
KD_EPOCHS="${KD_EPOCHS:-20}"
GBS="${GLOBAL_BATCH_SIZE:-2304}"
KD1_STEPS="${KD1_STEPS:-360}"
CODE_STEPS="${CODE_STEPS:-1800}"
KD2_STEPS="${KD2_STEPS:-360}"
CONV_STEPS="${CONV_STEPS:-1800}"
KD1_MB="${KD1_MB:-48}"
CODE_MB="${CODE_MB:-64}"
KD2_MB="${KD2_MB:-32}"
CONV_MB="${CONV_MB:-96}"

kd1_samples=$((SUBSET_SAMPLES * KD_EPOCHS))
kd2_samples=$((SUBSET_SAMPLES * KD_EPOCHS))
[ $((kd1_samples % GBS)) -eq 0 ] || { echo "ERROR: KD1 samples not divisible by GBS" >&2; exit 1; }
[ $((kd2_samples % GBS)) -eq 0 ] || { echo "ERROR: KD2 samples not divisible by GBS" >&2; exit 1; }
[ "$KD1_STEPS" -eq $((kd1_samples / GBS)) ] || { echo "ERROR: KD1_STEPS must equal subset*epochs/GBS" >&2; exit 1; }
[ "$KD2_STEPS" -eq $((kd2_samples / GBS)) ] || { echo "ERROR: KD2_STEPS must equal total-1pct-subset*epochs/GBS" >&2; exit 1; }
KD1_OFFSET=1800
CODE_OFFSET=$((KD1_OFFSET + KD1_STEPS))
KD2_OFFSET=$((CODE_OFFSET + CODE_STEPS))
CONV_OFFSET=$((KD2_OFFSET + KD2_STEPS))

WIKI_SOURCE="${WIKI_SOURCE:-$W/wiki/g2matched-top4-e8-ffn352-wiki-ffn-moe-mha-a100-bf16-mb128-1800}"
KD1_ID="${KD1_RUN_ID:-g2-e8to16-codeinit-logits-kd-wiki1pct-repeat20-seed${SEED}-mb${KD1_MB}-${KD1_STEPS}}"
KD1_OUT="$W/code/expansion_distill_init_repeat1pct20ep/$KD1_ID"
CODE_ID="${CODE_RUN_ID:-g2-code-wiki-joint-from-kd1pct20ep-allrouter-mb${CODE_MB}-${CODE_STEPS}}"
CODE_OUT="$W/code/joint_lm_replay_repeat1pct20ep/$CODE_ID"
KD2_ID="${KD2_RUN_ID:-g2-e16to24-convinit-logits-kd-wikicode-total1pct-wiki0p5-code0p5-repeat20-equal-seed${SEED}-mb${KD2_MB}-${KD2_STEPS}}"
KD2_OUT="$W/conversation/expansion_distill_init_joint_code_repeat1pct20ep/$KD2_ID"
CONV_ID="${CONV_RUN_ID:-g2-conv-wikicode-joint-from-kd1pct20ep-allrouter-112-mb${CONV_MB}-${CONV_STEPS}}"
CONV_OUT="$W/conversation/joint_lm_replay_repeat1pct20ep/$CONV_ID"

done_at() {
    [ -f "$1/latest_checkpointed_iteration.txt" ] &&
        [ "$(tr -d '[:space:]' < "$1/latest_checkpointed_iteration.txt")" = "$2" ]
}
require_at() {
    done_at "$1" "$2" || { echo "ERROR: expected checkpoint $1 at step $2" >&2; exit 1; }
}
run_stage() {
    local name="$1" output="$2" expected="$3" log="$4"; shift 4
    if done_at "$output" "$expected"; then echo "[SKIP] $name complete"; return; fi
    "$@" 2>&1 | tee "$log"
    require_at "$output" "$expected"
}

require_at "$WIKI_SOURCE" 1800
cat <<PLAN
[EXPERIMENT] fixed random 1% KD subset repeated for 20 epochs
[SUBSET] seed=$SEED, 41,472 samples/task, seq=512
[KD 1] Wiki 1% x20 = 829,440 samples = 360 steps at GBS 2304
[JOINT 1] Code LM + Wiki router replay, existing 1:1 aggregate, 1800 steps
[KD 2] Wiki 0.5% + Code 0.5%, equal 1:1, total 1% x20 = 829,440 samples = 360 steps
[JOINT 2] Conversation LM + Wiki/Code router replay, existing 1:1:2 aggregate, 1800 steps
[FINAL] $CONV_OUT
PLAN
if [ "${PLAN_ONLY:-0}" = 1 ]; then exit 0; fi

KD_SUBSET_SEED="$SEED" KD_SUBSET_SAMPLES="$SUBSET_SAMPLES" KD_SUBSET_ROOT="$SUBSETS" bash "$D/prepare_g2_kd_repeat_1pct_subsets.sh"

run_stage code_kd "$KD1_OUT" "$KD1_STEPS" "$L/stage1_code_kd.log" env \
    SOURCE_WEIGHTS_DIR="$WIKI_SOURCE" SOURCE_REQUIRED_ITERS=1800 \
    TRAIN_DATASET="$WIKI_SUBSET" TRAIN_ITERS="$KD1_STEPS" MICRO_BATCH_SIZE="$KD1_MB" GLOBAL_BATCH_SIZE="$GBS" \
    SAVE_INTERVAL="$KD1_STEPS" EVAL_INTERVAL="$KD1_STEPS" PROBE_STEP_OFFSET="$KD1_OFFSET" \
    SECONDARY_PROBE_STEP_OFFSET="$KD1_OFFSET" WANDB_STEP_OFFSET="$KD1_OFFSET" \
    STAGE_DIR_NAME=a100/mha/g2-checkpoints/code/expansion_distill_init_repeat1pct20ep \
    RUN_ID="$KD1_ID" TRAIN_WEIGHTS="$KD1_OUT" MASTER_PORT=29891 \
    bash "$D/run_g2_ffn_only_code_expert_distill_init_mha.sh" logits

run_stage code_joint "$CODE_OUT" "$CODE_STEPS" "$L/stage2_code_joint.log" env \
    SOURCE_WEIGHTS_DIR="$KD1_OUT" SOURCE_REQUIRED_ITERS="$KD1_STEPS" \
    TRAIN_ITERS="$CODE_STEPS" MICRO_BATCH_SIZE="$CODE_MB" GLOBAL_BATCH_SIZE="$GBS" \
    PROBE_STEP_OFFSET="$CODE_OFFSET" SECONDARY_PROBE_STEP_OFFSET="$CODE_OFFSET" WANDB_STEP_OFFSET="$CODE_OFFSET" \
    RUN_ID="$CODE_ID" TRAIN_WEIGHTS="$CODE_OUT" MASTER_PORT=29892 \
    bash "$D/run_g2_ffn_only_code_wiki_joint_lm_allrouter_mha.sh"

run_stage conversation_kd "$KD2_OUT" "$KD2_STEPS" "$L/stage3_conversation_kd.log" env \
    SOURCE_WEIGHTS_DIR="$CODE_OUT" SOURCE_REQUIRED_ITERS="$CODE_STEPS" \
    TRAIN_DATASET="$WIKI_HALF_SUBSET" TRAIN_DATASET_SECONDARY="$CODE_HALF_SUBSET" TRAIN_DATA_WEIGHT_MODE=equal_dataset \
    TRAIN_ITERS="$KD2_STEPS" MICRO_BATCH_SIZE="$KD2_MB" GLOBAL_BATCH_SIZE="$GBS" SAVE_INTERVAL="$KD2_STEPS" EVAL_INTERVAL="$KD2_STEPS" \
    PROBE_STEP_OFFSET="$KD2_OFFSET" SECONDARY_PROBE_STEP_OFFSET="$KD2_OFFSET" TERTIARY_PROBE_STEP_OFFSET="$KD2_OFFSET" WANDB_STEP_OFFSET="$KD2_OFFSET" \
    STAGE_DIR_NAME=a100/mha/g2-checkpoints/conversation/expansion_distill_init_joint_code_repeat1pct20ep \
    RUN_ID="$KD2_ID" TRAIN_WEIGHTS="$KD2_OUT" MASTER_PORT=29893 \
    bash "$D/run_g2_ffn_only_conversation_expert_distill_init_wikicode_mha.sh"

run_stage conversation_joint "$CONV_OUT" "$CONV_STEPS" "$L/stage4_conversation_joint.log" env \
    SOURCE_WEIGHTS_DIR="$KD2_OUT" SOURCE_REQUIRED_ITERS="$KD2_STEPS" \
    TRAIN_ITERS="$CONV_STEPS" MICRO_BATCH_SIZE="$CONV_MB" GLOBAL_BATCH_SIZE="$GBS" \
    PROBE_STEP_OFFSET="$CONV_OFFSET" SECONDARY_PROBE_STEP_OFFSET="$CONV_OFFSET" TERTIARY_PROBE_STEP_OFFSET="$CONV_OFFSET" WANDB_STEP_OFFSET="$CONV_OFFSET" \
    RUN_ID="$CONV_ID" TRAIN_WEIGHTS="$CONV_OUT" MASTER_PORT=29894 \
    bash "$D/run_g2_ffn_only_conversation_wikicode_joint_lm_allrouter_mha.sh"

echo "[DONE] KD 1%-repeat20 Code -> Conversation chain"
echo "[FINAL] $CONV_OUT"
