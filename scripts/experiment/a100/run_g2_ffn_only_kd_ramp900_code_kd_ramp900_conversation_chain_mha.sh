#!/bin/bash
set -euo pipefail

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/../../.." && pwd)"
W="${G2_ROOT:-$R/.local/weights/a100/mha/g2-checkpoints}"
L="${LOG_DIR:-$R/.local/logs/g2_kd_ramp900_code_kd_ramp900_conversation_chain}"
mkdir -p "$L"

CODE_STEPS="${CODE_STEPS:-1800}"
CODE_RAMP_STEPS="${CODE_RAMP_STEPS:-900}"
KD_STEPS="${KD_STEPS:-600}"
CONV_STEPS="${CONV_STEPS:-1800}"
CONV_RAMP_STEPS="${CONV_RAMP_STEPS:-900}"

CODE_MB="${CODE_MB:-96}"
KD_MB="${KD_MB:-32}"
CONV_MB="${CONV_MB:-96}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
PROBE_INTERVAL="${PROBE_INTERVAL:-50}"
PROBE_ITERS="${PROBE_ITERS:-25}"
PROBE_MB="${PROBE_MB:-32}"

# Completed e8->16 checkpoint initialized on Wiki with output-logits KD only.
INIT_ID="${INIT_RUN_ID:-g2-ffn-only-e8to16-code-expert-init-logits-wiki-distill-mha-a100-bf16-mb48-1800}"
INIT_OUT="${INIT_WEIGHTS:-$W/code/expansion_distill_init/$INIT_ID}"
INIT_EXPECTED="${INIT_EXPECTED_STEP:-1800}"

CODE_ID="${CODE_RUN_ID:-g2-ffn-only-code-wiki-joint-lm-allrouter-newexpert-ramp${CODE_RAMP_STEPS}-mb${CODE_MB}-${CODE_STEPS}}"
CODE_OUT="$W/code/joint_lm_replay_ramp/$CODE_ID"

# Expand 16->24, then initialize only the new experts/router rows with
# output-logits KD on an equal-token Wiki+Code mixture.
KD_ID="${KD_RUN_ID:-g2-ffn-only-e16to24-conv-init-from-code-ramp${CODE_RAMP_STEPS}-logits-wikicode-kd-mb${KD_MB}-${KD_STEPS}}"
KD_OUT="$W/conversation/expansion_distill_init_joint_code_ramp/$KD_ID"

CONV_ID="${CONV_RUN_ID:-g2-ffn-only-conv-wikicode-joint-lm-allrouter-112-newexpert-ramp${CONV_RAMP_STEPS}-mb${CONV_MB}-${CONV_STEPS}}"
CONV_OUT="$W/conversation/joint_lm_replay_ramp/$CONV_ID"

is_uint() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

done_at() {
    local output="$1"
    local expected="$2"
    [ -f "$output/latest_checkpointed_iteration.txt" ] \
        && [ "$(tr -d '[:space:]' < "$output/latest_checkpointed_iteration.txt")" = "$expected" ]
}

run_stage() {
    local name="$1"
    local output="$2"
    local expected="$3"
    local log_file="$4"
    shift 4

    if done_at "$output" "$expected"; then
        echo "[SKIP] $name already complete at step $expected: $output"
        return
    fi

    echo "[START] $name $(date -Is)"
    "$@" 2>&1 | tee "$log_file"
    if ! done_at "$output" "$expected"; then
        echo "[ERROR] $name incomplete; expected step $expected: $output" >&2
        exit 1
    fi
    echo "[END] $name completed at step $expected $(date -Is)"
}

for value in \
    "$CODE_STEPS" "$CODE_RAMP_STEPS" "$KD_STEPS" "$CONV_STEPS" \
    "$CONV_RAMP_STEPS" "$CODE_MB" "$KD_MB" "$CONV_MB" \
    "$GLOBAL_BATCH_SIZE" "$PROBE_INTERVAL" "$PROBE_ITERS" "$PROBE_MB"; do
    is_uint "$value" || {
        echo "[ERROR] step/batch/probe settings must be non-negative integers: $value" >&2
        exit 2
    }
done
if [ "$CODE_RAMP_STEPS" -le 0 ] || [ "$CODE_RAMP_STEPS" -gt "$CODE_STEPS" ]; then
    echo "[ERROR] CODE_RAMP_STEPS must be in 1..CODE_STEPS" >&2
    exit 2
fi
if [ "$CONV_RAMP_STEPS" -le 0 ] || [ "$CONV_RAMP_STEPS" -gt "$CONV_STEPS" ]; then
    echo "[ERROR] CONV_RAMP_STEPS must be in 1..CONV_STEPS" >&2
    exit 2
fi
if [ "$CODE_MB" -le 0 ] || [ "$KD_MB" -le 0 ] || [ "$CONV_MB" -le 0 ]; then
    echo "[ERROR] micro-batch sizes must be positive" >&2
    exit 2
fi

if ! done_at "$INIT_OUT" "$INIT_EXPECTED"; then
    echo "[ERROR] completed Wiki KD initialization checkpoint is missing: $INIT_OUT" >&2
    echo "        expected checkpoint step: $INIT_EXPECTED" >&2
    exit 1
fi

echo "[CHAIN] Wiki KD init -> Code ramp -> Wiki+Code KD init -> Conversation ramp"
echo "[RUN 1] Code+Wiki joint LM: ramp 1-${CODE_RAMP_STEPS}, full LR $((CODE_RAMP_STEPS + 1))-${CODE_STEPS}"
echo "[RUN 2] expand 16->24 + Wiki/Code output-logits KD: $KD_STEPS steps"
echo "[RUN 3] Conversation+Wiki+Code joint LM: ramp 1-${CONV_RAMP_STEPS}, full LR $((CONV_RAMP_STEPS + 1))-${CONV_STEPS}"
echo "[BATCH] code=$CODE_MB kd=$KD_MB conversation=$CONV_MB global=$GLOBAL_BATCH_SIZE"
echo "[SOURCE] $INIT_OUT"

if [ "${PLAN_ONLY:-0}" = "1" ]; then
    echo "[OUTPUT 1] $CODE_OUT"
    echo "[OUTPUT 2] $KD_OUT"
    echo "[OUTPUT 3] $CONV_OUT"
    exit 0
fi

run_stage \
    code_joint_ramp "$CODE_OUT" "$CODE_STEPS" "$L/stage1_code_joint_ramp.log" \
    env \
        SOURCE_WEIGHTS_DIR="$INIT_OUT" \
        SOURCE_REQUIRED_ITERS="$INIT_EXPECTED" \
        TRAIN_ITERS="$CODE_STEPS" \
        MICRO_BATCH_SIZE="$CODE_MB" \
        GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
        MOE_NEW_EXPERT_LR_RAMP_STEPS="$CODE_RAMP_STEPS" \
        LR_DECAY_ITERS="$CODE_STEPS" \
        LR_WSD_DECAY_ITERS="$((CODE_STEPS / 10))" \
        LR_WARMUP_FRACTION=0.01 \
        SAVE_INTERVAL="$CODE_RAMP_STEPS" \
        EVAL_INTERVAL="$CODE_RAMP_STEPS" \
        PROBE_NAME=code_probe \
        PROBE_DATASET="$R/data/code/test" \
        SECONDARY_PROBE_NAME=wiki_probe \
        SECONDARY_PROBE_DATASET="$R/data/wiki/test" \
        PROBE_EVAL_INTERVAL="$PROBE_INTERVAL" \
        SECONDARY_PROBE_EVAL_INTERVAL="$PROBE_INTERVAL" \
        PROBE_EVAL_ITERS="$PROBE_ITERS" \
        SECONDARY_PROBE_EVAL_ITERS="$PROBE_ITERS" \
        PROBE_MICRO_BATCH_SIZE="$PROBE_MB" \
        PROBE_STEP_OFFSET=3600 \
        SECONDARY_PROBE_STEP_OFFSET=3600 \
        WANDB_STEP_OFFSET=3600 \
        RUN_INITIAL_PROBE_EVAL=1 \
        RUN_ID="$CODE_ID" \
        TRAIN_WEIGHTS="$CODE_OUT" \
        MASTER_PORT=29987 \
        bash "$D/run_g2_ffn_only_code_wiki_joint_lm_allrouter_mha.sh"

run_stage \
    conversation_expansion_kd "$KD_OUT" "$KD_STEPS" "$L/stage2_wikicode_output_kd.log" \
    env \
        SOURCE_WEIGHTS_DIR="$CODE_OUT" \
        SOURCE_REQUIRED_ITERS="$CODE_STEPS" \
        TRAIN_ITERS="$KD_STEPS" \
        MICRO_BATCH_SIZE="$KD_MB" \
        GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
        MOE_NEW_EXPERT_LR_RAMP_STEPS=0 \
        PROBE_EVAL_INTERVAL="$PROBE_INTERVAL" \
        SECONDARY_PROBE_EVAL_INTERVAL="$PROBE_INTERVAL" \
        TERTIARY_PROBE_EVAL_INTERVAL="$PROBE_INTERVAL" \
        PROBE_EVAL_ITERS="$PROBE_ITERS" \
        SECONDARY_PROBE_EVAL_ITERS="$PROBE_ITERS" \
        TERTIARY_PROBE_EVAL_ITERS="$PROBE_ITERS" \
        PROBE_MICRO_BATCH_SIZE="$PROBE_MB" \
        RUN_INITIAL_PROBE_EVAL=1 \
        STAGE_DIR_NAME=a100/mha/g2-checkpoints/conversation/expansion_distill_init_joint_code_ramp \
        RUN_ID="$KD_ID" \
        TRAIN_WEIGHTS="$KD_OUT" \
        MASTER_PORT=29988 \
        bash "$D/run_g2_ffn_only_conversation_expert_distill_init_wikicode_mha.sh"

run_stage \
    conversation_joint_ramp "$CONV_OUT" "$CONV_STEPS" "$L/stage3_conversation_joint_ramp.log" \
    env \
        SOURCE_WEIGHTS_DIR="$KD_OUT" \
        SOURCE_REQUIRED_ITERS="$KD_STEPS" \
        TRAIN_ITERS="$CONV_STEPS" \
        MICRO_BATCH_SIZE="$CONV_MB" \
        GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
        MOE_NEW_EXPERT_LR_RAMP_STEPS="$CONV_RAMP_STEPS" \
        LR_DECAY_ITERS="$CONV_STEPS" \
        LR_WSD_DECAY_ITERS="$((CONV_STEPS / 10))" \
        LR_WARMUP_FRACTION=0.01 \
        SAVE_INTERVAL="$CONV_RAMP_STEPS" \
        EVAL_INTERVAL="$CONV_RAMP_STEPS" \
        PROBE_NAME=wiki_probe \
        PROBE_DATASET="$R/data/wiki/test" \
        SECONDARY_PROBE_NAME=code_probe \
        SECONDARY_PROBE_DATASET="$R/data/code/test" \
        TERTIARY_PROBE_NAME=conversation_probe \
        TERTIARY_PROBE_DATASET="$R/data/conversation/test" \
        PROBE_EVAL_INTERVAL="$PROBE_INTERVAL" \
        SECONDARY_PROBE_EVAL_INTERVAL="$PROBE_INTERVAL" \
        TERTIARY_PROBE_EVAL_INTERVAL="$PROBE_INTERVAL" \
        PROBE_EVAL_ITERS="$PROBE_ITERS" \
        SECONDARY_PROBE_EVAL_ITERS="$PROBE_ITERS" \
        TERTIARY_PROBE_EVAL_ITERS="$PROBE_ITERS" \
        PROBE_MICRO_BATCH_SIZE="$PROBE_MB" \
        PROBE_STEP_OFFSET=5400 \
        SECONDARY_PROBE_STEP_OFFSET=5400 \
        TERTIARY_PROBE_STEP_OFFSET=5400 \
        WANDB_STEP_OFFSET=5400 \
        RUN_INITIAL_PROBE_EVAL=1 \
        RUN_ID="$CONV_ID" \
        TRAIN_WEIGHTS="$CONV_OUT" \
        MASTER_PORT=29989 \
        bash "$D/run_g2_ffn_only_conversation_wikicode_joint_lm_allrouter_mha.sh"

echo "[DONE] ramp900 KD chain completed"
echo "[FINAL] $CONV_OUT"
