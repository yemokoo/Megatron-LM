#!/bin/bash
set -euo pipefail

A100_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$A100_SCRIPTS_DIR/../../.." && pwd)"
G2_ROOT="${G2_ROOT:-$PROJECT_ROOT/.local/weights/a100/mha/g2-checkpoints}"
FIX_TAG="${FIX_TAG:-attn-loadfix-v3-nosaveoptim}"
LOG_ROOT="${LOG_DIR:-$PROJECT_ROOT/.local/logs/g2_shared_router_joint_lm_code_kd_conversation_chain_${FIX_TAG}}"
mkdir -p "$LOG_ROOT"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

MODE="${1:-${INIT_MODE:-logits}}"
case "$MODE" in
    logits|logits_hidden|logits_hidden_router) ;;
    *)
        echo "usage: $0 [logits|logits_hidden|logits_hidden_router]" >&2
        exit 2
        ;;
esac
SAFE_MODE="${MODE//_/-}"
RUN_ONLY_STAGE="${RUN_ONLY_STAGE:-all}"
case "$RUN_ONLY_STAGE" in
    all|code_init|code|conversation_init|conversation) ;;
    *)
        echo "ERROR: RUN_ONLY_STAGE must be all, code_init, code, conversation_init, or conversation" >&2
        exit 2
        ;;
esac

WIKI_SOURCE_STEPS="${WIKI_SOURCE_STEPS:-1800}"
CODE_INIT_STEPS="${CODE_INIT_STEPS:-600}"
CODE_INIT_MB="${CODE_INIT_MB:-48}"
CODE_STEPS="${CODE_STEPS:-1800}"
CODE_MB="${CODE_MB:-96}"
CONV_INIT_STEPS="${CONV_INIT_STEPS:-600}"
CONV_INIT_MB="${CONV_INIT_MB:-36}"
CONV_STEPS="${CONV_STEPS:-1800}"
CONV_MB="${CONV_MB:-96}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
PROBE_INTERVAL="${PROBE_INTERVAL:-50}"
PROBE_ITERS="${PROBE_ITERS:-25}"
PROBE_MB="${PROBE_MB:-32}"

WIKI_SOURCE="${WIKI_SOURCE:-$G2_ROOT/wiki/g2-top4-e8-ffn352-r256-wiki-shared-router-qkvo-mha-a100-bf16-mb72-1800}"
CODE_INIT_ID="${CODE_INIT_RUN_ID:-g2-shared-router-e8to16-code-expert-init-${SAFE_MODE}-wiki-distill-qkvo-mha-a100-bf16-mb${CODE_INIT_MB}-${CODE_INIT_STEPS}-${FIX_TAG}}"
CODE_INIT_OUT="${CODE_INIT_WEIGHTS:-$G2_ROOT/code/shared_router_expansion_distill_init/$CODE_INIT_ID}"
CODE_ID="${CODE_RUN_ID:-g2-shared-router-code-wiki-joint-lm-allrouter-${SAFE_MODE}-init-mb${CODE_MB}-${CODE_STEPS}-${FIX_TAG}}"
CODE_OUT="${CODE_WEIGHTS:-$G2_ROOT/code/shared_router_joint_lm_replay/$CODE_ID}"
CONV_INIT_ID="${CONV_INIT_RUN_ID:-g2-shared-router-e16to24-conversation-expert-init-${SAFE_MODE}-wikicode-distill-qkvo-mha-a100-bf16-mb${CONV_INIT_MB}-${CONV_INIT_STEPS}-${FIX_TAG}}"
CONV_INIT_OUT="${CONV_INIT_WEIGHTS:-$G2_ROOT/conversation/shared_router_expansion_distill_init_joint_code/$CONV_INIT_ID}"
CONV_ID="${CONV_RUN_ID:-g2-shared-router-conv-wikicode-joint-lm-allrouter-112-${SAFE_MODE}-init-mb${CONV_MB}-${CONV_STEPS}-${FIX_TAG}}"
CONV_OUT="${CONV_WEIGHTS:-$G2_ROOT/conversation/shared_router_joint_lm_replay/$CONV_ID}"

CODE_INIT_OFFSET="$WIKI_SOURCE_STEPS"
CODE_OFFSET="$((WIKI_SOURCE_STEPS + CODE_INIT_STEPS))"
CONV_INIT_OFFSET="$((CODE_OFFSET + CODE_STEPS))"
CONV_OFFSET="$((CONV_INIT_OFFSET + CONV_INIT_STEPS))"

is_positive_uint() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}
done_at() {
    local output="$1"
    local expected="$2"
    if [ ! -f "$output/latest_checkpointed_iteration.txt" ]; then
        return 1
    fi
    [ "$(tr -d "[:space:]" < "$output/latest_checkpointed_iteration.txt")" = "$expected" ]
}
require_done() {
    local label="$1"
    local output="$2"
    local expected="$3"
    if ! done_at "$output" "$expected"; then
        echo "ERROR: $label checkpoint is missing or incomplete: $output (expected $expected)" >&2
        exit 1
    fi
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
    require_done "$name" "$output" "$expected"
    echo "[END] $name completed at step $expected $(date -Is)"
}

numeric_values=(
    "$WIKI_SOURCE_STEPS" "$CODE_INIT_STEPS" "$CODE_INIT_MB"
    "$CODE_STEPS" "$CODE_MB" "$CONV_INIT_STEPS" "$CONV_INIT_MB"
    "$CONV_STEPS" "$CONV_MB" "$GLOBAL_BATCH_SIZE"
    "$PROBE_INTERVAL" "$PROBE_ITERS" "$PROBE_MB"
)
for value in "${numeric_values[@]}"; do
    is_positive_uint "$value" || {
        echo "ERROR: all step, batch, and probe values must be positive integers: $value" >&2
        exit 2
    }
done

echo "[CHAIN] shared-router canonical continual 1-phase, ramp off"
echo "[FIX] $FIX_TAG (fresh output/log namespace)"
echo "[STAGE 0] Wiki e8 -> Code slots e16 KD init, mode=$MODE, steps=$CODE_INIT_STEPS"
echo "[STAGE 1] Code LM(new FFN/QKVO + all router) + Wiki LM(all router only), steps=$CODE_STEPS"
echo "[STAGE 2] Code e16 -> Conversation slots e24 KD init on Wiki+Code, steps=$CONV_INIT_STEPS"
echo "[STAGE 3] Conversation LM(new FFN/QKVO + all router) + WikiCode LM(all router only), steps=$CONV_STEPS"
echo "[RATIO] Stage 3 Wiki:Code:Conversation = 1:1:2"
echo "[RAMP] off for every canonical stage"
echo "[OUTPUT 0] $CODE_INIT_OUT"
echo "[OUTPUT 1] $CODE_OUT"
echo "[OUTPUT 2] $CONV_INIT_OUT"
echo "[OUTPUT 3] $CONV_OUT"

if [ "${PLAN_ONLY:-0}" = "1" ]; then
    exit 0
fi

if [ "$RUN_ONLY_STAGE" = "all" ] || [ "$RUN_ONLY_STAGE" = "code_init" ]; then
    code_init_cmd=(
        env
        "STAGE1_WEIGHTS_DIR=$WIKI_SOURCE"
        "SOURCE_REQUIRED_ITERS=$WIKI_SOURCE_STEPS"
        "TRAIN_ITERS=$CODE_INIT_STEPS"
        "MICRO_BATCH_SIZE=$CODE_INIT_MB"
        "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
        "SAVE_INTERVAL=$CODE_INIT_STEPS"
        "NO_SAVE_OPTIM=1"
        "PROBE_EVAL_INTERVAL=$PROBE_INTERVAL"
        "SECONDARY_PROBE_EVAL_INTERVAL=$PROBE_INTERVAL"
        "PROBE_EVAL_ITERS=$PROBE_ITERS"
        "SECONDARY_PROBE_EVAL_ITERS=$PROBE_ITERS"
        "PROBE_MICRO_BATCH_SIZE=$PROBE_MB"
        "PROBE_STEP_OFFSET=$CODE_INIT_OFFSET"
        "SECONDARY_PROBE_STEP_OFFSET=$CODE_INIT_OFFSET"
        "WANDB_STEP_OFFSET=$CODE_INIT_OFFSET"
        "RUN_ID=$CODE_INIT_ID"
        "TRAIN_WEIGHTS=$CODE_INIT_OUT"
        "MASTER_PORT=${CODE_INIT_MASTER_PORT:-29971}"
        bash "$A100_SCRIPTS_DIR/run_g2_shared_router_code_expert_distill_init_mha.sh" "$MODE"
    )
    run_stage code_kd_init "$CODE_INIT_OUT" "$CODE_INIT_STEPS"         "$LOG_ROOT/stage0_code_kd_init.log" "${code_init_cmd[@]}"
fi
if [ "$RUN_ONLY_STAGE" = "code_init" ]; then
    exit 0
fi

if [ "$RUN_ONLY_STAGE" = "all" ] || [ "$RUN_ONLY_STAGE" = "code" ]; then
    require_done "Code KD-init" "$CODE_INIT_OUT" "$CODE_INIT_STEPS"
    code_cmd=(
        env
        "STAGE1_WEIGHTS_DIR=$WIKI_SOURCE"
        "SOURCE_REQUIRED_ITERS=$WIKI_SOURCE_STEPS"
        "RESUME_FROM_WEIGHTS=$CODE_INIT_OUT"
        "DISTILL_SOURCE_REQUIRED_ITERS=$CODE_INIT_STEPS"
        "DISTILL_SOURCE_ITERS=$CODE_INIT_STEPS"
        "TRAIN_ITERS=$CODE_STEPS"
        "MICRO_BATCH_SIZE=$CODE_MB"
        "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
        SAVE_INTERVAL=600
        "PROBE_EVAL_INTERVAL=$PROBE_INTERVAL"
        "SECONDARY_PROBE_EVAL_INTERVAL=$PROBE_INTERVAL"
        "PROBE_EVAL_ITERS=$PROBE_ITERS"
        "SECONDARY_PROBE_EVAL_ITERS=$PROBE_ITERS"
        "PROBE_MICRO_BATCH_SIZE=$PROBE_MB"
        "PROBE_STEP_OFFSET=$CODE_OFFSET"
        "SECONDARY_PROBE_STEP_OFFSET=$CODE_OFFSET"
        "WANDB_STEP_OFFSET=$CODE_OFFSET"
        "RUN_ID=$CODE_ID"
        "TRAIN_WEIGHTS=$CODE_OUT"
        "MASTER_PORT=${CODE_MASTER_PORT:-29972}"
        bash "$A100_SCRIPTS_DIR/run_g2_shared_router_code_wiki_joint_lm_allrouter_mha.sh" "$MODE"
    )
    run_stage code_1phase "$CODE_OUT" "$CODE_STEPS"         "$LOG_ROOT/stage1_code_wiki_joint.log" "${code_cmd[@]}"
fi
if [ "$RUN_ONLY_STAGE" = "code" ]; then
    exit 0
fi

if [ "$RUN_ONLY_STAGE" = "all" ] || [ "$RUN_ONLY_STAGE" = "conversation_init" ]; then
    require_done "Code 1-phase" "$CODE_OUT" "$CODE_STEPS"
    conv_init_cmd=(
        env
        "STAGE1_WEIGHTS_DIR=$CODE_OUT"
        "SOURCE_REQUIRED_ITERS=$CODE_STEPS"
        "CODE_SOURCE_MODE=$MODE"
        "CODE_SOURCE_ITERS=$CODE_STEPS"
        "CODE_SOURCE_MB=$CODE_MB"
        "TRAIN_ITERS=$CONV_INIT_STEPS"
        "MICRO_BATCH_SIZE=$CONV_INIT_MB"
        "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
        "SAVE_INTERVAL=$CONV_INIT_STEPS"
        "PROBE_EVAL_INTERVAL=$PROBE_INTERVAL"
        "SECONDARY_PROBE_EVAL_INTERVAL=$PROBE_INTERVAL"
        "TERTIARY_PROBE_EVAL_INTERVAL=$PROBE_INTERVAL"
        "PROBE_EVAL_ITERS=$PROBE_ITERS"
        "SECONDARY_PROBE_EVAL_ITERS=$PROBE_ITERS"
        "TERTIARY_PROBE_EVAL_ITERS=$PROBE_ITERS"
        "PROBE_MICRO_BATCH_SIZE=$PROBE_MB"
        "PROBE_STEP_OFFSET=$CONV_INIT_OFFSET"
        "SECONDARY_PROBE_STEP_OFFSET=$CONV_INIT_OFFSET"
        "TERTIARY_PROBE_STEP_OFFSET=$CONV_INIT_OFFSET"
        "WANDB_STEP_OFFSET=$CONV_INIT_OFFSET"
        "RUN_ID=$CONV_INIT_ID"
        "TRAIN_WEIGHTS=$CONV_INIT_OUT"
        "MASTER_PORT=${CONV_INIT_MASTER_PORT:-29973}"
        bash "$A100_SCRIPTS_DIR/run_g2_shared_router_conversation_expert_distill_init_wikicode_mha.sh" "$MODE"
    )
    run_stage conversation_kd_init "$CONV_INIT_OUT" "$CONV_INIT_STEPS"         "$LOG_ROOT/stage2_conversation_wikicode_kd.log" "${conv_init_cmd[@]}"
fi
if [ "$RUN_ONLY_STAGE" = "conversation_init" ]; then
    exit 0
fi

if [ "$RUN_ONLY_STAGE" = "all" ] || [ "$RUN_ONLY_STAGE" = "conversation" ]; then
    require_done "Code 1-phase" "$CODE_OUT" "$CODE_STEPS"
    require_done "Conversation KD-init" "$CONV_INIT_OUT" "$CONV_INIT_STEPS"
    conv_cmd=(
        env
        "STAGE1_WEIGHTS_DIR=$CODE_OUT"
        "SOURCE_REQUIRED_ITERS=$CODE_STEPS"
        "RESUME_FROM_WEIGHTS=$CONV_INIT_OUT"
        "DISTILL_SOURCE_REQUIRED_ITERS=$CONV_INIT_STEPS"
        "DISTILL_SOURCE_ITERS=$CONV_INIT_STEPS"
        "CODE_SOURCE_MODE=$MODE"
        "CODE_SOURCE_ITERS=$CODE_STEPS"
        "CODE_SOURCE_MB=$CODE_MB"
        "TRAIN_ITERS=$CONV_STEPS"
        "MICRO_BATCH_SIZE=$CONV_MB"
        "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
        SAVE_INTERVAL=600
        "PROBE_EVAL_INTERVAL=$PROBE_INTERVAL"
        "SECONDARY_PROBE_EVAL_INTERVAL=$PROBE_INTERVAL"
        "TERTIARY_PROBE_EVAL_INTERVAL=$PROBE_INTERVAL"
        "PROBE_EVAL_ITERS=$PROBE_ITERS"
        "SECONDARY_PROBE_EVAL_ITERS=$PROBE_ITERS"
        "TERTIARY_PROBE_EVAL_ITERS=$PROBE_ITERS"
        "PROBE_MICRO_BATCH_SIZE=$PROBE_MB"
        "PROBE_STEP_OFFSET=$CONV_OFFSET"
        "SECONDARY_PROBE_STEP_OFFSET=$CONV_OFFSET"
        "TERTIARY_PROBE_STEP_OFFSET=$CONV_OFFSET"
        "WANDB_STEP_OFFSET=$CONV_OFFSET"
        "RUN_ID=$CONV_ID"
        "TRAIN_WEIGHTS=$CONV_OUT"
        "MASTER_PORT=${CONV_MASTER_PORT:-29974}"
        bash "$A100_SCRIPTS_DIR/run_g2_shared_router_conversation_wikicode_joint_lm_allrouter_mha.sh" "$MODE"
    )
    run_stage conversation_1phase "$CONV_OUT" "$CONV_STEPS"         "$LOG_ROOT/stage3_conversation_wikicode_joint.log" "${conv_cmd[@]}"
fi

echo "[DONE] requested shared-router stages completed"
echo "[FINAL] $CONV_OUT"
