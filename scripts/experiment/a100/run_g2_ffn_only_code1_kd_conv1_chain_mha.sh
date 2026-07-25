#!/bin/bash
set -euo pipefail

CHAIN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$CHAIN_SCRIPT_DIR/../../.." && pwd)"
source "$CHAIN_SCRIPT_DIR/common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# The G2 runners use a Hugging Face tokenizer during Megatron startup.  This
# local snapshot keeps the default offline chain independent of the shared HF
# cache (which is not guaranteed to persist on every host).
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-$PROJECT_ROOT/.local/models/pythia-12b-tokenizer}"

if [ ! -f "$TOKENIZER_MODEL/tokenizer.json" ] || [ ! -f "$TOKENIZER_MODEL/config.json" ]; then
    echo "[ERROR] offline tokenizer snapshot is incomplete: $TOKENIZER_MODEL" >&2
    exit 1
fi

CODE_STEPS="${CODE_STEPS:-1800}"
KD_STEPS="${KD_STEPS:-1800}"
CONV_STEPS="${CONV_STEPS:-1800}"
CODE_MB="${CODE_MB:-48}"
KD_MB="${KD_MB:-36}"
CONV_MB="${CONV_MB:-48}"

G2_ROOT="${G2_ROOT:-$PROJECT_ROOT/.local/weights/a100/mha/g2-checkpoints}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/.local/logs/g2_code1_kd_conv1_chain}"
mkdir -p "$LOG_DIR"

CODE_RUN_ID="${CODE_RUN_ID:-g2-ffn-only-interleaved-code1-router1-from-logits-wiki-distill-mb${CODE_MB}-code${CODE_STEPS}}"
CODE_WEIGHTS="${CODE_WEIGHTS:-$G2_ROOT/code/interleaved_router_finetune_distill_init/$CODE_RUN_ID}"
CODE_EXPECTED_STEP=$((CODE_STEPS * 2))

KD_RUN_ID="${KD_RUN_ID:-g2-ffn-only-e16to24-conversation-expert-init-logits-wikicode-distill-mha-a100-bf16-mb${KD_MB}-${KD_STEPS}}"
KD_WEIGHTS="${KD_WEIGHTS:-$G2_ROOT/conversation/expansion_distill_init/$KD_RUN_ID}"
KD_EXPECTED_STEP="$KD_STEPS"

CONV_RUN_ID="${CONV_RUN_ID:-g2-ffn-only-interleaved-conv1-router1-wikicodeconv-from-kd-mb${CONV_MB}-conv${CONV_STEPS}}"
CONV_WEIGHTS="${CONV_WEIGHTS:-$G2_ROOT/conversation/interleaved_router_finetune_from_kd/$CONV_RUN_ID}"
CONV_EXPECTED_STEP=$((CONV_STEPS * 2))

done_at() {
    local run_dir="$1"
    local expected="$2"
    local tracker="$run_dir/latest_checkpointed_iteration.txt"
    [ -f "$tracker" ] && [ "$(tr -d '\n\r[:space:]' < "$tracker")" = "$expected" ]
}

run_stage() {
    local label="$1"
    local output_dir="$2"
    local expected_step="$3"
    local log_file="$4"
    shift 4

    if done_at "$output_dir" "$expected_step"; then
        echo "[SKIP] $label already complete at iteration $expected_step"
        echo "       $output_dir"
        return 0
    fi

    echo
    echo "################################################################"
    echo "### START $label"
    echo "### output=$output_dir"
    echo "### expected_iteration=$expected_step"
    echo "### $(date -Is)"
    echo "################################################################"

    set +e
    "$@" 2>&1 | tee "$log_file"
    local command_status=${PIPESTATUS[0]}
    set -e

    if done_at "$output_dir" "$expected_step"; then
        echo "[DONE] $label reached iteration $expected_step"
        return 0
    fi

    echo "[ERROR] $label did not reach iteration $expected_step (exit=$command_status)" >&2
    echo "        output=$output_dir" >&2
    echo "        log=$log_file" >&2
    if [ "$command_status" -eq 0 ]; then
        command_status=1
    fi
    exit "$command_status"
}

echo "[CHAIN] Code1/router1 -> Wiki+Code output KD -> Conversation1/router1"
echo "[CHAIN] Stage 1: Code=$CODE_STEPS, router=$CODE_STEPS, output iteration=$CODE_EXPECTED_STEP"
echo "[CHAIN] Stage 2: Wiki+Code logits KD=$KD_STEPS, 16->24 experts"
echo "[CHAIN] Stage 3: Conversation=$CONV_STEPS, Wiki+Code+Conversation router=$CONV_STEPS"
echo "[CHAIN] Code output=$CODE_WEIGHTS"
echo "[CHAIN] KD output=$KD_WEIGHTS"
echo "[CHAIN] Conversation output=$CONV_WEIGHTS"

run_stage \
    "Stage 1 / Code1-router1" \
    "$CODE_WEIGHTS" \
    "$CODE_EXPECTED_STEP" \
    "$LOG_DIR/stage1_code1_router1.log" \
    env \
        MICRO_BATCH_SIZE="$CODE_MB" \
        MOE_INTERLEAVE_CODE_TOTAL_STEPS="$CODE_STEPS" \
        RUN_ID="$CODE_RUN_ID" \
        TRAIN_WEIGHTS="$CODE_WEIGHTS" \
        MASTER_PORT="${CODE_MASTER_PORT:-29961}" \
        bash "$CHAIN_SCRIPT_DIR/run_g2_ffn_only_interleaved_code1_router1_from_distill_init_mha.sh"

run_stage \
    "Stage 2 / Wiki+Code output-logits KD 16->24" \
    "$KD_WEIGHTS" \
    "$KD_EXPECTED_STEP" \
    "$LOG_DIR/stage2_wikicode_kd.log" \
    env \
        MICRO_BATCH_SIZE="$KD_MB" \
        TRAIN_ITERS="$KD_STEPS" \
        SOURCE_WEIGHTS_DIR="$CODE_WEIGHTS" \
        SOURCE_REQUIRED_ITERS="$CODE_EXPECTED_STEP" \
        RUN_ID="$KD_RUN_ID" \
        TRAIN_WEIGHTS="$KD_WEIGHTS" \
        RESUME_CONTINUAL_FROM_TRAIN_WEIGHTS=1 \
        MASTER_PORT="${KD_MASTER_PORT:-29962}" \
        bash "$CHAIN_SCRIPT_DIR/run_g2_ffn_only_conversation_expert_distill_init_wikicode_mha.sh"

run_stage \
    "Stage 3 / Conversation1-router1 Wiki+Code+Conversation" \
    "$CONV_WEIGHTS" \
    "$CONV_EXPECTED_STEP" \
    "$LOG_DIR/stage3_conv1_router1.log" \
    env \
        MICRO_BATCH_SIZE="$CONV_MB" \
        MOE_INTERLEAVE_CODE_TOTAL_STEPS="$CONV_STEPS" \
        KD_SOURCE_ITERS="$KD_STEPS" \
        SOURCE_WEIGHTS_DIR="$KD_WEIGHTS" \
        RUN_ID="$CONV_RUN_ID" \
        TRAIN_WEIGHTS="$CONV_WEIGHTS" \
        MASTER_PORT="${CONV_MASTER_PORT:-29963}" \
        bash "$CHAIN_SCRIPT_DIR/run_g2_ffn_only_interleaved_conversation_router_from_kd_mha.sh"

echo
echo "[ALL DONE] full chain completed at $(date -Is)"
echo "[ALL DONE] final checkpoint=$CONV_WEIGHTS"
echo "[ALL DONE] final iteration=$CONV_EXPECTED_STEP"
