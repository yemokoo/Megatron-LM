#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
A100_SCRIPTS_DIR="$SCRIPT_DIR"
source "$SCRIPT_DIR/common.sh"

# Pre-Conversation expert initialization:
#   16-expert Wiki+Code teacher -> expand to 24 experts -> output-logit KD on
#   an equal-prefix Wiki+Code mixture. This stage initializes experts/router
#   rows 16:24 only; Conversation task learning happens in the following stage.
export SOURCE_TASK=code
export TARGET_TASK=conversation
export FREEZE_SHARED=1
export TRAIN_ATTENTION_WITH_NEW_EXPERTS=0
export FREEZE_DENSE_ATTENTION_LORA_WITH_NEW_EXPERTS=1
export TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY=1

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export G2_ROOT="${G2_ROOT:-$LOCAL_WEIGHTS/a100/mha/g2-checkpoints}"

# Source = final checkpoint from the Code-1/router-1 run. It contains 1,800
# Code optimizer steps and 1,800 router-finetune steps, hence iteration 3,600.
export SOURCE_RUN_ID="${SOURCE_RUN_ID:-g2-ffn-only-interleaved-code1-router1-from-logits-wiki-distill-mb72-code1800}"
export SOURCE_RUN_SUBDIR="${SOURCE_RUN_SUBDIR:-a100/mha/g2-checkpoints/code/interleaved_router_finetune_distill_init}"
export SOURCE_WEIGHTS_DIR="${SOURCE_WEIGHTS_DIR:-$G2_ROOT/code/interleaved_router_finetune_distill_init/$SOURCE_RUN_ID}"
export SOURCE_REQUIRED_ITERS="${SOURCE_REQUIRED_ITERS:-3600}"

source_tracker="$SOURCE_WEIGHTS_DIR/latest_checkpointed_iteration.txt"
if [ ! -f "$source_tracker" ]; then
    echo "[ERROR] Code-1/router-1 teacher checkpoint is missing: $source_tracker" >&2
    exit 1
fi
source_step="$(tr -d '\n\r[:space:]' < "$source_tracker")"
if [ "$source_step" != "$SOURCE_REQUIRED_ITERS" ]; then
    echo "[ERROR] teacher checkpoint must be at iteration $SOURCE_REQUIRED_ITERS; got $source_step" >&2
    echo "        $SOURCE_WEIGHTS_DIR" >&2
    exit 1
fi

export SOURCE_NUM_EXPERTS=16
export NUM_EXPERTS=24
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-scripts/experiment/a100/flame-moe-bf16-no-shared.sh}"

export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-48}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-600}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-$TRAIN_ITERS}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}"
export TERTIARY_PROBE_EVAL_INTERVAL="${TERTIARY_PROBE_EVAL_INTERVAL:-50}"
export RUN_INITIAL_PROBE_EVAL="${RUN_INITIAL_PROBE_EVAL:-1}"
# The Code-stage teacher has Wiki/Code probes only; it has no conversation
# scalar at step 3600, so the generic three-probe baseline logger would abort
# before the KD run starts.  Stage-2 itself logs all three probes after launch.
export LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND="${LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND:-0}"

# Both old-task datasets receive equal total weight regardless of shard count.
# No Conversation samples are used in this initialization stage.
export TRAIN_DATASET="${TRAIN_DATASET:-$(dataset_dir_for_task wiki)}"
export TRAIN_DATASET_SECONDARY="${TRAIN_DATASET_SECONDARY:-$(dataset_dir_for_task code)}"
export TRAIN_DATA_WEIGHT_MODE=equal_dataset
export DATASET_NAME="${DATASET_NAME:-wiki_code_equal_expansion_distill}"
export DATASET_SOURCE="${DATASET_SOURCE:-Wiki + Code equal-dataset mixture for pre-Conversation expert initialization}"

for dataset_spec in \
    "wiki KD:$TRAIN_DATASET" \
    "code KD:$TRAIN_DATASET_SECONDARY"; do
    label="${dataset_spec%%:*}"
    dataset_dir="${dataset_spec#*:}"
    if ! compgen -G "$dataset_dir/*.bin" >/dev/null || ! compgen -G "$dataset_dir/*.idx" >/dev/null; then
        echo "[ERROR] missing $label .bin/.idx dataset files: $dataset_dir" >&2
        exit 1
    fi
done

export PROBE_DATASET="${PROBE_DATASET:-$(probe_dir_for_task wiki)}"
export PROBE_NAME="${PROBE_NAME:-wiki_probe}"
export SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET:-$(probe_dir_for_task code)}"
export SECONDARY_PROBE_NAME="${SECONDARY_PROBE_NAME:-code_probe}"
export TERTIARY_PROBE_DATASET="${TERTIARY_PROBE_DATASET:-$(probe_dir_for_task conversation)}"
export TERTIARY_PROBE_NAME="${TERTIARY_PROBE_NAME:-conversation_probe}"

# Output-logit KD only. LM, hidden-state, router, auxiliary, and z losses are
# disabled so this stage is initialization rather than task learning.
export ENABLE_OLD_MODEL_KL=1
export OLD_MODEL_KL_COEFF="${OLD_MODEL_KL_COEFF:-1.0}"
export OLD_MODEL_KL_TEMPERATURE="${OLD_MODEL_KL_TEMPERATURE:-1.0}"
export MOE_EXPANSION_DISTILL_MODE=logits
export MOE_EXPANSION_DISTILL_LM_LOSS_COEFF=0.0
export MOE_EXPANSION_DISTILL_HIDDEN_MSE_COEFF=0.0
export MOE_EXPANSION_DISTILL_ROUTER_KL_COEFF=0.0
export MOE_AUX_LOSS_COEFF=0.0
export MOE_Z_LOSS_COEFF=0.0

export STAGE_NAME="${STAGE_NAME:-g2_ffn_only_conversation_expert_logits_init}"
export STAGE_LABEL="${STAGE_LABEL:-g2_ffn_only_conversation_expert_logits_init}"
export STAGE_DIR_NAME="${STAGE_DIR_NAME:-a100/mha/g2-checkpoints/conversation/expansion_distill_init}"
export RUN_ID="${RUN_ID:-g2-ffn-only-e16to24-conversation-expert-init-logits-wikicode-distill-mha-a100-bf16-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$G2_ROOT/conversation/expansion_distill_init/$RUN_ID}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2 FFN-only - Conversation expert init logits KD on Wiki+Code}"

echo "[CONFIG] G2 FFN-only pre-Conversation expert initialization"
echo "[CONFIG] teacher=$SOURCE_WEIGHTS_DIR (16 experts, iteration $source_step)"
echo "[CONFIG] student=24 experts; trainable=new experts/router rows 16:24 only"
echo "[CONFIG] KD data: Wiki=$TRAIN_DATASET"
echo "[CONFIG] KD data: Code=$TRAIN_DATASET_SECONDARY"
echo "[CONFIG] objective=output-logit KL only; steps=$TRAIN_ITERS"
echo "[CONFIG] output=$TRAIN_WEIGHTS"

if [ "${PLAN_ONLY:-0}" = "1" ]; then
    exit 0
fi

exec bash "$A100_SCRIPTS_DIR/run_continual_moe_a100_bf16.sh"
