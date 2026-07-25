#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
A100_SCRIPTS_DIR="$SCRIPT_DIR"
source "$SCRIPT_DIR/common.sh"

# Stage 3:
#   Conversation LM chunk -> Wiki+Code+Conversation router-only chunk.
# Defaults remain 1/1; callers may select 10/10 through environment variables.
# The 24-expert source has already initialized experts/router rows 16:24 through
# output-only KD. Conversation LM training is the actual task-learning stage.
export SOURCE_TASK=code
export TARGET_TASK=conversation
export FREEZE_SHARED=1
export TRAIN_ATTENTION_WITH_NEW_EXPERTS=0
export FREEZE_DENSE_ATTENTION_LORA_WITH_NEW_EXPERTS=1
export TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY=1
export LOAD_EXPANDED_SOURCE=1

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export G2_ROOT="${G2_ROOT:-$LOCAL_WEIGHTS/a100/mha/g2-checkpoints}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

export KD_SOURCE_MB="${KD_SOURCE_MB:-48}"
export KD_SOURCE_ITERS="${KD_SOURCE_ITERS:-1800}"
export KD_SOURCE_RUN_ID="${KD_SOURCE_RUN_ID:-g2-ffn-only-e16to24-conversation-expert-init-logits-wikicode-distill-mha-a100-bf16-mb${KD_SOURCE_MB}-${KD_SOURCE_ITERS}}"
export SOURCE_WEIGHTS_DIR="${SOURCE_WEIGHTS_DIR:-$G2_ROOT/conversation/expansion_distill_init/$KD_SOURCE_RUN_ID}"
export SOURCE_RUN_ID="$KD_SOURCE_RUN_ID"
export SOURCE_RUN_SUBDIR="a100/mha/g2-checkpoints/conversation/expansion_distill_init"
export SOURCE_REQUIRED_ITERS="$KD_SOURCE_ITERS"

source_tracker="$SOURCE_WEIGHTS_DIR/latest_checkpointed_iteration.txt"
if [ ! -f "$source_tracker" ]; then
    echo "[ERROR] pre-Conversation Wiki+Code KD checkpoint is missing: $source_tracker" >&2
    exit 1
fi
source_step="$(tr -d '\n\r[:space:]' < "$source_tracker")"
if [ "$source_step" != "$KD_SOURCE_ITERS" ]; then
    echo "[ERROR] expected KD source iteration $KD_SOURCE_ITERS, got $source_step" >&2
    echo "        $SOURCE_WEIGHTS_DIR" >&2
    exit 1
fi

# Freeze boundary remains 16: experts/router rows 0:16 are old Wiki+Code
# capacity; KD-initialized rows 16:24 remain trainable for Conversation.
export SOURCE_NUM_EXPERTS=16
export NUM_EXPERTS=24
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-scripts/experiment/a100/flame-moe-bf16-no-shared.sh}"

export MOE_INTERLEAVE_CODE_TOTAL_STEPS="${MOE_INTERLEAVE_CODE_TOTAL_STEPS:-1800}"
export MOE_INTERLEAVE_CODE_STEPS="${MOE_INTERLEAVE_CODE_STEPS:-1}"
export MOE_INTERLEAVE_ROUTER_STEPS="${MOE_INTERLEAVE_ROUTER_STEPS:-1}"
export MOE_INTERLEAVE_ROUTER_AFTER_FINAL="${MOE_INTERLEAVE_ROUTER_AFTER_FINAL:-1}"
if [ "$MOE_INTERLEAVE_CODE_STEPS" -le 0 ] || [ "$MOE_INTERLEAVE_ROUTER_STEPS" -le 0 ]; then
    echo "[ERROR] interleave Code/Router step counts must be positive" >&2
    exit 1
fi
num_code_chunks=$(((MOE_INTERLEAVE_CODE_TOTAL_STEPS + MOE_INTERLEAVE_CODE_STEPS - 1) / MOE_INTERLEAVE_CODE_STEPS))
num_router_chunks="$num_code_chunks"
if [ "$MOE_INTERLEAVE_ROUTER_AFTER_FINAL" != "1" ]; then
    num_router_chunks=$((num_router_chunks - 1))
fi
router_total_steps=$((num_router_chunks * MOE_INTERLEAVE_ROUTER_STEPS))
export TRAIN_ITERS=$((MOE_INTERLEAVE_CODE_TOTAL_STEPS + router_total_steps))

export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-96}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
batch_unit=$((MICRO_BATCH_SIZE * NPROC_PER_NODE))
if [ $((GLOBAL_BATCH_SIZE % batch_unit)) -ne 0 ]; then
    echo "[ERROR] global batch $GLOBAL_BATCH_SIZE must be divisible by mb*gpus=$batch_unit" >&2
    exit 1
fi
export NUM_MICROBATCHES=$((GLOBAL_BATCH_SIZE / batch_unit))

export LR="${LR:-3e-4}"
export MIN_LR="${MIN_LR:-3e-5}"
export MOE_INTERLEAVE_CODE_LR="${MOE_INTERLEAVE_CODE_LR:-$LR}"
export MOE_INTERLEAVE_CODE_MIN_LR="${MOE_INTERLEAVE_CODE_MIN_LR:-$MIN_LR}"
export MOE_INTERLEAVE_ROUTER_LR="${MOE_INTERLEAVE_ROUTER_LR:-$LR}"
export MOE_INTERLEAVE_ROUTER_MIN_LR="${MOE_INTERLEAVE_ROUTER_MIN_LR:-$MIN_LR}"
export MOE_INTERLEAVE_CODE_AUX_LOSS_COEFF="${MOE_INTERLEAVE_CODE_AUX_LOSS_COEFF:-0.01}"
export MOE_INTERLEAVE_CODE_Z_LOSS_COEFF="${MOE_INTERLEAVE_CODE_Z_LOSS_COEFF:-0.001}"
export MOE_INTERLEAVE_ROUTER_AUX_LOSS_COEFF=0.0
export MOE_INTERLEAVE_ROUTER_Z_LOSS_COEFF=0.0

# Conversation is the LM/task-learning stream. Router-only steps use the
# equal-dataset Wiki+Code+Conversation mixture. It is exactly 1:1:1 even when
# tasks have different shard counts; the router phase totals 1,800 steps.
export TRAIN_DATASET="${TRAIN_DATASET:-$(dataset_dir_for_task conversation)}"
export INTERLEAVE_WIKI_DATASET="${INTERLEAVE_WIKI_DATASET:-$(dataset_dir_for_task wiki)}"
export INTERLEAVE_SECONDARY_DATASET="${INTERLEAVE_SECONDARY_DATASET:-$(dataset_dir_for_task code)}"
export INTERLEAVE_ROUTER_DATA_WEIGHT_MODE=equal_dataset
export DATASET_NAME="${DATASET_NAME:-conversation_lm_router_wikicodeconv_equal}"
export DATASET_SOURCE="${DATASET_SOURCE:-Conversation LM; router-only equal-dataset Wiki + Code + Conversation}"

for dataset_spec in \
    "Conversation train:$TRAIN_DATASET" \
    "Wiki router:$INTERLEAVE_WIKI_DATASET" \
    "Code router:$INTERLEAVE_SECONDARY_DATASET"; do
    label="${dataset_spec%%:*}"
    dataset_dir="${dataset_spec#*:}"
    if ! compgen -G "$dataset_dir/*.bin" >/dev/null || ! compgen -G "$dataset_dir/*.idx" >/dev/null; then
        echo "[ERROR] missing $label .bin/.idx files: $dataset_dir" >&2
        exit 1
    fi
done

export PROBE_DATASET="${PROBE_DATASET:-$(probe_dir_for_task conversation)}"
export PROBE_NAME="${PROBE_NAME:-conversation_probe}"
export SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET:-$(probe_dir_for_task wiki)}"
export SECONDARY_PROBE_NAME="${SECONDARY_PROBE_NAME:-wiki_probe}"
export TERTIARY_PROBE_DATASET="${TERTIARY_PROBE_DATASET:-$(probe_dir_for_task code)}"
export TERTIARY_PROBE_NAME="${TERTIARY_PROBE_NAME:-code_probe}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}"
export TERTIARY_PROBE_EVAL_INTERVAL="${TERTIARY_PROBE_EVAL_INTERVAL:-50}"
export RUN_INITIAL_PROBE_EVAL="${RUN_INITIAL_PROBE_EVAL:-1}"
export RUN_INITIAL_VALID_EVAL="${RUN_INITIAL_VALID_EVAL:-0}"

export ENABLE_OLD_MODEL_KL=0
export OLD_MODEL_KL_COEFF=0.0
export MOE_EXPANSION_DISTILL_MODE=none
export LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND=0

export SAVE_INTERVAL="${SAVE_INTERVAL:-$TRAIN_ITERS}"
export RECOVERY_SAVE_INTERVAL="${RECOVERY_SAVE_INTERVAL:-600}"
export NO_SAVE_OPTIM=1
export EVAL_INTERVAL="${EVAL_INTERVAL:-$TRAIN_ITERS}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"

export STAGE_DIR_NAME="a100/mha/g2-checkpoints/conversation/interleaved_router_finetune_from_kd"
export RUN_ID="${RUN_ID:-g2-ffn-only-interleaved-conv${MOE_INTERLEAVE_CODE_STEPS}-router${MOE_INTERLEAVE_ROUTER_STEPS}-wikicodeconv-from-kd-mb${MICRO_BATCH_SIZE}-conv${MOE_INTERLEAVE_CODE_TOTAL_STEPS}}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$G2_ROOT/conversation/interleaved_router_finetune_from_kd/$RUN_ID}"
export RUN_LOG="${RUN_LOG:-$TRAIN_WEIGHTS/logs/run.log}"
export DIRECT_LOCAL_SAVE="${DIRECT_LOCAL_SAVE:-1}"
export RESUME_CONTINUAL_FROM_TRAIN_WEIGHTS="${RESUME_CONTINUAL_FROM_TRAIN_WEIGHTS:-1}"

export PROBE_STEP_OFFSET="${PROBE_STEP_OFFSET:-$KD_SOURCE_ITERS}"
export SECONDARY_PROBE_STEP_OFFSET="${SECONDARY_PROBE_STEP_OFFSET:-$KD_SOURCE_ITERS}"
export TERTIARY_PROBE_STEP_OFFSET="${TERTIARY_PROBE_STEP_OFFSET:-$KD_SOURCE_ITERS}"
export WANDB_STEP_OFFSET="${WANDB_STEP_OFFSET:-$KD_SOURCE_ITERS}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2 FFN-only Conversation${MOE_INTERLEAVE_CODE_STEPS}/router${MOE_INTERLEAVE_ROUTER_STEPS} from Wiki+Code KD}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$RUN_ID}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_LOG_CHECKPOINTS=0

echo "[CONFIG] source KD=$SOURCE_WEIGHTS_DIR (24 experts, iteration $source_step)"
echo "[CONFIG] output=$TRAIN_WEIGHTS"
echo "[CONFIG] Conversation LM $MOE_INTERLEAVE_CODE_STEPS step(s) / Wiki+Code+Conversation router-only $MOE_INTERLEAVE_ROUTER_STEPS step(s)"
echo "[CONFIG] Conversation steps=$MOE_INTERLEAVE_CODE_TOTAL_STEPS"
echo "[CONFIG] Router steps=$router_total_steps (chunks=$num_router_chunks, total optimizer steps=$TRAIN_ITERS)"
echo "[CONFIG] Router mixture=Wiki:Code:Conversation = 1:1:1"
echo "[CONFIG] trainable in LM phase=new experts/router rows 16:24"
echo "[CONFIG] trainable in router phase=all router rows only"
echo "[CONFIG] mb=$MICRO_BATCH_SIZE gbs=$GLOBAL_BATCH_SIZE microbatches=$NUM_MICROBATCHES"

if [ "${PLAN_ONLY:-0}" = "1" ]; then
    exit 0
fi

exec bash "$A100_SCRIPTS_DIR/run_continual_moe_a100_bf16.sh"
