#!/bin/bash
set -euo pipefail

A100_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$A100_SCRIPTS_DIR/../../.." && pwd)"
source "$A100_SCRIPTS_DIR/common.sh"

MODE="${1:-${MOE_EXPANSION_DISTILL_MODE_SOURCE:-logits}}"
case "$MODE" in
    logits|logits_hidden|logits_hidden_router) ;;
    *)
        echo "usage: $0 [logits|logits_hidden|logits_hidden_router]" >&2
        exit 2
        ;;
esac
SAFE_MODE="${MODE//_/-}"

# Pre-Conversation initialization: expand the 16-expert Code-stage model to 24
# experts, then distill on an equal-dataset Wiki+Code mixture. FFN and QKVO LoRA
# expert rows 16:24 share the same per-layer router rows.
export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export G2_ROOT="${G2_ROOT:-$LOCAL_WEIGHTS/a100/mha/g2-checkpoints}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-$PROJECT_ROOT/.local/models/pythia-12b-tokenizer}"

export CODE_SOURCE_MODE="${CODE_SOURCE_MODE:-$MODE}"
CODE_SOURCE_SAFE_MODE="${CODE_SOURCE_MODE//_/-}"
export CODE_SOURCE_MB="${CODE_SOURCE_MB:-96}"
export CODE_SOURCE_ITERS="${CODE_SOURCE_ITERS:-1800}"
export CODE_SOURCE_RUN_ID="${CODE_SOURCE_RUN_ID:-g2-shared-router-code-wiki-joint-lm-allrouter-${CODE_SOURCE_SAFE_MODE}-init-mb${CODE_SOURCE_MB}-${CODE_SOURCE_ITERS}}"
export CODE_SOURCE_ROOT="${CODE_SOURCE_ROOT:-$G2_ROOT/code/shared_router_joint_lm_replay}"
export STAGE1_WEIGHTS_DIR="${STAGE1_WEIGHTS_DIR:-$CODE_SOURCE_ROOT/$CODE_SOURCE_RUN_ID}"
export SOURCE_REQUIRED_ITERS="${SOURCE_REQUIRED_ITERS:-$CODE_SOURCE_ITERS}"
export RESUME_FROM_WEIGHTS=""

export SOURCE_NUM_EXPERTS=16
export NUM_EXPERTS=24
export MOE_ROUTER_TOPK=4
# DoF 스윕: expert 폭을 env 로 개방 (미지정 시 기존 352 그대로)
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export NUM_QUERY_GROUPS=16
export ATTN_LORA_RANK="${ATTN_LORA_RANK:-256}"
export ATTN_LORA_ALPHA="${ATTN_LORA_ALPHA:-$ATTN_LORA_RANK}"
export ATTN_FULL_RANK_LORA_RANK="${ATTN_FULL_RANK_LORA_RANK:-256}"
export ATTN_FULL_RANK_LORA_ALPHA="${ATTN_FULL_RANK_LORA_ALPHA:-$ATTN_FULL_RANK_LORA_RANK}"
export ATTN_FULL_RANK_LORA_TARGETS="${ATTN_FULL_RANK_LORA_TARGETS:-qkvo}"
export ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="${ATTN_FULL_RANK_LORA_ACTIVE_TARGETS:-}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-configs/model/flame-shared-router-hybrid-experts.sh}"

export SHARED_ROUTER_HYBRID_TRAIN_ALL_EXPERTS_AND_ROUTER=0
export SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS=0
export SHARED_ROUTER_HYBRID_FREEZE_PREEXISTING_ONLY=0
export SHARED_ROUTER_HYBRID_PARTIAL_FREEZE_MASK=""
export SHARED_ROUTER_HYBRID_TOPK_WITH_ALL_NEW_EXPERTS=0

export TRAIN_DATASET="${TRAIN_DATASET:-$(dataset_dir_for_task wiki)}"
export TRAIN_DATASET_SECONDARY="${TRAIN_DATASET_SECONDARY:-$(dataset_dir_for_task code)}"
export TRAIN_DATA_WEIGHT_MODE=equal_dataset
export DATASET_NAME="${DATASET_NAME:-wiki_code_equal_shared_router_expansion_distill}"
export DATASET_SOURCE="${DATASET_SOURCE:-Wiki + Code equal-dataset mixture for pre-Conversation shared-router initialization}"

export ENABLE_OLD_MODEL_KL=1
export OLD_MODEL_KL_COEFF="${OLD_MODEL_KL_COEFF:-1.0}"
export OLD_MODEL_KL_TEMPERATURE="${OLD_MODEL_KL_TEMPERATURE:-1.0}"
export MOE_EXPANSION_DISTILL_MODE="$MODE"
export MOE_EXPANSION_DISTILL_LM_LOSS_COEFF=0.0
export MOE_EXPANSION_DISTILL_HIDDEN_MSE_COEFF="${MOE_EXPANSION_DISTILL_HIDDEN_MSE_COEFF:-1.0}"
export MOE_EXPANSION_DISTILL_ROUTER_KL_COEFF="${MOE_EXPANSION_DISTILL_ROUTER_KL_COEFF:-1.0}"
export MOE_EXPANSION_DISTILL_HIDDEN_LAYERS="${MOE_EXPANSION_DISTILL_HIDDEN_LAYERS:-all}"
export MOE_AUX_LOSS_COEFF=0.0
export MOE_Z_LOSS_COEFF=0.0
export MOE_NEW_EXPERT_LR_RAMP_STEPS=0

export TRAIN_ITERS="${TRAIN_ITERS:-600}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-36}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-$TRAIN_ITERS}"
export LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-$((TRAIN_ITERS / 10))}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-$TRAIN_ITERS}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-$TRAIN_ITERS}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}"
export TERTIARY_PROBE_EVAL_INTERVAL="${TERTIARY_PROBE_EVAL_INTERVAL:-50}"
export PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-25}"
export SECONDARY_PROBE_EVAL_ITERS="${SECONDARY_PROBE_EVAL_ITERS:-25}"
export TERTIARY_PROBE_EVAL_ITERS="${TERTIARY_PROBE_EVAL_ITERS:-25}"
export PROBE_MICRO_BATCH_SIZE="${PROBE_MICRO_BATCH_SIZE:-32}"
export RUN_INITIAL_PROBE_EVAL="${RUN_INITIAL_PROBE_EVAL:-1}"
export RUN_INITIAL_VALID_EVAL="${RUN_INITIAL_VALID_EVAL:-1}"
export PROBE_DATASET="${PROBE_DATASET:-$(probe_dir_for_task wiki)}"
export PROBE_NAME="${PROBE_NAME:-wiki_probe}"
export SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET:-$(probe_dir_for_task code)}"
export SECONDARY_PROBE_NAME="${SECONDARY_PROBE_NAME:-code_probe}"
export TERTIARY_PROBE_DATASET="${TERTIARY_PROBE_DATASET:-$(probe_dir_for_task conversation)}"
export TERTIARY_PROBE_NAME="${TERTIARY_PROBE_NAME:-conversation_probe}"
export PROBE_STEP_OFFSET="${PROBE_STEP_OFFSET:-4200}"
export SECONDARY_PROBE_STEP_OFFSET="${SECONDARY_PROBE_STEP_OFFSET:-$PROBE_STEP_OFFSET}"
export TERTIARY_PROBE_STEP_OFFSET="${TERTIARY_PROBE_STEP_OFFSET:-$PROBE_STEP_OFFSET}"
export WANDB_STEP_OFFSET="${WANDB_STEP_OFFSET:-$PROBE_STEP_OFFSET}"
export DIRECT_LOCAL_SAVE="${DIRECT_LOCAL_SAVE:-1}"
export NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-1}"
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"
export ATTN_LORA_GROUPED_GEMM="${ATTN_LORA_GROUPED_GEMM:-1}"
export MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}"

export RUN_ID="${RUN_ID:-g2-shared-router-e16to24-conversation-expert-init-${SAFE_MODE}-wikicode-distill-qkvo-mha-a100-bf16-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$G2_ROOT/conversation/shared_router_expansion_distill_init_joint_code/$RUN_ID}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2 shared-router pre-Conversation ${MODE} KD on Wiki+Code}"

if [ "${PLAN_ONLY:-0}" = "1" ]; then
    echo "[PLAN] shared-router Conversation initialization"
    echo "[PLAN] source=$STAGE1_WEIGHTS_DIR"
    echo "[PLAN] target=$TRAIN_WEIGHTS"
    echo "[PLAN] experts=16->24; FFN + QKVO rank256; shared Top-4 router"
    echo "[PLAN] KD data=Wiki+Code equal_dataset; mode=$MODE; LM coefficient=0"
    echo "[PLAN] trainable=new expert/router rows 16:24; ramp=off"
    exit 0
fi

source_tracker="$STAGE1_WEIGHTS_DIR/latest_checkpointed_iteration.txt"
if [ ! -f "$source_tracker" ]; then
    echo "ERROR: completed 16-expert Code checkpoint is missing: $source_tracker" >&2
    exit 1
fi
source_step="$(tr -d "[:space:]" < "$source_tracker")"
if [ "$source_step" != "$SOURCE_REQUIRED_ITERS" ]; then
    echo "ERROR: expected Code source step $SOURCE_REQUIRED_ITERS, got $source_step" >&2
    exit 1
fi
for dataset_spec in "Wiki KD:$TRAIN_DATASET" "Code KD:$TRAIN_DATASET_SECONDARY"; do
    label="${dataset_spec%%:*}"
    dataset_dir="${dataset_spec#*:}"
    if ! compgen -G "$dataset_dir/*.bin" >/dev/null || ! compgen -G "$dataset_dir/*.idx" >/dev/null; then
        echo "ERROR: missing $label .bin/.idx files: $dataset_dir" >&2
        exit 1
    fi
done

echo "[CONFIG] G2 shared-router pre-Conversation KD initialization"
echo "[CONFIG] source=$STAGE1_WEIGHTS_DIR (16 experts, step $source_step)"
echo "[CONFIG] target=$TRAIN_WEIGHTS (24 experts)"
echo "[CONFIG] trainable=new FFN/QKVO experts + new router rows 16:24"
echo "[CONFIG] KD data=Wiki+Code equal_dataset; mode=$MODE; LM coefficient=0"
echo "[CONFIG] new-expert LR ramp=off"

exec bash "$PROJECT_ROOT/scripts/experiment/continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh"
