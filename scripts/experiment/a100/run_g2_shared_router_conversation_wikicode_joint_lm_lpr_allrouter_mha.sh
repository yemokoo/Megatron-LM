#!/bin/bash
set -euo pipefail

A100_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$A100_SCRIPTS_DIR/../../.." && pwd)"
source "$A100_SCRIPTS_DIR/common.sh"

# LPR-joint variant of the hybrid conversation 1-phase run.  Replay is wiki+code,
# so the old-expert forcing is per task: wiki -> experts 0:8, code -> 8:16
# (JOINT_REPLAY_LPR_TASK_RANGES empty falls back to one group [0,16)).
export TRAIN_ENTRY=pretrain_gpt_lprjoint.py
export JOINT_REPLAY_LPR_COEFF="${JOINT_REPLAY_LPR_COEFF:-0.1}"
export JOINT_REPLAY_LPR_OLD_EXPERTS="${JOINT_REPLAY_LPR_OLD_EXPERTS:-16}"
export JOINT_REPLAY_LPR_TASK_RANGES="${JOINT_REPLAY_LPR_TASK_RANGES:-0:8,8:16}"
export JOINT_REPLAY_LPR_PREFIX_COUNTS="${JOINT_REPLAY_LPR_PREFIX_COUNTS:-1,1}"

MODE="${1:-${MOE_EXPANSION_DISTILL_MODE_SOURCE:-logits}}"
case "$MODE" in
    logits|logits_hidden|logits_hidden_router) ;;
    *)
        echo "usage: $0 [logits|logits_hidden|logits_hidden_router]" >&2
        exit 2
        ;;
esac
SAFE_MODE="${MODE//_/-}"

# Conversation 1-phase update:
#   Conversation LM -> new FFN/QKVO experts 16:24 + router rows 0:24
#   Wiki+Code LM    -> router rows 0:24 only
# Both gradients are accumulated before one optimizer step. Wiki and Code have
# equal weight inside the replay stream, producing Wiki:Code:Conversation 1:1:2.
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

export DISTILL_SOURCE_MB="${DISTILL_SOURCE_MB:-36}"
export DISTILL_SOURCE_ITERS="${DISTILL_SOURCE_ITERS:-600}"
export DISTILL_SOURCE_RUN_ID="${DISTILL_SOURCE_RUN_ID:-g2-shared-router-e16to24-conversation-expert-init-${SAFE_MODE}-wikicode-distill-qkvo-mha-a100-bf16-mb${DISTILL_SOURCE_MB}-${DISTILL_SOURCE_ITERS}}"
export DISTILL_SOURCE_ROOT="${DISTILL_SOURCE_ROOT:-$G2_ROOT/conversation/shared_router_expansion_distill_init_joint_code}"
export RESUME_FROM_WEIGHTS="${RESUME_FROM_WEIGHTS:-$DISTILL_SOURCE_ROOT/$DISTILL_SOURCE_RUN_ID}"
export DISTILL_SOURCE_REQUIRED_ITERS="${DISTILL_SOURCE_REQUIRED_ITERS:-$DISTILL_SOURCE_ITERS}"
export RESUME_LOAD_OPTIM=0
export RESUME_RESET_ITERATION=1

export SOURCE_NUM_EXPERTS=16
export NUM_EXPERTS=24
export MOE_ROUTER_TOPK=4
export MOE_FFN_HIDDEN_SIZE=352
export NUM_QUERY_GROUPS=16
export ATTN_LORA_RANK="${ATTN_LORA_RANK:-256}"
export ATTN_LORA_ALPHA="${ATTN_LORA_ALPHA:-$ATTN_LORA_RANK}"
export ATTN_FULL_RANK_LORA_RANK="${ATTN_FULL_RANK_LORA_RANK:-256}"
export ATTN_FULL_RANK_LORA_ALPHA="${ATTN_FULL_RANK_LORA_ALPHA:-$ATTN_FULL_RANK_LORA_RANK}"
export ATTN_FULL_RANK_LORA_TARGETS="${ATTN_FULL_RANK_LORA_TARGETS:-qkvo}"
export ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="${ATTN_FULL_RANK_LORA_ACTIVE_TARGETS:-}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-configs/model/flame-shared-router-hybrid-experts.sh}"

export SHARED_ROUTER_HYBRID_TRAIN_ALL_EXPERTS_AND_ROUTER=0
export SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS=1
export SHARED_ROUTER_HYBRID_FREEZE_PREEXISTING_ONLY=0
export SHARED_ROUTER_HYBRID_PARTIAL_FREEZE_MASK=""
export SHARED_ROUTER_HYBRID_TOPK_WITH_ALL_NEW_EXPERTS=0

export TRAIN_DATASET="${TRAIN_DATASET:-$(dataset_dir_for_task conversation)}"
export DATASET_NAME="${DATASET_NAME:-conversation_train_with_wikicode_joint_replay}"
export DATASET_SOURCE="${DATASET_SOURCE:-Conversation primary LM + Wiki/Code router-only replay LM}"
export MOE_JOINT_REPLAY_LM=1
# 옛 데이터 리플레이 예산 = MoE-LPR 의 라우터 리튠 총 소비량과 동일하게 맞춘다.
#   360 step x 2304 = 829,440 시퀀스 = primary(1800 x 2304)의 0.2
# 0 으로 두면 예전처럼 매 스텝 리플레이 글로벌 배치 1개(=1:1)가 된다.
export MOE_JOINT_REPLAY_TOTAL_SAMPLES="${MOE_JOINT_REPLAY_TOTAL_SAMPLES:-829440}"
export MOE_JOINT_REPLAY_MICRO_BATCH_SIZE="${MOE_JOINT_REPLAY_MICRO_BATCH_SIZE:-0}"

export JOINT_REPLAY_DATASET="${JOINT_REPLAY_DATASET:-$(dataset_dir_for_task wiki)}"
export JOINT_REPLAY_SECONDARY_DATASET="${JOINT_REPLAY_SECONDARY_DATASET:-$(dataset_dir_for_task code)}"
export JOINT_REPLAY_DATA_WEIGHT_MODE=equal_dataset

export ENABLE_OLD_MODEL_KL=0
export OLD_MODEL_KL_COEFF=0.0
export OLD_MODEL_KL_TEMPERATURE=1.0
export MOE_EXPANSION_DISTILL_MODE=none
export ROUTER_MEMORY_KL_COEFF=0.0
export ROUTER_MEMORY_INTERVAL=0
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.01}"
export MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.001}"
# Ramp is an ablation and is deliberately disabled in canonical 1-phase.
export MOE_NEW_EXPERT_LR_RAMP_STEPS=0

export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-96}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-$TRAIN_ITERS}"
export LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-$((TRAIN_ITERS / 10))}"
export LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.01}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-600}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-$TRAIN_ITERS}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export TRAIN_LOG_STEP_TIME_ONLY="${TRAIN_LOG_STEP_TIME_ONLY:-0}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}"
export TERTIARY_PROBE_EVAL_INTERVAL="${TERTIARY_PROBE_EVAL_INTERVAL:-50}"
export PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-25}"
export SECONDARY_PROBE_EVAL_ITERS="${SECONDARY_PROBE_EVAL_ITERS:-25}"
export TERTIARY_PROBE_EVAL_ITERS="${TERTIARY_PROBE_EVAL_ITERS:-25}"
export PROBE_MICRO_BATCH_SIZE="${PROBE_MICRO_BATCH_SIZE:-32}"
export RUN_INITIAL_PROBE_EVAL="${RUN_INITIAL_PROBE_EVAL:-1}"
export RUN_INITIAL_VALID_EVAL="${RUN_INITIAL_VALID_EVAL:-1}"
export PROBE_DATASET="${PROBE_DATASET:-$(probe_dir_for_task conversation)}"
export PROBE_NAME="${PROBE_NAME:-conversation_probe}"
export SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET:-$(probe_dir_for_task wiki)}"
export SECONDARY_PROBE_NAME="${SECONDARY_PROBE_NAME:-wiki_probe}"
export TERTIARY_PROBE_DATASET="${TERTIARY_PROBE_DATASET:-$(probe_dir_for_task code)}"
export TERTIARY_PROBE_NAME="${TERTIARY_PROBE_NAME:-code_probe}"
export PROBE_STEP_OFFSET="${PROBE_STEP_OFFSET:-4800}"
export SECONDARY_PROBE_STEP_OFFSET="${SECONDARY_PROBE_STEP_OFFSET:-$PROBE_STEP_OFFSET}"
export TERTIARY_PROBE_STEP_OFFSET="${TERTIARY_PROBE_STEP_OFFSET:-$PROBE_STEP_OFFSET}"
export WANDB_STEP_OFFSET="${WANDB_STEP_OFFSET:-$PROBE_STEP_OFFSET}"
export DIRECT_LOCAL_SAVE="${DIRECT_LOCAL_SAVE:-1}"
export NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-1}"
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"
export ATTN_LORA_GROUPED_GEMM="${ATTN_LORA_GROUPED_GEMM:-1}"
export MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}"

export RUN_ID="${RUN_ID:-g2-shared-router-conv-wikicode-joint-lm-lpr${JOINT_REPLAY_LPR_COEFF}-allrouter-112-${SAFE_MODE}-init-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$G2_ROOT/conversation/shared_router_joint_lm_replay_lpr/$RUN_ID}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2 shared-router Conversation+WikiCode 1-phase 1:1:2}"

if [ "${PLAN_ONLY:-0}" = "1" ]; then
    echo "[PLAN] shared-router Conversation 1-phase"
    echo "[PLAN] Code source=$STAGE1_WEIGHTS_DIR"
    echo "[PLAN] KD-init source=$RESUME_FROM_WEIGHTS"
    echo "[PLAN] target=$TRAIN_WEIGHTS"
    echo "[PLAN] Conversation gradients=new FFN/QKVO 16:24 + router 0:24"
    echo "[PLAN] Wiki+Code gradients=router 0:24 only; replay equal_dataset"
    echo "[PLAN] effective data ratio=Wiki:Code:Conversation 1:1:2; ramp=off"
    exit 0
fi

checkpoint_specs=(
    "Code source:$STAGE1_WEIGHTS_DIR:$SOURCE_REQUIRED_ITERS"
    "Conversation KD:$RESUME_FROM_WEIGHTS:$DISTILL_SOURCE_REQUIRED_ITERS"
)
for checkpoint_spec in "${checkpoint_specs[@]}"; do
    label="${checkpoint_spec%%:*}"
    rest="${checkpoint_spec#*:}"
    checkpoint_dir="${rest%%:*}"
    expected_step="${rest##*:}"
    tracker="$checkpoint_dir/latest_checkpointed_iteration.txt"
    if [ ! -f "$tracker" ]; then
        echo "ERROR: missing $label checkpoint: $tracker" >&2
        exit 1
    fi
    actual_step="$(tr -d "[:space:]" < "$tracker")"
    if [ "$actual_step" != "$expected_step" ]; then
        echo "ERROR: expected $label step $expected_step, got $actual_step" >&2
        exit 1
    fi
done
dataset_specs=(
    "Conversation:$TRAIN_DATASET"
    "Wiki replay:$JOINT_REPLAY_DATASET"
    "Code replay:$JOINT_REPLAY_SECONDARY_DATASET"
)
for dataset_spec in "${dataset_specs[@]}"; do
    label="${dataset_spec%%:*}"
    dataset_dir="${dataset_spec#*:}"
    if ! compgen -G "$dataset_dir/*.bin" >/dev/null || ! compgen -G "$dataset_dir/*.idx" >/dev/null; then
        echo "ERROR: missing $label .bin/.idx files: $dataset_dir" >&2
        exit 1
    fi
done

echo "[CONFIG] G2 shared-router Conversation 1-phase"
echo "[CONFIG] KD-init source=$RESUME_FROM_WEIGHTS (24 experts)"
echo "[CONFIG] Conversation gradient=new FFN/QKVO 16:24 + all router rows"
echo "[CONFIG] Wiki+Code gradient=all router rows only"
echo "[CONFIG] replay weighting=Wiki:Code equal; total ratio=1:1:2"
echo "[CONFIG] new-expert LR ramp=off"
echo "[CONFIG] target=$TRAIN_WEIGHTS"

exec bash "$PROJECT_ROOT/scripts/experiment/continual_code_from_wiki_shared_router_hybrid_expand_local_bf16_lprjoint.sh"
