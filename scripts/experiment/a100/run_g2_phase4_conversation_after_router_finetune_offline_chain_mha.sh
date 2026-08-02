#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
# common.sh sources activate_kt_env.sh, which clobbers SCRIPT_DIR with its own
# location (scripts/miscellaneous). Restore it so the run_guarded_training.sh /
# run_continual_moe_a100_bf16.sh lookups below resolve inside scripts/experiment/a100.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-EleutherAI/pythia-12b}"

export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-600}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1800}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}"
export TERTIARY_PROBE_EVAL_INTERVAL="${TERTIARY_PROBE_EVAL_INTERVAL:-50}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export SEQ_LENGTH="${SEQ_LENGTH:-512}"
export SOURCE_LOGICAL_STEP="${SOURCE_LOGICAL_STEP:-5400}"
export TARGET_LOGICAL_STEP="$((SOURCE_LOGICAL_STEP + TRAIN_ITERS))"
export PAUSE_SECONDS="${PAUSE_SECONDS:-180}"
export DEBUG_TRAINABLE_PARAMS_AND_EXIT="${DEBUG_TRAINABLE_PARAMS_AND_EXIT:-0}"
export RUN_ONLY_STAGE="${RUN_ONLY_STAGE:-all}"
# Required iteration count of the ffn_only source checkpoint. Default 3600 = the
# phase3 router-retuned wiki+code checkpoint. Override to 1800 to start conv from
# the raw code checkpoint (no router finetune) — the "no-router-FT" exp3 variant.
export FFN_SOURCE_REQUIRED_ITERS="${FFN_SOURCE_REQUIRED_ITERS:-3600}"

export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-16}"
export NUM_EXPERTS="${NUM_EXPERTS:-24}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"
export ATTN_LORA_GROUPED_GEMM="${ATTN_LORA_GROUPED_GEMM:-1}"
export MOE_PERMUTE_FUSION="${MOE_PERMUTE_FUSION:-0}"
export MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}"

export ATTN_LORA_RANK="${ATTN_LORA_RANK:-256}"
export ATTN_LORA_ALPHA="${ATTN_LORA_ALPHA:-256}"
export ATTN_FULL_RANK_LORA_RANK="${ATTN_FULL_RANK_LORA_RANK:-256}"
export ATTN_FULL_RANK_LORA_ALPHA="${ATTN_FULL_RANK_LORA_ALPHA:-256}"
export ATTN_FULL_RANK_LORA_TARGETS="${ATTN_FULL_RANK_LORA_TARGETS:-qkvo}"
export ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="${ATTN_FULL_RANK_LORA_ACTIVE_TARGETS:-}"

G2_ROOT="${G2_ROOT:-$PROJECT_ROOT/.local/weights/a100/mha/g2-checkpoints}"
PHASE4_ROOT="${PHASE4_ROOT:-$G2_ROOT/conversation/phase4}"

CONVERSATION_TRAIN="${CONVERSATION_TRAIN:-$PROJECT_ROOT/data/conversation/train}"
CODE_PROBE_DATASET="${CODE_PROBE_DATASET:-$PROJECT_ROOT/data/code/test}"
WIKI_PROBE_DATASET="${WIKI_PROBE_DATASET:-$PROJECT_ROOT/data/wiki/test}"
CONVERSATION_PROBE_DATASET="${CONVERSATION_PROBE_DATASET:-$PROJECT_ROOT/data/conversation/test}"

FFN_SOURCE="${FFN_SOURCE:-$G2_ROOT/code/phase3/g2matched-attn-freeze-phase3-router-only-retune-wikicode-no-reinit-mb96-1800}"
EXP1_FREEZE_WIKI_SOURCE="${EXP1_FREEZE_WIKI_SOURCE:-$G2_ROOT/code/phase3/g2-exp2-phase3-router-only-retune-wikicode-from-new-experts-all-router-no-reinit-mb72-1800}"
EXP2_UNFREEZE_WIKI_SOURCE="${EXP2_UNFREEZE_WIKI_SOURCE:-$G2_ROOT/code/phase3/g2-exp1-phase3-router-only-retune-wikicode-from-all-experts-router-no-reinit-mb72-1800}"

CONVERSATION_TOKEN_COUNT="${CONVERSATION_TOKEN_COUNT:-1403196153}"
REQUIRED_TOKEN_COUNT="$((TRAIN_ITERS * GLOBAL_BATCH_SIZE * SEQ_LENGTH))"
APPROX_EPOCHS="$("$PYTHON_BIN" - <<PY
tokens = int("$REQUIRED_TOKEN_COUNT")
dataset = int("$CONVERSATION_TOKEN_COUNT")
print(f"{tokens / dataset:.2f}")
PY
)"

check_completed_source() {
    local label="$1"
    local run_dir="$2"
    local expected_step="$3"
    local tracker="$run_dir/latest_checkpointed_iteration.txt"

    if [ ! -f "$tracker" ]; then
        echo "[ERROR] missing source checkpoint tracker for $label: $tracker" >&2
        exit 1
    fi

    local step
    step="$(tr -d '\n\r[:space:]' < "$tracker")"
    if [ "$step" != "$expected_step" ]; then
        echo "[ERROR] wrong source checkpoint step for $label. expected $expected_step, got $step" >&2
        echo "        $run_dir" >&2
        exit 1
    fi
}

check_dataset() {
    local label="$1"
    local dataset_dir="$2"
    if ! compgen -G "$dataset_dir/*.bin" >/dev/null; then
        echo "[ERROR] missing $label dataset bins: $dataset_dir" >&2
        exit 1
    fi
    if ! compgen -G "$dataset_dir/*.idx" >/dev/null; then
        echo "[ERROR] missing $label dataset indices: $dataset_dir" >&2
        exit 1
    fi
}

is_completed_target() {
    local train_weights="$1"
    local tracker="$train_weights/latest_checkpointed_iteration.txt"
    [ -f "$tracker" ] && [ "$(tr -d '\n\r[:space:]' < "$tracker")" = "$TRAIN_ITERS" ]
}

ensure_target_is_safe() {
    local label="$1"
    local train_weights="$2"
    if is_completed_target "$train_weights"; then
        echo "[SKIP] $label already completed: $train_weights"
        return 1
    fi
    if [ -e "$train_weights" ]; then
        echo "[ERROR] destination already exists but is not complete for $label:" >&2
        echo "        $train_weights" >&2
        echo "        inspect or remove it before rerunning." >&2
        exit 1
    fi
    return 0
}

pause_after_stage() {
    if [ "$PAUSE_SECONDS" -gt 0 ]; then
        echo "[PAUSE] sleeping ${PAUSE_SECONDS}s before next phase4 run $(date)"
        sleep "$PAUSE_SECONDS"
    fi
}

common_env=(
    WANDB_MODE="$WANDB_MODE"
    WANDB_PROJECT="$WANDB_PROJECT"
    TOKENIZER_MODEL="$TOKENIZER_MODEL"
    DIRECT_LOCAL_SAVE=1
    TRAIN_ITERS="$TRAIN_ITERS"
    SAVE_INTERVAL="$SAVE_INTERVAL"
    EVAL_INTERVAL="$EVAL_INTERVAL"
    LOG_INTERVAL="$LOG_INTERVAL"
    PROBE_EVAL_INTERVAL="$PROBE_EVAL_INTERVAL"
    SECONDARY_PROBE_EVAL_INTERVAL="$SECONDARY_PROBE_EVAL_INTERVAL"
    GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE"
    SEQ_LENGTH="$SEQ_LENGTH"
    SOURCE_NUM_EXPERTS="$SOURCE_NUM_EXPERTS"
    NUM_EXPERTS="$NUM_EXPERTS"
    MOE_ROUTER_TOPK="$MOE_ROUTER_TOPK"
    MOE_FFN_HIDDEN_SIZE="$MOE_FFN_HIDDEN_SIZE"
    MOE_GROUPED_GEMM="$MOE_GROUPED_GEMM"
    MOE_PERMUTE_FUSION="$MOE_PERMUTE_FUSION"
    MOE_ROUTER_DTYPE="$MOE_ROUTER_DTYPE"
    TRAIN_DATASET="$CONVERSATION_TRAIN"
    DATASET_NAME="conversation_opensubtitles_exact_dedup"
    DATASET_SOURCE="OpenSubtitles-v2018 English exact-dedup train; 1.403B tokens; 1800 steps ~= 1.51 epochs"
    PROBE_NAME=wiki_probe
    PROBE_DATASET="$WIKI_PROBE_DATASET"
    SECONDARY_PROBE_NAME=code_probe
    SECONDARY_PROBE_DATASET="$CODE_PROBE_DATASET"
    TERTIARY_PROBE_NAME=conversation_probe
    TERTIARY_PROBE_DATASET="$CONVERSATION_PROBE_DATASET"
    TERTIARY_PROBE_EVAL_ITERS=25
    TERTIARY_PROBE_EVAL_INTERVAL="$TERTIARY_PROBE_EVAL_INTERVAL"
    PROBE_STEP_OFFSET="$SOURCE_LOGICAL_STEP"
    SECONDARY_PROBE_STEP_OFFSET="$SOURCE_LOGICAL_STEP"
    TERTIARY_PROBE_STEP_OFFSET="$SOURCE_LOGICAL_STEP"
    WANDB_STEP_OFFSET="$SOURCE_LOGICAL_STEP"
    RUN_INITIAL_PROBE_EVAL=1
    RUN_INITIAL_VALID_EVAL=1
    LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND=0
    WANDB_RESUME=allow
)

run_ffn_only() {
    # Freeze behaviour is overridable so the same code path serves both:
    #   - freeze       (default): old experts + shared/attention all frozen
    #   - attn_unfreeze        : old experts frozen, attention (shared) trained
    local freeze_shared="${FFN_FREEZE_SHARED:-1}"
    local train_attn="${FFN_TRAIN_ATTENTION_WITH_NEW_EXPERTS:-0}"
    local freeze_dense_attn_lora="${FFN_FREEZE_DENSE_ATTENTION_LORA:-1}"
    local mb="${FFN_MICRO_BATCH_SIZE:-96}"
    local default_run_id
    if [ "$train_attn" = "1" ]; then
        default_run_id="g2-ffn-only-attn-unfreeze-phase4-conversation-from-router-retuned-e16to24-mb${mb}-1800"
    else
        default_run_id="g2-ffn-only-phase4-conversation-from-router-retuned-e16to24-mb${mb}-1800"
    fi
    local run_id="${FFN_RUN_ID:-$default_run_id}"
    local train_weights="${FFN_TRAIN_WEIGHTS:-$PHASE4_ROOT/$run_id}"
    local label="ffn_only"
    [ "$train_attn" = "1" ] && label="ffn_only_attn_unfreeze"
    local debug_path="$train_weights/logs/trainable_params_debug.json"

    ensure_target_is_safe "$label" "$train_weights" || return 0

    echo "[START] $label (freeze_shared=$freeze_shared train_attn=$train_attn) $(date)"
    env \
        "${common_env[@]}" \
        RUN_ID="$run_id" \
        TRAIN_WEIGHTS="$train_weights" \
        SOURCE_TASK=code \
        TARGET_TASK=conversation \
        STAGE_NAME=phase4_conversation_after_router_retune_ffn_only \
        STAGE_LABEL=phase4_conversation_ffn_only \
        SOURCE_WEIGHTS_DIR="$FFN_SOURCE" \
        SOURCE_REQUIRED_ITERS="$FFN_SOURCE_REQUIRED_ITERS" \
        FREEZE_SHARED="$freeze_shared" \
        TRAIN_ATTENTION_WITH_NEW_EXPERTS="$train_attn" \
        FREEZE_DENSE_ATTENTION_LORA_WITH_NEW_EXPERTS="$freeze_dense_attn_lora" \
        MICRO_BATCH_SIZE="${FFN_MICRO_BATCH_SIZE:-96}" \
        ENABLE_OLD_MODEL_KL=0 \
        OLD_MODEL_KL_COEFF=0.0 \
        OLD_MODEL_KL_TEMPERATURE=1.0 \
        WANDB_RUN_ID="$run_id" \
        WANDB_EXP_NAME="${FFN_WANDB_EXP_NAME:-G2 FFN-only - phase4 conversation offline}" \
        WANDB_SAVE_DIR="$train_weights/wandb" \
        DEBUG_TRAINABLE_PARAMS_AND_EXIT="$DEBUG_TRAINABLE_PARAMS_AND_EXIT" \
        DEBUG_TRAINABLE_PARAMS_PATH="${FFN_DEBUG_TRAINABLE_PARAMS_PATH:-$debug_path}" \
        MASTER_PORT="${FFN_MASTER_PORT:-29811}" \
        "$SCRIPT_DIR/run_guarded_training.sh" \
        bash "$SCRIPT_DIR/run_continual_moe_a100_bf16.sh"
    echo "[END] $label $(date)"
}

run_shared_router() {
    local label="$1"
    local source_dir="$2"
    local train_all_experts_and_router="$3"
    local train_all_router_rows="$4"
    local run_id="$5"
    local wandb_name="$6"
    local port="$7"
    local train_weights="$PHASE4_ROOT/$run_id"
    local debug_path="$train_weights/logs/trainable_params_debug.json"

    ensure_target_is_safe "$label" "$train_weights" || return 0

    echo "[START] $label $(date)"
    env \
        "${common_env[@]}" \
        RUN_ID="$run_id" \
        TRAIN_WEIGHTS="$train_weights" \
        STAGE1_WEIGHTS_DIR="$source_dir" \
        MICRO_BATCH_SIZE="${SHARED_ROUTER_MICRO_BATCH_SIZE:-72}" \
        ATTN_LORA_RANK="$ATTN_LORA_RANK" \
        ATTN_LORA_ALPHA="$ATTN_LORA_ALPHA" \
        ATTN_FULL_RANK_LORA_RANK="$ATTN_FULL_RANK_LORA_RANK" \
        ATTN_FULL_RANK_LORA_ALPHA="$ATTN_FULL_RANK_LORA_ALPHA" \
        ATTN_FULL_RANK_LORA_TARGETS="$ATTN_FULL_RANK_LORA_TARGETS" \
        ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="$ATTN_FULL_RANK_LORA_ACTIVE_TARGETS" \
        ATTN_LORA_GROUPED_GEMM="$ATTN_LORA_GROUPED_GEMM" \
        SHARED_ROUTER_HYBRID_TRAIN_ALL_EXPERTS_AND_ROUTER="$train_all_experts_and_router" \
        SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS="$train_all_router_rows" \
        ROUTER_MEMORY_KL_COEFF=0.0 \
        ROUTER_MEMORY_INTERVAL=0 \
        WANDB_RUN_ID="$run_id" \
        WANDB_EXP_NAME="$wandb_name" \
        WANDB_SAVE_DIR="$train_weights/wandb" \
        DEBUG_TRAINABLE_PARAMS_AND_EXIT="$DEBUG_TRAINABLE_PARAMS_AND_EXIT" \
        DEBUG_TRAINABLE_PARAMS_PATH="${DEBUG_TRAINABLE_PARAMS_PATH:-$debug_path}" \
        MASTER_PORT="$port" \
        "$SCRIPT_DIR/run_guarded_training.sh" \
        bash "$PROJECT_ROOT/scripts/experiment/continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh"
    echo "[END] $label $(date)"
}

echo "[CONFIG] G2 phase4 conversation after router finetune offline chain"
echo "[CONFIG] conversation=$CONVERSATION_TRAIN"
echo "[CONFIG] tokens=${CONVERSATION_TOKEN_COUNT}, required_tokens=${REQUIRED_TOKEN_COUNT}, approx_epochs=${APPROX_EPOCHS}"
echo "[CONFIG] train_iters=$TRAIN_ITERS, logical_steps=${SOURCE_LOGICAL_STEP}->${TARGET_LOGICAL_STEP}"
echo "[CONFIG] debug_trainable_params_and_exit=$DEBUG_TRAINABLE_PARAMS_AND_EXIT"
echo "[CONFIG] run_only_stage=$RUN_ONLY_STAGE"
echo "[CONFIG] probes=code_probe + wiki_probe + conversation_probe"
echo "[CONFIG] experts=${SOURCE_NUM_EXPERTS}->${NUM_EXPERTS}, topk=${MOE_ROUTER_TOPK}"
echo "[CONFIG] ffn_source=$FFN_SOURCE"
echo "[CONFIG] exp1_freeze_wiki_source=$EXP1_FREEZE_WIKI_SOURCE"
echo "[CONFIG] exp2_unfreeze_wiki_source=$EXP2_UNFREEZE_WIKI_SOURCE"
echo "[CONFIG] phase4_root=$PHASE4_ROOT"

check_dataset "conversation train" "$CONVERSATION_TRAIN"
check_dataset "conversation probe" "$CONVERSATION_PROBE_DATASET"
check_dataset "code probe" "$CODE_PROBE_DATASET"
check_dataset "wiki probe" "$WIKI_PROBE_DATASET"
check_completed_source "ffn_only" "$FFN_SOURCE" "$FFN_SOURCE_REQUIRED_ITERS"
check_completed_source "exp1_freeze_wiki" "$EXP1_FREEZE_WIKI_SOURCE" 3600
check_completed_source "exp2_unfreeze_wiki" "$EXP2_UNFREEZE_WIKI_SOURCE" 3600

case "$RUN_ONLY_STAGE" in
    all)
        run_ffn_only
        pause_after_stage
        run_shared_router \
            "exp1_freeze_wiki" \
            "$EXP1_FREEZE_WIKI_SOURCE" \
            0 \
            0 \
            "${EXP1_RUN_ID:-g2-exp1-freeze-wiki-phase4-conversation-from-router-retuned-e16to24-mb72-1800}" \
            "${EXP1_WANDB_EXP_NAME:-G2 - exp1 freeze-wiki - phase4 conversation offline}" \
            "${EXP1_MASTER_PORT:-29812}"
        pause_after_stage
        run_shared_router \
            "exp2_unfreeze_wiki" \
            "$EXP2_UNFREEZE_WIKI_SOURCE" \
            1 \
            1 \
            "${EXP2_RUN_ID:-g2-exp2-unfreeze-wiki-phase4-conversation-from-router-retuned-e16to24-mb72-1800}" \
            "${EXP2_WANDB_EXP_NAME:-G2 - exp2 unfreeze-wiki - phase4 conversation offline}" \
            "${EXP2_MASTER_PORT:-29813}"
        ;;
    ffn_only)
        run_ffn_only
        ;;
    ffn_only_attn_unfreeze)
        # Exp 3: expand 16->24, old experts frozen, attention (shared) trained.
        FFN_TRAIN_ATTENTION_WITH_NEW_EXPERTS=1 \
        FFN_FREEZE_DENSE_ATTENTION_LORA=0 \
        FFN_FREEZE_SHARED=1 \
        FFN_MASTER_PORT="${FFN_MASTER_PORT:-29814}" \
            run_ffn_only
        ;;
    exp1_freeze_wiki)
        run_shared_router \
            "exp1_freeze_wiki" \
            "$EXP1_FREEZE_WIKI_SOURCE" \
            0 \
            0 \
            "${EXP1_RUN_ID:-g2-exp1-freeze-wiki-phase4-conversation-from-router-retuned-e16to24-mb72-1800}" \
            "${EXP1_WANDB_EXP_NAME:-G2 - exp1 freeze-wiki - phase4 conversation offline}" \
            "${EXP1_MASTER_PORT:-29812}"
        ;;
    exp2_unfreeze_wiki)
        run_shared_router \
            "exp2_unfreeze_wiki" \
            "$EXP2_UNFREEZE_WIKI_SOURCE" \
            1 \
            1 \
            "${EXP2_RUN_ID:-g2-exp2-unfreeze-wiki-phase4-conversation-from-router-retuned-e16to24-mb72-1800}" \
            "${EXP2_WANDB_EXP_NAME:-G2 - exp2 unfreeze-wiki - phase4 conversation offline}" \
            "${EXP2_MASTER_PORT:-29813}"
        ;;
    *)
        echo "[ERROR] invalid RUN_ONLY_STAGE=$RUN_ONLY_STAGE" >&2
        echo "        choose one of: all, ffn_only, ffn_only_attn_unfreeze, exp1_freeze_wiki, exp2_unfreeze_wiki" >&2
        exit 1
        ;;
esac

echo "[ALL DONE] $(date)"
