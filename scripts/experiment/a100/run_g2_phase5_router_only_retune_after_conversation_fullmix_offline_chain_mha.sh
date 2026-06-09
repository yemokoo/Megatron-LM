#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"

usage() {
    cat >&2 <<'EOF'
usage: run_g2_phase5_router_only_retune_after_conversation_fullmix_offline_chain_mha.sh [all|ffn_only|exp1_freeze_wiki|exp2_unfreeze_wiki]

Router-only finetune after phase4 conversation training.
Each stage starts from the completed phase4 checkpoint, freezes every parameter
except router weights, and trains on full wiki+code+conversation train data with
standard LM loss only (aux/z losses disabled).
EOF
}

TARGET="${1:-all}"
case "$TARGET" in
    all|ffn_only|exp1_freeze_wiki|exp2_unfreeze_wiki)
        ;;
    *)
        usage
        exit 1
        ;;
esac

export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_LOG_CHECKPOINTS="${WANDB_LOG_CHECKPOINTS:-0}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-EleutherAI/pythia-12b}"

export SAVE_INTERVAL="${SAVE_INTERVAL:-600}"
export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-1}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}"
export TERTIARY_PROBE_EVAL_INTERVAL="${TERTIARY_PROBE_EVAL_INTERVAL:-50}"
export FFN_MICRO_BATCH_SIZE="${FFN_MICRO_BATCH_SIZE:-48}"
export SHARED_ROUTER_MICRO_BATCH_SIZE="${SHARED_ROUTER_MICRO_BATCH_SIZE:-48}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export SEQ_LENGTH="${SEQ_LENGTH:-512}"
export PAUSE_SECONDS="${PAUSE_SECONDS:-180}"

export G2_ROOT="${G2_ROOT:-$PROJECT_ROOT/.local/weights/a100/mha/g2-checkpoints}"
export PHASE4_ROOT="${PHASE4_ROOT:-$G2_ROOT/conversation/phase4}"
export PHASE5_ROOT="${PHASE5_ROOT:-$G2_ROOT/conversation/phase5_router_finetune_fullmix}"
mkdir -p "$PHASE5_ROOT"

export TRAIN_DATASET_WIKI="${TRAIN_DATASET_WIKI:-$(dataset_dir_for_task wiki)}"
export TRAIN_DATASET_CODE="${TRAIN_DATASET_CODE:-$(dataset_dir_for_task code)}"
export TRAIN_DATASET_CONVERSATION="${TRAIN_DATASET_CONVERSATION:-$(dataset_dir_for_task conversation)}"
export PROBE_DATASET="${PROBE_DATASET:-$(probe_dir_for_task code)}"
export SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET:-$(probe_dir_for_task wiki)}"
export TERTIARY_PROBE_DATASET="${TERTIARY_PROBE_DATASET:-$(probe_dir_for_task conversation)}"

export PHASE4_SOURCE_STEP="${PHASE4_SOURCE_STEP:-1800}"
export SOURCE_LOGICAL_STEP="${SOURCE_LOGICAL_STEP:-7200}"
export DISPLAY_STEP_OFFSET="$((SOURCE_LOGICAL_STEP - PHASE4_SOURCE_STEP))"

export NUM_EXPERTS="${NUM_EXPERTS:-24}"
export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-24}"
export RESUME_FROM_NUM_EXPERTS="${RESUME_FROM_NUM_EXPERTS:-16}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export MOE_AUX_LOSS_COEFF=0.0
export MOE_Z_LOSS_COEFF=0.0
export MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}"
export MIXED_DATA_WEIGHT_MODE="${MIXED_DATA_WEIGHT_MODE:-token_proportional}"

export FFN_SOURCE="${FFN_SOURCE:-$PHASE4_ROOT/g2-ffn-only-phase4-conversation-from-router-retuned-e16to24-mb48-1800}"
export EXP1_FREEZE_WIKI_SOURCE="${EXP1_FREEZE_WIKI_SOURCE:-$PHASE4_ROOT/g2-exp1-freeze-wiki-phase4-conversation-from-router-retuned-e16to24-mb48-1800}"
export EXP2_UNFREEZE_WIKI_SOURCE="${EXP2_UNFREEZE_WIKI_SOURCE:-$PHASE4_ROOT/g2-exp2-unfreeze-wiki-phase4-conversation-from-router-retuned-e16to24-mb48-1800}"

tracker_value() {
    tr -d '\n\r[:space:]' < "$1/latest_checkpointed_iteration.txt"
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

check_source() {
    local label="$1"
    local source="$2"
    if [ ! -f "$source/latest_checkpointed_iteration.txt" ]; then
        echo "[ERROR] missing phase4 checkpoint tracker for $label: $source/latest_checkpointed_iteration.txt" >&2
        exit 1
    fi
    local step
    step="$(tracker_value "$source")"
    if [ "$step" != "$PHASE4_SOURCE_STEP" ]; then
        echo "[ERROR] wrong phase4 checkpoint step for $label. expected $PHASE4_SOURCE_STEP, got $step" >&2
        echo "        $source" >&2
        exit 1
    fi
}

compute_fullmix_retune_iters() {
    "$PYTHON_BIN" - "$TRAIN_DATASET_WIKI" "$TRAIN_DATASET_CODE" "$TRAIN_DATASET_CONVERSATION" <<'PY'
import math
import os
import sys
from pathlib import Path
from megatron.core.datasets import indexed_dataset

total_tokens = 0
for dataset_arg in sys.argv[1:]:
    dataset_dir = Path(dataset_arg)
    for idx_path in sorted(dataset_dir.glob("*.idx")):
        ds = indexed_dataset.IndexedDataset(str(idx_path.with_suffix("")), multimodal=False, mmap=True)
        total_tokens += int(ds.sequence_lengths.sum())
seq_length = int(os.environ.get("SEQ_LENGTH", "512"))
global_batch_size = int(os.environ.get("GLOBAL_BATCH_SIZE", "2304"))
denom = seq_length * global_batch_size
print(max(1, math.ceil(total_tokens / denom)))
PY
}

export RETUNE_ITERS="${RETUNE_ITERS:-$(compute_fullmix_retune_iters)}"
export TARGET_LOCAL_STEP="$((PHASE4_SOURCE_STEP + RETUNE_ITERS))"
export TARGET_LOGICAL_STEP="$((SOURCE_LOGICAL_STEP + RETUNE_ITERS))"

is_completed_target() {
    local train_weights="$1"
    local tracker="$train_weights/latest_checkpointed_iteration.txt"
    [ -f "$tracker" ] && [ "$(tr -d '\n\r[:space:]' < "$tracker")" = "$TARGET_LOCAL_STEP" ]
}

ensure_phase5_copy() {
    local label="$1"
    local source="$2"
    local train_weights="$3"
    local run_id="$4"

    if is_completed_target "$train_weights"; then
        echo "[SKIP] $label already completed: $train_weights"
        return 1
    fi
    if [ -e "$train_weights" ]; then
        echo "[ERROR] destination already exists but is not complete for $label:" >&2
        echo "        $train_weights" >&2
        echo "        inspect or move it before rerunning." >&2
        exit 1
    fi

    echo "[COPY] $label source:      $source"
    echo "[COPY] $label destination: $train_weights"
    mkdir -p "$train_weights"
    rsync -aH --info=progress2 \
        --exclude 'wandb/' \
        "$source/" "$train_weights/"

    {
        echo "phase=5"
        echo "purpose=router-only full wiki+code+conversation finetune copy"
        echo "stage=$label"
        echo "run_id=$run_id"
        echo "source=$source"
        echo "source_step=$PHASE4_SOURCE_STEP"
        echo "source_logical_step=$SOURCE_LOGICAL_STEP"
        echo "retune_iters=$RETUNE_ITERS"
        echo "target_local_step=$TARGET_LOCAL_STEP"
        echo "target_logical_step=$TARGET_LOGICAL_STEP"
        echo "mixed_data_weight_mode=$MIXED_DATA_WEIGHT_MODE"
        echo "copied_at=$(date -Is)"
    } > "$train_weights/PHASE5_SOURCE.txt"
    echo "[DONE] prepared $train_weights"
}

pause_after_stage() {
    if [ "$PAUSE_SECONDS" -gt 0 ]; then
        echo "[PAUSE] sleeping ${PAUSE_SECONDS}s before next phase5 run $(date)"
        sleep "$PAUSE_SECONDS"
    fi
}

run_ffn_only() {
    local run_id="${FFN_RUN_ID:-g2-ffn-only-phase5-router-only-retune-wikicodeconv-fullmix-from-phase4-mb48-${RETUNE_ITERS}}"
    local train_weights="$PHASE5_ROOT/$run_id"
    local label="ffn_only"

    ensure_phase5_copy "$label" "$FFN_SOURCE" "$train_weights" "$run_id" || return 0

    echo "[START] $label $(date)"
    env \
        RUN_ID="$run_id" \
        TRAIN_WEIGHTS="$train_weights" \
        SOURCE_STEP="$PHASE4_SOURCE_STEP" \
        RETUNE_ITERS="$RETUNE_ITERS" \
        TRAIN_ITERS="$TARGET_LOCAL_STEP" \
        TRAIN_DATASET_WIKI="$TRAIN_DATASET_WIKI" \
        TRAIN_DATASET_CODE="$TRAIN_DATASET_CODE" \
        TRAIN_DATASET_CONVERSATION="$TRAIN_DATASET_CONVERSATION" \
        DATASET_NAME="wiki_code_conversation_fullmix_router_only" \
        DATASET_SOURCE="Full Wikipedia train + Python code train + OpenSubtitles conversation train" \
        MIXED_DATA_WEIGHT_MODE="$MIXED_DATA_WEIGHT_MODE" \
        PROBE_DATASET="$PROBE_DATASET" \
        SECONDARY_PROBE_DATASET="$SECONDARY_PROBE_DATASET" \
        TERTIARY_PROBE_DATASET="$TERTIARY_PROBE_DATASET" \
        TERTIARY_PROBE_NAME=conversation_probe \
        TERTIARY_PROBE_EVAL_INTERVAL="$TERTIARY_PROBE_EVAL_INTERVAL" \
        PROBE_STEP_OFFSET="$DISPLAY_STEP_OFFSET" \
        SECONDARY_PROBE_STEP_OFFSET="$DISPLAY_STEP_OFFSET" \
        TERTIARY_PROBE_STEP_OFFSET="$DISPLAY_STEP_OFFSET" \
        WANDB_STEP_OFFSET="$DISPLAY_STEP_OFFSET" \
        WANDB_EXP_NAME="${FFN_WANDB_EXP_NAME:-G2 FFN-only - phase5 router-only full wiki+code+conv}" \
        WANDB_RUN_ID="$run_id" \
        MASTER_PORT="${FFN_MASTER_PORT:-29821}" \
        MODEL_CONFIG_SCRIPT="scripts/experiment/a100/flame-moe-bf16-no-shared.sh" \
        MICRO_BATCH_SIZE="$FFN_MICRO_BATCH_SIZE" \
        GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
        SAVE_INTERVAL="$SAVE_INTERVAL" \
        SAVE_CHECKPOINTS="$SAVE_CHECKPOINTS" \
        LOG_INTERVAL="$LOG_INTERVAL" \
        PROBE_EVAL_INTERVAL="$PROBE_EVAL_INTERVAL" \
        SECONDARY_PROBE_EVAL_INTERVAL="$SECONDARY_PROBE_EVAL_INTERVAL" \
        SOURCE_NUM_EXPERTS="$SOURCE_NUM_EXPERTS" \
        NUM_EXPERTS="$NUM_EXPERTS" \
        RESUME_FROM_NUM_EXPERTS="$RESUME_FROM_NUM_EXPERTS" \
        MOE_AUX_LOSS_COEFF="$MOE_AUX_LOSS_COEFF" \
        MOE_Z_LOSS_COEFF="$MOE_Z_LOSS_COEFF" \
        ATTN_FULL_RANK_LORA_RANK=0 \
        ATTN_FULL_RANK_LORA_ALPHA=0 \
        bash "$SCRIPT_DIR/run_guarded_training.sh" \
            bash "$SCRIPT_DIR/phase3_router_only_retune_moe_mixed_local_bf16.sh"
    echo "[END] $label $(date)"
}

run_shared_router() {
    local label="$1"
    local source="$2"
    local run_id="$3"
    local wandb_name="$4"
    local port="$5"
    local train_weights="$PHASE5_ROOT/$run_id"

    ensure_phase5_copy "$label" "$source" "$train_weights" "$run_id" || return 0

    echo "[START] $label $(date)"
    env \
        RUN_ID="$run_id" \
        TRAIN_WEIGHTS="$train_weights" \
        SOURCE_STEP="$PHASE4_SOURCE_STEP" \
        RETUNE_ITERS="$RETUNE_ITERS" \
        TRAIN_ITERS="$TARGET_LOCAL_STEP" \
        TRAIN_DATASET_WIKI="$TRAIN_DATASET_WIKI" \
        TRAIN_DATASET_CODE="$TRAIN_DATASET_CODE" \
        TRAIN_DATASET_CONVERSATION="$TRAIN_DATASET_CONVERSATION" \
        DATASET_NAME="wiki_code_conversation_fullmix_router_only" \
        DATASET_SOURCE="Full Wikipedia train + Python code train + OpenSubtitles conversation train" \
        MIXED_DATA_WEIGHT_MODE="$MIXED_DATA_WEIGHT_MODE" \
        PROBE_DATASET="$PROBE_DATASET" \
        SECONDARY_PROBE_DATASET="$SECONDARY_PROBE_DATASET" \
        TERTIARY_PROBE_DATASET="$TERTIARY_PROBE_DATASET" \
        TERTIARY_PROBE_NAME=conversation_probe \
        TERTIARY_PROBE_EVAL_INTERVAL="$TERTIARY_PROBE_EVAL_INTERVAL" \
        PROBE_STEP_OFFSET="$DISPLAY_STEP_OFFSET" \
        SECONDARY_PROBE_STEP_OFFSET="$DISPLAY_STEP_OFFSET" \
        TERTIARY_PROBE_STEP_OFFSET="$DISPLAY_STEP_OFFSET" \
        WANDB_STEP_OFFSET="$DISPLAY_STEP_OFFSET" \
        WANDB_EXP_NAME="$wandb_name" \
        WANDB_RUN_ID="$run_id" \
        MASTER_PORT="$port" \
        MICRO_BATCH_SIZE="$SHARED_ROUTER_MICRO_BATCH_SIZE" \
        GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
        SAVE_INTERVAL="$SAVE_INTERVAL" \
        SAVE_CHECKPOINTS="$SAVE_CHECKPOINTS" \
        LOG_INTERVAL="$LOG_INTERVAL" \
        PROBE_EVAL_INTERVAL="$PROBE_EVAL_INTERVAL" \
        SECONDARY_PROBE_EVAL_INTERVAL="$SECONDARY_PROBE_EVAL_INTERVAL" \
        SOURCE_NUM_EXPERTS="$SOURCE_NUM_EXPERTS" \
        NUM_EXPERTS="$NUM_EXPERTS" \
        RESUME_FROM_NUM_EXPERTS="$RESUME_FROM_NUM_EXPERTS" \
        MOE_AUX_LOSS_COEFF="$MOE_AUX_LOSS_COEFF" \
        MOE_Z_LOSS_COEFF="$MOE_Z_LOSS_COEFF" \
        bash "$SCRIPT_DIR/run_guarded_training.sh" \
            bash "$SCRIPT_DIR/phase3_router_only_retune_shared_router_hybrid_mixed_local_bf16.sh"
    echo "[END] $label $(date)"
}

echo "[CONFIG] G2 phase5 router-only fullmix retune after conversation"
echo "[CONFIG] target=$TARGET"
echo "[CONFIG] phase4_source_step=$PHASE4_SOURCE_STEP, source_logical_step=$SOURCE_LOGICAL_STEP"
echo "[CONFIG] retune_iters=$RETUNE_ITERS, local_steps=${PHASE4_SOURCE_STEP}->${TARGET_LOCAL_STEP}, logical_steps=${SOURCE_LOGICAL_STEP}->${TARGET_LOGICAL_STEP}"
echo "[CONFIG] data_weight_mode=$MIXED_DATA_WEIGHT_MODE"
echo "[CONFIG] train_data wiki=$TRAIN_DATASET_WIKI"
echo "[CONFIG] train_data code=$TRAIN_DATASET_CODE"
echo "[CONFIG] train_data conversation=$TRAIN_DATASET_CONVERSATION"
echo "[CONFIG] loss=LM only, aux=$MOE_AUX_LOSS_COEFF, z=$MOE_Z_LOSS_COEFF"
echo "[CONFIG] trainable=router weights only; frozen=all experts, dense trunk, embeddings, output weights"
echo "[CONFIG] phase5_root=$PHASE5_ROOT"

check_dataset "wiki train" "$TRAIN_DATASET_WIKI"
check_dataset "code train" "$TRAIN_DATASET_CODE"
check_dataset "conversation train" "$TRAIN_DATASET_CONVERSATION"
check_dataset "code probe" "$PROBE_DATASET"
check_dataset "wiki probe" "$SECONDARY_PROBE_DATASET"
check_dataset "conversation probe" "$TERTIARY_PROBE_DATASET"
check_source "ffn_only" "$FFN_SOURCE"
check_source "exp1_freeze_wiki" "$EXP1_FREEZE_WIKI_SOURCE"
check_source "exp2_unfreeze_wiki" "$EXP2_UNFREEZE_WIKI_SOURCE"

case "$TARGET" in
    all)
        run_ffn_only
        pause_after_stage
        run_shared_router \
            "exp1_freeze_wiki" \
            "$EXP1_FREEZE_WIKI_SOURCE" \
            "${EXP1_RUN_ID:-g2-exp1-freeze-wiki-phase5-router-only-retune-wikicodeconv-fullmix-from-phase4-mb48-${RETUNE_ITERS}}" \
            "${EXP1_WANDB_EXP_NAME:-G2 exp1 freeze-wiki - phase5 router-only full wiki+code+conv}" \
            "${EXP1_MASTER_PORT:-29822}"
        pause_after_stage
        run_shared_router \
            "exp2_unfreeze_wiki" \
            "$EXP2_UNFREEZE_WIKI_SOURCE" \
            "${EXP2_RUN_ID:-g2-exp2-unfreeze-wiki-phase5-router-only-retune-wikicodeconv-fullmix-from-phase4-mb48-${RETUNE_ITERS}}" \
            "${EXP2_WANDB_EXP_NAME:-G2 exp2 unfreeze-wiki - phase5 router-only full wiki+code+conv}" \
            "${EXP2_MASTER_PORT:-29823}"
        ;;
    ffn_only)
        run_ffn_only
        ;;
    exp1_freeze_wiki)
        run_shared_router \
            "exp1_freeze_wiki" \
            "$EXP1_FREEZE_WIKI_SOURCE" \
            "${EXP1_RUN_ID:-g2-exp1-freeze-wiki-phase5-router-only-retune-wikicodeconv-fullmix-from-phase4-mb48-${RETUNE_ITERS}}" \
            "${EXP1_WANDB_EXP_NAME:-G2 exp1 freeze-wiki - phase5 router-only full wiki+code+conv}" \
            "${EXP1_MASTER_PORT:-29822}"
        ;;
    exp2_unfreeze_wiki)
        run_shared_router \
            "exp2_unfreeze_wiki" \
            "$EXP2_UNFREEZE_WIKI_SOURCE" \
            "${EXP2_RUN_ID:-g2-exp2-unfreeze-wiki-phase5-router-only-retune-wikicodeconv-fullmix-from-phase4-mb48-${RETUNE_ITERS}}" \
            "${EXP2_WANDB_EXP_NAME:-G2 exp2 unfreeze-wiki - phase5 router-only full wiki+code+conv}" \
            "${EXP2_MASTER_PORT:-29823}"
        ;;
esac

echo "[ALL DONE] $(date)"
