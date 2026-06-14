#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_DATASET="${LOCAL_DATASET:-$LOCAL_BASE/dataset}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export G2_ROOT="${G2_ROOT:-$LOCAL_WEIGHTS/a100/mha/g2-checkpoints}"

export CODE_TOTAL_STEPS="${CODE_TOTAL_STEPS:-1800}"
export INTERLEAVE_CODE_STEPS="${INTERLEAVE_CODE_STEPS:-100}"
export INTERLEAVE_ROUTER_STEPS="${INTERLEAVE_ROUTER_STEPS:-100}"
export RUN_ROUTER_AFTER_FINAL="${RUN_ROUTER_AFTER_FINAL:-1}"

export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-8}"
export NUM_EXPERTS="${NUM_EXPERTS:-16}"
export RESUME_FROM_NUM_EXPERTS="${RESUME_FROM_NUM_EXPERTS:-$SOURCE_NUM_EXPERTS}"
export NUM_LAYERS="${NUM_LAYERS:-9}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-5472}"
export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_LAYER_FREQ="${MOE_LAYER_FREQ:-[0,1,1,1,1,1,1,1,1]}"
export ATTN_LORA_RANK="${ATTN_LORA_RANK:-256}"
export ATTN_LORA_ALPHA="${ATTN_LORA_ALPHA:-256}"
export ATTN_FULL_RANK_LORA_RANK="${ATTN_FULL_RANK_LORA_RANK:-256}"
export ATTN_FULL_RANK_LORA_ALPHA="${ATTN_FULL_RANK_LORA_ALPHA:-256}"
export ATTN_FULL_RANK_LORA_TARGETS="${ATTN_FULL_RANK_LORA_TARGETS:-qkvo}"
export ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="${ATTN_FULL_RANK_LORA_ACTIVE_TARGETS:-}"
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"
export ATTN_LORA_GROUPED_GEMM="${ATTN_LORA_GROUPED_GEMM:-1}"

export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-72}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-600}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export TRAIN_LOG_STEP_TIME_ONLY="${TRAIN_LOG_STEP_TIME_ONLY:-1}"

export CODE_LR="${CODE_LR:-3e-4}"
export CODE_MIN_LR="${CODE_MIN_LR:-3e-5}"
export CODE_MOE_AUX_LOSS_COEFF="${CODE_MOE_AUX_LOSS_COEFF:-${MOE_AUX_LOSS_COEFF:-0.01}}"
export CODE_MOE_Z_LOSS_COEFF="${CODE_MOE_Z_LOSS_COEFF:-${MOE_Z_LOSS_COEFF:-0.001}}"
export ROUTER_LR="${ROUTER_LR:-3e-4}"
export ROUTER_MIN_LR="${ROUTER_MIN_LR:-3e-5}"
export ROUTER_MOE_AUX_LOSS_COEFF="${ROUTER_MOE_AUX_LOSS_COEFF:-0.0}"
export ROUTER_MOE_Z_LOSS_COEFF="${ROUTER_MOE_Z_LOSS_COEFF:-0.0}"
export MIXED_DATA_WEIGHT_MODE="${MIXED_DATA_WEIGHT_MODE:-equal}"

export STAGE1_WEIGHTS_DIR="${STAGE1_WEIGHTS_DIR:-$G2_ROOT/wiki/g2-top4-e8-ffn352-r256-wiki-shared-router-qkvo-mha-a100-bf16-mb72-1800}"
export TRAIN_DATASET="${TRAIN_DATASET:-$(dataset_dir_for_task code)}"
export TRAIN_DATASET_WIKI="${TRAIN_DATASET_WIKI:-$(dataset_dir_for_task wiki)}"
export TRAIN_DATASET_CODE="${TRAIN_DATASET_CODE:-$(dataset_dir_for_task code)}"
export PROBE_DATASET="${PROBE_DATASET:-$(probe_dir_for_task code)}"
export SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET:-$(probe_dir_for_task wiki)}"

export BASE_RUN_ID="${BASE_RUN_ID:-g2-exp2-interleaved-code${INTERLEAVE_CODE_STEPS}-router${INTERLEAVE_ROUTER_STEPS}-wiki-to-code-new-experts-router-retune-mb${MICRO_BATCH_SIZE}-1800}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$G2_ROOT/code/interleaved_router_finetune/$BASE_RUN_ID}"
export LOG_DIR="${LOG_DIR:-$TRAIN_WEIGHTS/logs}"
export STATE_FILE="${STATE_FILE:-$TRAIN_WEIGHTS/interleaved_state.env}"
export INTERLEAVE_METADATA="${INTERLEAVE_METADATA:-$LOG_DIR/interleaved_run_metadata.json}"

validate_nonnegative_int() {
    local name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] $name must be a non-negative integer: $value" >&2
        exit 1
    fi
}

read_tracker() {
    local tracker="$TRAIN_WEIGHTS/latest_checkpointed_iteration.txt"
    if [ -f "$tracker" ]; then
        tr -d '\n\r[:space:]' < "$tracker"
    else
        echo 0
    fi
}

write_state() {
    local code_steps_done="$1"
    local checkpoint_step="$2"
    mkdir -p "$(dirname "$STATE_FILE")"
    {
        echo "CODE_STEPS_DONE=$code_steps_done"
        echo "CHECKPOINT_STEP=$checkpoint_step"
        echo "CODE_TOTAL_STEPS=$CODE_TOTAL_STEPS"
        echo "INTERLEAVE_CODE_STEPS=$INTERLEAVE_CODE_STEPS"
        echo "INTERLEAVE_ROUTER_STEPS=$INTERLEAVE_ROUTER_STEPS"
    } > "$STATE_FILE"
}

validate_nonnegative_int CODE_TOTAL_STEPS "$CODE_TOTAL_STEPS"
validate_nonnegative_int INTERLEAVE_CODE_STEPS "$INTERLEAVE_CODE_STEPS"
validate_nonnegative_int INTERLEAVE_ROUTER_STEPS "$INTERLEAVE_ROUTER_STEPS"
if [ "$CODE_TOTAL_STEPS" -le 0 ] || [ "$INTERLEAVE_CODE_STEPS" -le 0 ]; then
    echo "[ERROR] CODE_TOTAL_STEPS and INTERLEAVE_CODE_STEPS must be > 0" >&2
    exit 1
fi

if [ ! -f "$STAGE1_WEIGHTS_DIR/latest_checkpointed_iteration.txt" ]; then
    echo "[ERROR] wiki source checkpoint tracker is missing: $STAGE1_WEIGHTS_DIR/latest_checkpointed_iteration.txt" >&2
    exit 1
fi
for dataset_dir in "$TRAIN_DATASET" "$TRAIN_DATASET_WIKI" "$TRAIN_DATASET_CODE"; do
    if ! compgen -G "$dataset_dir/*.bin" >/dev/null; then
        echo "[ERROR] no .bin files found in dataset: $dataset_dir" >&2
        exit 1
    fi
done

mkdir -p "$LOG_DIR"

if [ -f "$STATE_FILE" ]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
else
    CODE_STEPS_DONE="${CODE_STEPS_DONE:-0}"
fi
validate_nonnegative_int CODE_STEPS_DONE "$CODE_STEPS_DONE"

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

metadata = {
    "stage": "g2_exp2_interleaved_code_router_finetune",
    "description": "Code training chunks train only new experts/new router rows; router-finetune chunks train all router rows only.",
    "base_run_id": os.environ["BASE_RUN_ID"],
    "train_weights": os.environ["TRAIN_WEIGHTS"],
    "stage1_weights_dir": os.environ["STAGE1_WEIGHTS_DIR"],
    "code_total_steps": int(os.environ["CODE_TOTAL_STEPS"]),
    "interleave_code_steps": int(os.environ["INTERLEAVE_CODE_STEPS"]),
    "interleave_router_steps": int(os.environ["INTERLEAVE_ROUTER_STEPS"]),
    "run_router_after_final": os.environ["RUN_ROUTER_AFTER_FINAL"] == "1",
    "code_phase_trainable": "new FFN experts, new attention experts, and new router rows only",
    "code_phase_loss": {
        "moe_aux_loss_coeff": float(os.environ["CODE_MOE_AUX_LOSS_COEFF"]),
        "moe_z_loss_coeff": float(os.environ["CODE_MOE_Z_LOSS_COEFF"]),
    },
    "router_phase_trainable": "all shared-router rows only",
    "router_phase_loss": {
        "moe_aux_loss_coeff": float(os.environ["ROUTER_MOE_AUX_LOSS_COEFF"]),
        "moe_z_loss_coeff": float(os.environ["ROUTER_MOE_Z_LOSS_COEFF"]),
    },
    "source_num_experts": int(os.environ["SOURCE_NUM_EXPERTS"]),
    "target_num_experts": int(os.environ["NUM_EXPERTS"]),
    "moe_router_topk": int(os.environ["MOE_ROUTER_TOPK"]),
    "mixed_data_weight_mode": os.environ["MIXED_DATA_WEIGHT_MODE"],
    "train_dataset_code": os.environ["TRAIN_DATASET"],
    "router_finetune_dataset_wiki": os.environ["TRAIN_DATASET_WIKI"],
    "router_finetune_dataset_code": os.environ["TRAIN_DATASET_CODE"],
}
Path(os.environ["INTERLEAVE_METADATA"]).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
PY

echo "[CONFIG] G2 exp2 interleaved code training + router finetune"
echo "[CONFIG] stage1=$STAGE1_WEIGHTS_DIR"
echo "[CONFIG] train_weights=$TRAIN_WEIGHTS"
echo "[CONFIG] code_total_steps=$CODE_TOTAL_STEPS"
echo "[CONFIG] interleave: code $INTERLEAVE_CODE_STEPS step(s), router $INTERLEAVE_ROUTER_STEPS step(s)"
echo "[CONFIG] code loss: aux=$CODE_MOE_AUX_LOSS_COEFF z=$CODE_MOE_Z_LOSS_COEFF"
echo "[CONFIG] router loss: aux=$ROUTER_MOE_AUX_LOSS_COEFF z=$ROUTER_MOE_Z_LOSS_COEFF"
echo "[CONFIG] code trainable=new experts + new router rows only"
echo "[CONFIG] router trainable=all router rows only"
echo "[CONFIG] starting code_steps_done=$CODE_STEPS_DONE checkpoint_step=$(read_tracker)"
echo "[CONFIG] metadata=$INTERLEAVE_METADATA"

while [ "$CODE_STEPS_DONE" -lt "$CODE_TOTAL_STEPS" ]; do
    current_step="$(read_tracker)"
    remaining_code_steps=$((CODE_TOTAL_STEPS - CODE_STEPS_DONE))
    code_chunk="$INTERLEAVE_CODE_STEPS"
    if [ "$code_chunk" -gt "$remaining_code_steps" ]; then
        code_chunk="$remaining_code_steps"
    fi
    code_target_step=$((current_step + code_chunk))

    echo
    echo "[CODE] start checkpoint=$current_step code_done=$CODE_STEPS_DONE chunk=$code_chunk target_checkpoint=$code_target_step"

    resume_args=()
    if [ "$current_step" -gt 0 ]; then
        resume_args+=(RESUME_FROM_WEIGHTS="$TRAIN_WEIGHTS" RESUME_LOAD_OPTIM=0 RESUME_RESET_ITERATION=0)
    fi

    env \
        RUN_ID="${BASE_RUN_ID}-code-${code_target_step}" \
        STAGE1_WEIGHTS_DIR="$STAGE1_WEIGHTS_DIR" \
        TRAIN_WEIGHTS="$TRAIN_WEIGHTS" \
        TRAIN_DATASET="$TRAIN_DATASET" \
        TRAIN_ITERS="$code_target_step" \
        SAVE_INTERVAL="$SAVE_INTERVAL" \
        PROBE_EVAL_INTERVAL="$PROBE_EVAL_INTERVAL" \
        SECONDARY_PROBE_EVAL_INTERVAL="$SECONDARY_PROBE_EVAL_INTERVAL" \
        LOG_INTERVAL="$LOG_INTERVAL" \
        TRAIN_LOG_STEP_TIME_ONLY="$TRAIN_LOG_STEP_TIME_ONLY" \
        MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" \
        GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
        LR="$CODE_LR" \
        MIN_LR="$CODE_MIN_LR" \
        MOE_AUX_LOSS_COEFF="$CODE_MOE_AUX_LOSS_COEFF" \
        MOE_Z_LOSS_COEFF="$CODE_MOE_Z_LOSS_COEFF" \
        SOURCE_NUM_EXPERTS="$SOURCE_NUM_EXPERTS" \
        NUM_EXPERTS="$NUM_EXPERTS" \
        NUM_LAYERS="$NUM_LAYERS" \
        HIDDEN_SIZE="$HIDDEN_SIZE" \
        FFN_HIDDEN_SIZE="$FFN_HIDDEN_SIZE" \
        NUM_QUERY_GROUPS="$NUM_QUERY_GROUPS" \
        MOE_FFN_HIDDEN_SIZE="$MOE_FFN_HIDDEN_SIZE" \
        MOE_ROUTER_TOPK="$MOE_ROUTER_TOPK" \
        MOE_LAYER_FREQ="$MOE_LAYER_FREQ" \
        ATTN_LORA_RANK="$ATTN_LORA_RANK" \
        ATTN_LORA_ALPHA="$ATTN_LORA_ALPHA" \
        ATTN_FULL_RANK_LORA_RANK="$ATTN_FULL_RANK_LORA_RANK" \
        ATTN_FULL_RANK_LORA_ALPHA="$ATTN_FULL_RANK_LORA_ALPHA" \
        ATTN_FULL_RANK_LORA_TARGETS="$ATTN_FULL_RANK_LORA_TARGETS" \
        ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="$ATTN_FULL_RANK_LORA_ACTIVE_TARGETS" \
        MOE_GROUPED_GEMM="$MOE_GROUPED_GEMM" \
        ATTN_LORA_GROUPED_GEMM="$ATTN_LORA_GROUPED_GEMM" \
        SHARED_ROUTER_HYBRID_TRAIN_ALL_EXPERTS_AND_ROUTER=0 \
        SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS=0 \
        SHARED_ROUTER_HYBRID_PARTIAL_FREEZE_MASK= \
        PROBE_DATASET="$PROBE_DATASET" \
        SECONDARY_PROBE_DATASET="$SECONDARY_PROBE_DATASET" \
        "${resume_args[@]}" \
        bash scripts/experiment/continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh

    CODE_STEPS_DONE=$((CODE_STEPS_DONE + code_chunk))
    current_step="$(read_tracker)"
    write_state "$CODE_STEPS_DONE" "$current_step"
    echo "[CODE] done code_done=$CODE_STEPS_DONE checkpoint_step=$current_step"

    if [ "$INTERLEAVE_ROUTER_STEPS" -le 0 ]; then
        continue
    fi
    if [ "$CODE_STEPS_DONE" -ge "$CODE_TOTAL_STEPS" ] && [ "$RUN_ROUTER_AFTER_FINAL" != "1" ]; then
        echo "[ROUTER] skipped after final code chunk because RUN_ROUTER_AFTER_FINAL=$RUN_ROUTER_AFTER_FINAL"
        continue
    fi

    router_target_step=$((current_step + INTERLEAVE_ROUTER_STEPS))
    echo
    echo "[ROUTER] start checkpoint=$current_step chunk=$INTERLEAVE_ROUTER_STEPS target_checkpoint=$router_target_step"

    env \
        RUN_ID="${BASE_RUN_ID}-router-${router_target_step}" \
        TRAIN_WEIGHTS="$TRAIN_WEIGHTS" \
        SOURCE_STEP="$current_step" \
        RETUNE_ITERS="$INTERLEAVE_ROUTER_STEPS" \
        TRAIN_ITERS="$router_target_step" \
        SAVE_INTERVAL="$SAVE_INTERVAL" \
        PROBE_EVAL_INTERVAL="$PROBE_EVAL_INTERVAL" \
        SECONDARY_PROBE_EVAL_INTERVAL="$SECONDARY_PROBE_EVAL_INTERVAL" \
        LOG_INTERVAL="$LOG_INTERVAL" \
        TRAIN_LOG_STEP_TIME_ONLY="$TRAIN_LOG_STEP_TIME_ONLY" \
        MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" \
        GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
        LR="$ROUTER_LR" \
        MIN_LR="$ROUTER_MIN_LR" \
        MOE_AUX_LOSS_COEFF="$ROUTER_MOE_AUX_LOSS_COEFF" \
        MOE_Z_LOSS_COEFF="$ROUTER_MOE_Z_LOSS_COEFF" \
        SOURCE_NUM_EXPERTS="$SOURCE_NUM_EXPERTS" \
        RESUME_FROM_NUM_EXPERTS="$RESUME_FROM_NUM_EXPERTS" \
        NUM_EXPERTS="$NUM_EXPERTS" \
        NUM_LAYERS="$NUM_LAYERS" \
        HIDDEN_SIZE="$HIDDEN_SIZE" \
        FFN_HIDDEN_SIZE="$FFN_HIDDEN_SIZE" \
        NUM_QUERY_GROUPS="$NUM_QUERY_GROUPS" \
        MOE_FFN_HIDDEN_SIZE="$MOE_FFN_HIDDEN_SIZE" \
        MOE_ROUTER_TOPK="$MOE_ROUTER_TOPK" \
        MOE_LAYER_FREQ="$MOE_LAYER_FREQ" \
        ATTN_LORA_RANK="$ATTN_LORA_RANK" \
        ATTN_LORA_ALPHA="$ATTN_LORA_ALPHA" \
        ATTN_FULL_RANK_LORA_RANK="$ATTN_FULL_RANK_LORA_RANK" \
        ATTN_FULL_RANK_LORA_ALPHA="$ATTN_FULL_RANK_LORA_ALPHA" \
        ATTN_FULL_RANK_LORA_TARGETS="$ATTN_FULL_RANK_LORA_TARGETS" \
        ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="$ATTN_FULL_RANK_LORA_ACTIVE_TARGETS" \
        MOE_GROUPED_GEMM="$MOE_GROUPED_GEMM" \
        ATTN_LORA_GROUPED_GEMM="$ATTN_LORA_GROUPED_GEMM" \
        TRAIN_DATASET_WIKI="$TRAIN_DATASET_WIKI" \
        TRAIN_DATASET_CODE="$TRAIN_DATASET_CODE" \
        MIXED_DATA_WEIGHT_MODE="$MIXED_DATA_WEIGHT_MODE" \
        PROBE_DATASET="$PROBE_DATASET" \
        SECONDARY_PROBE_DATASET="$SECONDARY_PROBE_DATASET" \
        bash scripts/experiment/a100/phase3_router_only_retune_shared_router_hybrid_mixed_local_bf16.sh

    current_step="$(read_tracker)"
    write_state "$CODE_STEPS_DONE" "$current_step"
    echo "[ROUTER] done code_done=$CODE_STEPS_DONE checkpoint_step=$current_step"
done

echo
echo "[ALL DONE] code_done=$CODE_STEPS_DONE final_checkpoint_step=$(read_tracker)"
echo "[ALL DONE] train_weights=$TRAIN_WEIGHTS"
