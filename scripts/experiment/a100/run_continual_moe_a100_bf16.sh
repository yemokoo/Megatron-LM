#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

export SOURCE_TASK="${SOURCE_TASK:?SOURCE_TASK must be set}"
export TARGET_TASK="${TARGET_TASK:?TARGET_TASK must be set}"
export FREEZE_SHARED="${FREEZE_SHARED:-0}"
export TRAIN_ATTENTION_WITH_NEW_EXPERTS="${TRAIN_ATTENTION_WITH_NEW_EXPERTS:-0}"
export FREEZE_DENSE_ATTENTION_LORA_WITH_NEW_EXPERTS="${FREEZE_DENSE_ATTENTION_LORA_WITH_NEW_EXPERTS:-0}"

if [ "$SOURCE_TASK" = "wiki" ] && [ "$TARGET_TASK" = "code" ]; then
    export STAGE_NAME="${STAGE_NAME:-a_to_b}"
    export STAGE_DIR_NAME="${STAGE_DIR_NAME:-a100/a-to-b-moe-bf16}"
    export STAGE_LABEL="${STAGE_LABEL:-a_to_b}"
elif [ "$SOURCE_TASK" = "code" ] && [ "$TARGET_TASK" = "wiki" ]; then
    export STAGE_NAME="${STAGE_NAME:-b_to_a}"
    export STAGE_DIR_NAME="${STAGE_DIR_NAME:-a100/b-to-a-moe-bf16}"
    export STAGE_LABEL="${STAGE_LABEL:-b_to_a}"
elif [ "$SOURCE_TASK" = "wiki" ] && [ "$TARGET_TASK" = "conversation" ]; then
    export STAGE_NAME="${STAGE_NAME:-wiki_to_conversation}"
    export STAGE_DIR_NAME="${STAGE_DIR_NAME:-a100/wiki-to-conversation-moe-bf16}"
    export STAGE_LABEL="${STAGE_LABEL:-wiki_to_conversation}"
elif [ "$SOURCE_TASK" = "code" ] && [ "$TARGET_TASK" = "conversation" ]; then
    export STAGE_NAME="${STAGE_NAME:-code_to_conversation}"
    export STAGE_DIR_NAME="${STAGE_DIR_NAME:-a100/code-to-conversation-moe-bf16}"
    export STAGE_LABEL="${STAGE_LABEL:-code_to_conversation}"
else
    echo "ERROR: unsupported continual direction ${SOURCE_TASK} -> ${TARGET_TASK}" >&2
    exit 1
fi

if [ "$FREEZE_SHARED" = "1" ]; then
    if [ "$TRAIN_ATTENTION_WITH_NEW_EXPERTS" = "1" ]; then
        export STAGE_NAME="${STAGE_NAME}_attn_unfreeze"
        export STAGE_DIR_NAME="${STAGE_DIR_NAME}-attn-unfreeze"
        export STAGE_LABEL="${STAGE_LABEL}_attn_unfreeze"
    else
        export STAGE_NAME="${STAGE_NAME}_freeze"
        export STAGE_DIR_NAME="${STAGE_DIR_NAME}-freeze"
        export STAGE_LABEL="${STAGE_LABEL}_freeze"
    fi
fi

export RUN_ID="${RUN_ID:-${STAGE_LABEL}-a100-bf16-$(date -u +%Y%m%d-%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29611}"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_DATASET="${LOCAL_DATASET:-$LOCAL_BASE/dataset}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
export STAGE_INPUTS_TO_SCRATCH="${STAGE_INPUTS_TO_SCRATCH:-0}"
export DIRECT_LOCAL_SAVE="${DIRECT_LOCAL_SAVE:-1}"
export STORAGE_PLAN_ONLY="${STORAGE_PLAN_ONLY:-0}"
# When the source checkpoint is ALREADY expanded to the target expert count
# (e.g. a pre-Code expansion-distill init), load it as a fresh finetune without
# re-expanding: --finetune resets iteration to 0 and --moe-resume-from-num-experts
# reapplies the new-experts+router freeze mask.
export LOAD_EXPANDED_SOURCE="${LOAD_EXPANDED_SOURCE:-0}"
export MOE_INTERLEAVE_CODE_STEPS="${MOE_INTERLEAVE_CODE_STEPS:-0}"
export MOE_INTERLEAVE_ROUTER_STEPS="${MOE_INTERLEAVE_ROUTER_STEPS:-0}"
export MOE_INTERLEAVE_CODE_TOTAL_STEPS="${MOE_INTERLEAVE_CODE_TOTAL_STEPS:-0}"
export MOE_INTERLEAVE_ROUTER_AFTER_FINAL="${MOE_INTERLEAVE_ROUTER_AFTER_FINAL:-1}"
export MOE_JOINT_REPLAY_LM="${MOE_JOINT_REPLAY_LM:-0}"
export MOE_JOINT_REPLAY_OLD_DATA_KD="${MOE_JOINT_REPLAY_OLD_DATA_KD:-0}"
export MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL="${MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL:-0}"
export MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE="${MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE:-0}"
export MOE_JOINT_REPLAY_TOTAL_SAMPLES="${MOE_JOINT_REPLAY_TOTAL_SAMPLES:-0}"
export MOE_JOINT_REPLAY_MICRO_BATCH_SIZE="${MOE_JOINT_REPLAY_MICRO_BATCH_SIZE:-0}"
export MOE_JOINT_REPLAY_OLD_LIKE_GT_PATH="${MOE_JOINT_REPLAY_OLD_LIKE_GT_PATH:-}"
export MOE_JOINT_REPLAY_OLD_LIKE_UNIT="${MOE_JOINT_REPLAY_OLD_LIKE_UNIT:-positive_sequence}"
export MOE_JOINT_REPLAY_OLD_LIKE_TARGET_TRAIN_FRACTION="${MOE_JOINT_REPLAY_OLD_LIKE_TARGET_TRAIN_FRACTION:-0}"
export MOE_JOINT_REPLAY_OLD_LIKE_SELECTED_TOKEN_COUNT="${MOE_JOINT_REPLAY_OLD_LIKE_SELECTED_TOKEN_COUNT:-0}"
export MOE_JOINT_REPLAY_OLD_LIKE_POSITIVE_SAMPLE_COUNT="${MOE_JOINT_REPLAY_OLD_LIKE_POSITIVE_SAMPLE_COUNT:-0}"
export MOE_JOINT_REPLAY_OLD_LIKE_FULL_TRAIN_TOKEN_COUNT="${MOE_JOINT_REPLAY_OLD_LIKE_FULL_TRAIN_TOKEN_COUNT:-0}"
export OLD_LIKE_REPLAY_SUBSET_COUNT="${OLD_LIKE_REPLAY_SUBSET_COUNT:-0}"
export OLD_LIKE_REPLAY_SUBSET_SHA256="${OLD_LIKE_REPLAY_SUBSET_SHA256:-}"
export MOE_JOINT_NEW_EXPERT_QUOTA="${MOE_JOINT_NEW_EXPERT_QUOTA:-0}"
export MOE_JOINT_NEW_EXPERT_QUOTA_SCHEDULE="${MOE_JOINT_NEW_EXPERT_QUOTA_SCHEDULE:-}"
export MOE_JOINT_NEW_EXPERT_QUOTA_MIN_NEW_SLOTS="${MOE_JOINT_NEW_EXPERT_QUOTA_MIN_NEW_SLOTS:-}"
export MOE_JOINT_NEW_EXPERT_QUOTA_LOSS_COEFF="${MOE_JOINT_NEW_EXPERT_QUOTA_LOSS_COEFF:-1.0}"
export MOE_JOINT_NEW_EXPERT_QUOTA_PRESERVE_NATURAL_GRADS="${MOE_JOINT_NEW_EXPERT_QUOTA_PRESERVE_NATURAL_GRADS:-0}"
export OLD_HIDDEN_KL_COEFF_START="${OLD_HIDDEN_KL_COEFF_START:-}"
export OLD_HIDDEN_KL_COEFF_DECAY_STEPS="${OLD_HIDDEN_KL_COEFF_DECAY_STEPS:-0}"
export LOG_ROUTER_GRAD_NORM_SOURCES="${LOG_ROUTER_GRAD_NORM_SOURCES:-0}"
export JOINT_REPLAY_DATASET="${JOINT_REPLAY_DATASET:-}"
export JOINT_REPLAY_SECONDARY_DATASET="${JOINT_REPLAY_SECONDARY_DATASET:-}"
export JOINT_REPLAY_DATA_WEIGHT_MODE="${JOINT_REPLAY_DATA_WEIGHT_MODE:-equal_dataset}"
export RECOVERY_SAVE_INTERVAL="${RECOVERY_SAVE_INTERVAL:-0}"
export NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-0}"
export OLD_MODEL_KL_WEIGHTS_DIR="${OLD_MODEL_KL_WEIGHTS_DIR:-}"
export OLD_MODEL_KL_NUM_EXPERTS="${OLD_MODEL_KL_NUM_EXPERTS:-}"
export FINGERPRINT_KD_ENABLE="${FINGERPRINT_KD_ENABLE:-0}"
export FINGERPRINT_KD_FORCE_ENABLE_ZERO_COEFF="${FINGERPRINT_KD_FORCE_ENABLE_ZERO_COEFF:-0}"
export FINGERPRINT_KD_COEFF="${FINGERPRINT_KD_COEFF:-0.0}"
export FINGERPRINT_KD_LM_LOSS_COEFF="${FINGERPRINT_KD_LM_LOSS_COEFF:-1.0}"
export LOG_FINGERPRINT_GRAD_NORM_GROUPS="${LOG_FINGERPRINT_GRAD_NORM_GROUPS:-0}"
export TRAIN_ROUTER_USAGE_LOG_INTERVAL="${TRAIN_ROUTER_USAGE_LOG_INTERVAL:-0}"
export TRAIN_ROUTER_USAGE_NUM_EXISTING_EXPERTS="${TRAIN_ROUTER_USAGE_NUM_EXISTING_EXPERTS:-}"
export FINGERPRINT_KD_BUNDLE="${FINGERPRINT_KD_BUNDLE:-}"
export FINGERPRINT_KD_RANK="${FINGERPRINT_KD_RANK:-64}"
export FINGERPRINT_KD_LAYERS="${FINGERPRINT_KD_LAYERS:-2,3,4,5,6,7,8,9}"
export FINGERPRINT_KD_SCORE_REPRESENTATION="${FINGERPRINT_KD_SCORE_REPRESENTATION:-stable}"
export FINGERPRINT_KD_LOSS_REPRESENTATION="${FINGERPRINT_KD_LOSS_REPRESENTATION:-stable}"
export FINGERPRINT_KD_GATE_MODE="${FINGERPRINT_KD_GATE_MODE:-soft}"
export FINGERPRINT_KD_THRESHOLD="${FINGERPRINT_KD_THRESHOLD:-0.1297607421875}"
export FINGERPRINT_KD_SOFT_TEMPERATURE="${FINGERPRINT_KD_SOFT_TEMPERATURE:-0.0069580078125}"
export FINGERPRINT_KD_WEIGHT_ASSIGNMENT="${FINGERPRINT_KD_WEIGHT_ASSIGNMENT:-stable}"

export SSD_MOUNT="${LOCAL_SSD_ROOT}/${RUN_ID}"
export SSD_TRAIN_DATASET="${SSD_MOUNT}/dataset/train"
export SSD_TRAIN_DATASET_SECONDARY="${SSD_MOUNT}/dataset/train_secondary"
export SSD_SOURCE_WEIGHTS="${SSD_MOUNT}/source_weights"
export SSD_TARGET_WEIGHTS="${SSD_MOUNT}/target_weights"
export SSD_INTERLEAVE_WIKI_DATASET="${SSD_MOUNT}/dataset/interleave_wiki_train"
export SSD_INTERLEAVE_SECONDARY_DATASET="${SSD_MOUNT}/dataset/interleave_secondary_train"
export SSD_JOINT_REPLAY_DATASET="${SSD_MOUNT}/dataset/joint_replay_train"
export SSD_JOINT_REPLAY_SECONDARY_DATASET="${SSD_MOUNT}/dataset/joint_replay_secondary_train"
export SSD_OLD_LIKE_GT="${SSD_MOUNT}/old_like_gt"

export NUM_LAYERS="${NUM_LAYERS:-9}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-5472}"
export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-704}"
export MOE_LAYER_FREQ="${MOE_LAYER_FREQ:-[0]*1+[1]*8}"
export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-4}"
export NUM_EXPERTS="${NUM_EXPERTS:-7}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-2}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=1
export TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-local}"
export PRECISION="bf16"
export TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY="${TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY:-$FREEZE_SHARED}"

export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-300}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export TENSORBOARD_LOG_INTERVAL="${TENSORBOARD_LOG_INTERVAL:-1}"
export OLD_MODEL_KL_COEFF="${OLD_MODEL_KL_COEFF:-1.0}"
export OLD_MODEL_KL_TEMPERATURE="${OLD_MODEL_KL_TEMPERATURE:-1.0}"
export OLD_HIDDEN_KL_COEFF="${OLD_HIDDEN_KL_COEFF:-1.0}"
export OLD_HIDDEN_KL_TEMPERATURE="${OLD_HIDDEN_KL_TEMPERATURE:-1.0}"
export OLD_HIDDEN_KL_LAYERS="${OLD_HIDDEN_KL_LAYERS:-all_but_last}"
export OLD_HIDDEN_MSE_COEFF="${OLD_HIDDEN_MSE_COEFF:-1.0}"
export OLD_HIDDEN_MSE_LAYERS="${OLD_HIDDEN_MSE_LAYERS:-all}"
export MOE_EXPANSION_DISTILL_MODE="${MOE_EXPANSION_DISTILL_MODE:-none}"
export MOE_EXPANSION_DISTILL_LM_LOSS_COEFF="${MOE_EXPANSION_DISTILL_LM_LOSS_COEFF:-1.0}"
export MOE_EXPANSION_DISTILL_HIDDEN_MSE_COEFF="${MOE_EXPANSION_DISTILL_HIDDEN_MSE_COEFF:-1.0}"
export MOE_EXPANSION_DISTILL_ROUTER_KL_COEFF="${MOE_EXPANSION_DISTILL_ROUTER_KL_COEFF:-1.0}"
export MOE_EXPANSION_DISTILL_HIDDEN_LAYERS="${MOE_EXPANSION_DISTILL_HIDDEN_LAYERS:-all}"
if [ -z "${ENABLE_OLD_MODEL_KL:-}" ]; then
    if [ "$MOE_EXPANSION_DISTILL_MODE" != "none" ]; then
        export ENABLE_OLD_MODEL_KL=1
    elif [ "$TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY" = "1" ]; then
        export ENABLE_OLD_MODEL_KL=0
    else
        export ENABLE_OLD_MODEL_KL=1
    fi
else
    export ENABLE_OLD_MODEL_KL
fi
export GPU_LOG_INTERVAL_SECONDS="${GPU_LOG_INTERVAL_SECONDS:-30}"
export LR="${LR:-3e-4}"
export MIN_LR="${MIN_LR:-3e-5}"
export LR_DECAY_STYLE="${LR_DECAY_STYLE:-WSD}"
export LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.01}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-$TRAIN_ITERS}"
export LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-$((TRAIN_ITERS / 10))}"
export MOE_NEW_EXPERT_LR_RAMP_STEPS="${MOE_NEW_EXPERT_LR_RAMP_STEPS:-0}"
export MOE_ROUTER_LR_MULTIPLIER="${MOE_ROUTER_LR_MULTIPLIER:-1.0}"
export MOE_SEPARATE_ROUTER_EXPERT_GRAD_CLIP="${MOE_SEPARATE_ROUTER_EXPERT_GRAD_CLIP:-0}"
export MOE_ALLOW_PARTIAL_OPTIMIZER_STATE="${MOE_ALLOW_PARTIAL_OPTIMIZER_STATE:-0}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-scripts/experiment/a100/flame-moe-bf16-no-shared.sh}"
export DATASET_SPLIT="${DATASET_SPLIT:-100,0,0}"

export TRAIN_DATASET="${TRAIN_DATASET:-$(dataset_dir_for_task "$TARGET_TASK")}"
export TRAIN_DATASET_SECONDARY="${TRAIN_DATASET_SECONDARY:-}"
export TRAIN_DATA_WEIGHT_MODE="${TRAIN_DATA_WEIGHT_MODE:-equal_prefix}"
export INTERLEAVE_WIKI_DATASET="${INTERLEAVE_WIKI_DATASET:-$(dataset_dir_for_task wiki)}"
export INTERLEAVE_SECONDARY_DATASET="${INTERLEAVE_SECONDARY_DATASET:-}"
export INTERLEAVE_ROUTER_DATA_WEIGHT_MODE="${INTERLEAVE_ROUTER_DATA_WEIGHT_MODE:-equal_prefix}"
export SOURCE_RUN_SUBDIR="${SOURCE_RUN_SUBDIR:-$(weights_subdir_for_task "$SOURCE_TASK")}"
export SOURCE_WEIGHTS_DIR="${SOURCE_WEIGHTS_DIR:-}"
export SOURCE_REQUIRED_ITERS="${SOURCE_REQUIRED_ITERS:-1}"
export SOURCE_RUN_ID="${SOURCE_RUN_ID:-}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$LOCAL_WEIGHTS/$STAGE_DIR_NAME/$RUN_ID}"
export LOG_DIR="${LOG_DIR:-$TRAIN_WEIGHTS/logs}"
export RUN_LOG="${RUN_LOG:-$LOG_DIR/${STAGE_LABEL}.log}"
export GPU_LOG="${GPU_LOG:-$LOG_DIR/gpu_usage.csv}"
export RUN_METADATA="${RUN_METADATA:-$LOG_DIR/run_metadata.json}"
export DATASET_NAME="${DATASET_NAME:-$(dataset_name_for_task "$TARGET_TASK")}"
export DATASET_SOURCE="${DATASET_SOURCE:-$(dataset_source_for_task "$TARGET_TASK")}"
export PROBE_DATASET="${PROBE_DATASET:-$(probe_dir_for_task "$TARGET_TASK")}"
export PROBE_NAME="${PROBE_NAME:-${TARGET_TASK}_probe}"
export PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-25}"
export PROBE_MICRO_BATCH_SIZE="${PROBE_MICRO_BATCH_SIZE:-}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-100}"
export SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET:-$(probe_dir_for_task "$SOURCE_TASK")}"
export SECONDARY_PROBE_NAME="${SECONDARY_PROBE_NAME:-${SOURCE_TASK}_probe}"
export SECONDARY_PROBE_EVAL_ITERS="${SECONDARY_PROBE_EVAL_ITERS:-25}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-100}"
export TERTIARY_PROBE_DATASET="${TERTIARY_PROBE_DATASET:-}"
export TERTIARY_PROBE_NAME="${TERTIARY_PROBE_NAME:-}"
export TERTIARY_PROBE_EVAL_ITERS="${TERTIARY_PROBE_EVAL_ITERS:-25}"
export TERTIARY_PROBE_EVAL_INTERVAL="${TERTIARY_PROBE_EVAL_INTERVAL:-0}"
export PROBE_STEP_OFFSET="${PROBE_STEP_OFFSET:-}"
export SECONDARY_PROBE_STEP_OFFSET="${SECONDARY_PROBE_STEP_OFFSET:-}"
export TERTIARY_PROBE_STEP_OFFSET="${TERTIARY_PROBE_STEP_OFFSET:-}"
export RUN_INITIAL_PROBE_EVAL="${RUN_INITIAL_PROBE_EVAL:-1}"
export RUN_INITIAL_VALID_EVAL="${RUN_INITIAL_VALID_EVAL:-1}"
export WANDB_STEP_OFFSET="${WANDB_STEP_OFFSET:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-$RUN_ID}"
export WANDB_SAVE_DIR="${WANDB_SAVE_DIR:-$TRAIN_WEIGHTS/wandb}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$RUN_ID}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND="${LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND:-1}"
export RESUME_CONTINUAL_FROM_TRAIN_WEIGHTS="${RESUME_CONTINUAL_FROM_TRAIN_WEIGHTS:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

if [ "$TRAIN_ATTENTION_WITH_NEW_EXPERTS" = "1" ] && [ "$TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY" != "1" ]; then
    echo "ERROR: TRAIN_ATTENTION_WITH_NEW_EXPERTS=1 requires TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY=1." >&2
    exit 1
fi

export SOURCE_WEIGHTS_DIR="$(resolve_completed_run_dir)"
case "$STAGE_INPUTS_TO_SCRATCH" in 0|1) ;; *) echo "ERROR: STAGE_INPUTS_TO_SCRATCH must be 0 or 1" >&2; exit 1 ;; esac
case "$DIRECT_LOCAL_SAVE" in 0|1) ;; *) echo "ERROR: DIRECT_LOCAL_SAVE must be 0 or 1" >&2; exit 1 ;; esac
[[ -d "$TRAIN_DATASET" ]] || { echo "ERROR: training dataset missing: $TRAIN_DATASET" >&2; exit 1; }
if [ "$MOE_JOINT_REPLAY_LM" = "1" ]; then
    [[ -n "$JOINT_REPLAY_DATASET" && -d "$JOINT_REPLAY_DATASET" ]] || {
        echo "ERROR: joint replay dataset missing: ${JOINT_REPLAY_DATASET:-<empty>}" >&2
        exit 1
    }
fi

if [ "$STAGE_INPUTS_TO_SCRATCH" = "0" ]; then
    # Inputs already live on /data2. Use the persistent paths directly rather
    # than duplicating checkpoints, datasets, replay data, and GT on /data2.
    export SSD_TRAIN_DATASET="$TRAIN_DATASET"
    export SSD_TRAIN_DATASET_SECONDARY="$TRAIN_DATASET_SECONDARY"
    export SSD_SOURCE_WEIGHTS="$SOURCE_WEIGHTS_DIR"
    export SSD_TARGET_WEIGHTS="$TRAIN_WEIGHTS"
    export SSD_INTERLEAVE_WIKI_DATASET="$INTERLEAVE_WIKI_DATASET"
    export SSD_INTERLEAVE_SECONDARY_DATASET="$INTERLEAVE_SECONDARY_DATASET"
    export SSD_JOINT_REPLAY_DATASET="$JOINT_REPLAY_DATASET"
    export SSD_JOINT_REPLAY_SECONDARY_DATASET="$JOINT_REPLAY_SECONDARY_DATASET"
    export SSD_OLD_LIKE_GT="$MOE_JOINT_REPLAY_OLD_LIKE_GT_PATH"
elif [ "$DIRECT_LOCAL_SAVE" = "1" ]; then
    export SSD_TARGET_WEIGHTS="$TRAIN_WEIGHTS"
fi

export PROBE_STEP_OFFSET="${PROBE_STEP_OFFSET:-$(read_train_iters_from_run)}"
export SECONDARY_PROBE_STEP_OFFSET="${SECONDARY_PROBE_STEP_OFFSET:-$PROBE_STEP_OFFSET}"
export TERTIARY_PROBE_STEP_OFFSET="${TERTIARY_PROBE_STEP_OFFSET:-$PROBE_STEP_OFFSET}"
export WANDB_STEP_OFFSET="${WANDB_STEP_OFFSET:-$PROBE_STEP_OFFSET}"

RESUME_FROM_TARGET=0
if [ "$RESUME_CONTINUAL_FROM_TRAIN_WEIGHTS" = "1" ] && {
    [ -f "$TRAIN_WEIGHTS/latest_checkpointed_iteration.txt" ] ||
    [ -f "$TRAIN_WEIGHTS/non_persistent/latest_checkpointed_iteration.txt" ];
}; then
    RESUME_FROM_TARGET=1
fi

case "$SOURCE_TASK" in
    wiki)
        export SOURCE_PRIMARY_PROBE_CANDIDATES="${SOURCE_PRIMARY_PROBE_CANDIDATES:-wiki_probe,wiki_a_probe}"
        ;;
    code)
        export SOURCE_PRIMARY_PROBE_CANDIDATES="${SOURCE_PRIMARY_PROBE_CANDIDATES:-code_probe,code_b_probe}"
        ;;
    conversation)
        export SOURCE_PRIMARY_PROBE_CANDIDATES="${SOURCE_PRIMARY_PROBE_CANDIDATES:-conversation_probe,conversation_c_probe}"
        ;;
esac

case "$TARGET_TASK" in
    wiki)
        export SOURCE_SECONDARY_PROBE_CANDIDATES="${SOURCE_SECONDARY_PROBE_CANDIDATES:-wiki_probe,wiki_a_probe}"
        ;;
    code)
        export SOURCE_SECONDARY_PROBE_CANDIDATES="${SOURCE_SECONDARY_PROBE_CANDIDATES:-code_probe,code_b_probe}"
        ;;
    conversation)
        export SOURCE_SECONDARY_PROBE_CANDIDATES="${SOURCE_SECONDARY_PROBE_CANDIDATES:-conversation_probe,conversation_c_probe}"
        ;;
esac

if [ "$LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND" = "1" ] && [ -n "$WANDB_PROJECT" ]; then
    export RUN_INITIAL_PROBE_EVAL=0
fi

mkdir -p "$SSD_TRAIN_DATASET" "$SSD_SOURCE_WEIGHTS" "$SSD_TARGET_WEIGHTS" "$TRAIN_WEIGHTS" "$LOG_DIR"
if [ -n "$TRAIN_DATASET_SECONDARY" ]; then
    mkdir -p "$SSD_TRAIN_DATASET_SECONDARY"
fi
if [ "$MOE_JOINT_REPLAY_LM" = "1" ]; then
    mkdir -p "$SSD_JOINT_REPLAY_DATASET"
    [ -z "$JOINT_REPLAY_SECONDARY_DATASET" ] || mkdir -p "$SSD_JOINT_REPLAY_SECONDARY_DATASET"
fi
if [ -n "$MOE_JOINT_REPLAY_OLD_LIKE_GT_PATH" ]; then
    [ "$MOE_JOINT_REPLAY_LM" = "1" ] || { echo "ERROR: old-like GT requires joint replay" >&2; exit 1; }
    [ -f "$MOE_JOINT_REPLAY_OLD_LIKE_GT_PATH/metadata.json" ] || {
        echo "ERROR: old-like GT metadata missing: $MOE_JOINT_REPLAY_OLD_LIKE_GT_PATH/metadata.json" >&2
        exit 1
    }
    mkdir -p "$SSD_OLD_LIKE_GT"
fi
if [ "$MOE_INTERLEAVE_CODE_STEPS" -gt 0 ]; then
    mkdir -p "$SSD_INTERLEAVE_WIKI_DATASET"
    if [ -n "$INTERLEAVE_SECONDARY_DATASET" ]; then
        mkdir -p "$SSD_INTERLEAVE_SECONDARY_DATASET"
    fi
fi

if [ "$STORAGE_PLAN_ONLY" = "1" ]; then
    printf 'STAGE_INPUTS_TO_SCRATCH=%s\nSOURCE=%s\nDATASET=%s\nREPLAY=%s\nGT=%s\nSAVE=%s\n' \
        "$STAGE_INPUTS_TO_SCRATCH" "$SSD_SOURCE_WEIGHTS" "$SSD_TRAIN_DATASET" \
        "$SSD_JOINT_REPLAY_DATASET" "$SSD_OLD_LIKE_GT" "$SSD_TARGET_WEIGHTS"
    exit 0
fi

GPU_LOG_PID=""
SYNC_DONE=0
cleanup() {
    local exit_code=$?
    if [ -n "${GPU_LOG_PID:-}" ]; then
        kill "$GPU_LOG_PID" 2>/dev/null || true
    fi
    if [ "${SYNC_DONE:-0}" != "1" ] && [ -d "$SSD_TARGET_WEIGHTS" ] && [ "$SSD_TARGET_WEIGHTS" != "$TRAIN_WEIGHTS" ]; then
        rsync -rlptD "$SSD_TARGET_WEIGHTS/" "$TRAIN_WEIGHTS/" || true
        SYNC_DONE=1
    fi
    return "$exit_code"
}
trap cleanup EXIT INT TERM

exec > >(
    tee -a "$RUN_LOG" | "$PYTHON_BIN" -u -c '
from datetime import datetime, timedelta
import re, sys
iter_re = re.compile(r"(\[[^]]+\]) iteration\s+(\d+)/\s*(\d+)")
elapsed_re = re.compile(r"elapsed time per iteration \(ms\):\s*([0-9.]+)")
tflops_re = re.compile(r"throughput per GPU \(TFLOP/s/GPU\):\s*([0-9.]+)")
val_re = re.compile(r"validation loss at iteration\s+(\d+).*lm loss value:\s*([^|]+)")
save_re = re.compile(r"saving checkpoint at iteration\s+(\d+)")
keep_re = re.compile(r"a100 bf16 continual|ERROR:|Traceback|failed \(exitcode|checkpoint at|probe |Expanded MoE checkpoint")

def parse_timestamp(value):
    try:
        return datetime.strptime(value.strip("[]"), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None

def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"

for line in sys.stdin:
    line = line.rstrip("\n")
    m = iter_re.search(line)
    if m:
        step = int(m.group(2))
        total = int(m.group(3))
        parts = [f"{m.group(1)} step {step}/{total}"]
        elapsed = elapsed_re.search(line)
        if elapsed:
            seconds_per_iter = float(elapsed.group(1)) / 1000.0
            remaining_seconds = max(0, total - step) * seconds_per_iter
            parts.append(f"{seconds_per_iter:.1f}s/it")
            if step < total:
                parts.append(f"ETA {format_duration(remaining_seconds)}")
                ts = parse_timestamp(m.group(1))
                if ts is not None:
                    end_at = ts + timedelta(seconds=remaining_seconds)
                    parts.append(f"end {end_at:%Y-%m-%d %H:%M:%S}")
        tflops = tflops_re.search(line)
        if tflops:
            parts.append(f"GPU {float(tflops.group(1)):.1f} TFLOP/s")
        print(" | ".join(parts), flush=True)
        continue
    m = val_re.search(line)
    if m:
        print(f"validation step {m.group(1)} | lm loss {m.group(2).strip()}", flush=True)
        continue
    m = save_re.search(line)
    if m:
        print(f"saving checkpoint step {m.group(1)}", flush=True)
        continue
    if keep_re.search(line):
        print(line, flush=True)
'
) 2>&1

echo "a100 bf16 continual run log: $RUN_LOG"
echo "a100 bf16 continual gpu log: $GPU_LOG"
echo "a100 bf16 continual metadata: $RUN_METADATA"
echo "a100 bf16 source checkpoint: $SOURCE_WEIGHTS_DIR"
echo "a100 bf16 input staging: $([ "$STAGE_INPUTS_TO_SCRATCH" = 1 ] && echo enabled || echo disabled)"
if [ "$RESUME_FROM_TARGET" = "1" ]; then
    echo "a100 bf16 continual resume checkpoint: $TRAIN_WEIGHTS"
fi

if [ "$STAGE_INPUTS_TO_SCRATCH" = "1" ]; then
    rsync -rlptD \
        --exclude 'logs/' \
        --exclude 'wandb/' \
        --exclude 'events.out.tfevents*' \
        --exclude 'progress.txt' \
        "$SOURCE_WEIGHTS_DIR/" "$SSD_SOURCE_WEIGHTS/"
    rsync -rlptD --info=progress2 "$TRAIN_DATASET/" "$SSD_TRAIN_DATASET/"
    if [ -n "$TRAIN_DATASET_SECONDARY" ]; then
        rsync -rlptD --info=progress2 "$TRAIN_DATASET_SECONDARY/" "$SSD_TRAIN_DATASET_SECONDARY/"
    fi
    if [ "$MOE_JOINT_REPLAY_LM" = "1" ]; then
        rsync -rlptD --info=progress2 "$JOINT_REPLAY_DATASET/" "$SSD_JOINT_REPLAY_DATASET/"
        [ -z "$JOINT_REPLAY_SECONDARY_DATASET" ] || rsync -rlptD --info=progress2 "$JOINT_REPLAY_SECONDARY_DATASET/" "$SSD_JOINT_REPLAY_SECONDARY_DATASET/"
    fi
    if [ -n "$MOE_JOINT_REPLAY_OLD_LIKE_GT_PATH" ]; then
        rsync -rlptD --delete "$MOE_JOINT_REPLAY_OLD_LIKE_GT_PATH/" "$SSD_OLD_LIKE_GT/"
    fi
    if [ "$MOE_INTERLEAVE_CODE_STEPS" -gt 0 ]; then
        rsync -rlptD --info=progress2 "$INTERLEAVE_WIKI_DATASET/" "$SSD_INTERLEAVE_WIKI_DATASET/"
        if [ -n "$INTERLEAVE_SECONDARY_DATASET" ]; then
            rsync -rlptD --info=progress2 "$INTERLEAVE_SECONDARY_DATASET/" "$SSD_INTERLEAVE_SECONDARY_DATASET/"
        fi
    fi
fi
if [ "$RESUME_FROM_TARGET" = "1" ] && [ "$SSD_TARGET_WEIGHTS" != "$TRAIN_WEIGHTS" ]; then
    rsync -rlptD \
        --exclude 'logs/' \
        --exclude 'wandb/' \
        --exclude 'events.out.tfevents*' \
        --exclude 'progress.txt' \
        "$TRAIN_WEIGHTS/" "$SSD_TARGET_WEIGHTS/"
fi

write_continual_metadata
source "$MODEL_CONFIG_SCRIPT"

INFRA_ARGS=(
    --transformer-impl "$TRANSFORMER_IMPL"
    --pipeline-model-parallel-size "$PIPELINE_MODEL_PARALLEL_SIZE"
    --expert-model-parallel-size "$EXPERT_MODEL_PARALLEL_SIZE"
    --moe-token-dispatcher-type alltoall
    --distributed-timeout-minutes 30
    --no-persist-layer-norm
)

TRAIN_ARGS=(
    --bf16
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --lr "$LR"
    --min-lr "$MIN_LR"
    --lr-decay-style "$LR_DECAY_STYLE"
    --lr-decay-iters "$LR_DECAY_ITERS"
    --lr-warmup-fraction "$LR_WARMUP_FRACTION"
    --lr-wsd-decay-iters "$LR_WSD_DECAY_ITERS"
    --train-iters "$TRAIN_ITERS"
)

TRAIN_DATA_PATH_DIRS=("$SSD_TRAIN_DATASET")
if [ -n "$TRAIN_DATASET_SECONDARY" ]; then
    TRAIN_DATA_PATH_DIRS+=("$SSD_TRAIN_DATASET_SECONDARY")
fi
case "$TRAIN_DATA_WEIGHT_MODE" in
    equal_dataset)
        TRAIN_DATA_PATH="$(build_equal_dataset_data_path "${TRAIN_DATA_PATH_DIRS[@]}")"
        ;;
    equal_prefix)
        TRAIN_DATA_PATH="$(build_data_path "${TRAIN_DATA_PATH_DIRS[@]}")"
        ;;
    *)
        echo "ERROR: unsupported TRAIN_DATA_WEIGHT_MODE=$TRAIN_DATA_WEIGHT_MODE" >&2
        exit 1
        ;;
esac

DATA_ARGS=(
    --seq-length "${SEQ_LENGTH:-512}"
    --data-path $TRAIN_DATA_PATH
    --split "$DATASET_SPLIT"
)

INTERLEAVE_ARGS=()
if [ "$MOE_JOINT_REPLAY_LM" = "1" ]; then
    [ "$MOE_INTERLEAVE_CODE_STEPS" -eq 0 ] || { echo "ERROR: joint replay conflicts with interleave" >&2; exit 1; }
    JOINT_DIRS=("$SSD_JOINT_REPLAY_DATASET")
    [ -z "$JOINT_REPLAY_SECONDARY_DATASET" ] || JOINT_DIRS+=("$SSD_JOINT_REPLAY_SECONDARY_DATASET")
    # third replay source (read in place; the router-FT subsets are small)
    [ -z "${JOINT_REPLAY_TERTIARY_DATASET:-}" ] || JOINT_DIRS+=("$JOINT_REPLAY_TERTIARY_DATASET")
    case "$JOINT_REPLAY_DATA_WEIGHT_MODE" in
      equal_dataset) JOINT_PATH="$(build_equal_dataset_data_path "${JOINT_DIRS[@]}")" ;;
      equal_prefix) JOINT_PATH="$(build_data_path "${JOINT_DIRS[@]}")" ;;
      *) echo "ERROR: invalid joint replay weight mode" >&2; exit 1 ;;
    esac
    INTERLEAVE_ARGS+=(--moe-joint-replay-lm --moe-joint-replay-data-path $JOINT_PATH)
    if [ "$MOE_JOINT_REPLAY_TOTAL_SAMPLES" -gt 0 ]; then
        INTERLEAVE_ARGS+=(
            --moe-joint-replay-total-samples "$MOE_JOINT_REPLAY_TOTAL_SAMPLES"
            --moe-joint-replay-micro-batch-size "$MOE_JOINT_REPLAY_MICRO_BATCH_SIZE"
        )
    fi
    if [ -n "$MOE_JOINT_REPLAY_OLD_LIKE_GT_PATH" ]; then
        INTERLEAVE_ARGS+=(
            --moe-joint-replay-old-like-gt-path "$SSD_OLD_LIKE_GT"
            --moe-joint-replay-old-like-unit "$MOE_JOINT_REPLAY_OLD_LIKE_UNIT"
            --moe-joint-replay-old-like-target-train-fraction "$MOE_JOINT_REPLAY_OLD_LIKE_TARGET_TRAIN_FRACTION"
            --moe-joint-replay-old-like-selected-token-count "$MOE_JOINT_REPLAY_OLD_LIKE_SELECTED_TOKEN_COUNT"
            --moe-joint-replay-old-like-positive-sample-count "$MOE_JOINT_REPLAY_OLD_LIKE_POSITIVE_SAMPLE_COUNT"
            --moe-joint-replay-old-like-full-train-token-count "$MOE_JOINT_REPLAY_OLD_LIKE_FULL_TRAIN_TOKEN_COUNT"
        )
    fi
    if [ "$MOE_JOINT_NEW_EXPERT_QUOTA" != "0" ] && [ "$MOE_JOINT_NEW_EXPERT_QUOTA" != "0.0" ]; then
        INTERLEAVE_ARGS+=(--moe-joint-new-expert-quota "$MOE_JOINT_NEW_EXPERT_QUOTA")
    fi
    if [ -n "$MOE_JOINT_NEW_EXPERT_QUOTA_SCHEDULE" ]; then
        INTERLEAVE_ARGS+=(--moe-joint-new-expert-quota-schedule "$MOE_JOINT_NEW_EXPERT_QUOTA_SCHEDULE")
    fi
    if [ -n "$MOE_JOINT_NEW_EXPERT_QUOTA_MIN_NEW_SLOTS" ]; then
        INTERLEAVE_ARGS+=(--moe-joint-new-expert-quota-min-new-slots "$MOE_JOINT_NEW_EXPERT_QUOTA_MIN_NEW_SLOTS")
    fi
    INTERLEAVE_ARGS+=(--moe-joint-new-expert-quota-loss-coeff "$MOE_JOINT_NEW_EXPERT_QUOTA_LOSS_COEFF")
    if [ "$MOE_JOINT_NEW_EXPERT_QUOTA_PRESERVE_NATURAL_GRADS" = "1" ]; then
        INTERLEAVE_ARGS+=(--moe-joint-new-expert-quota-preserve-natural-grads)
    fi
    if [ "$MOE_JOINT_REPLAY_OLD_DATA_KD" = "1" ]; then
        [ "$ENABLE_OLD_MODEL_KL" = "1" ] || { echo "ERROR: old-data KD replay requires ENABLE_OLD_MODEL_KL=1" >&2; exit 1; }
        [ -n "$OLD_MODEL_KL_WEIGHTS_DIR" ] || { echo "ERROR: old-data KD replay requires OLD_MODEL_KL_WEIGHTS_DIR" >&2; exit 1; }
        [ -d "$OLD_MODEL_KL_WEIGHTS_DIR" ] || { echo "ERROR: teacher checkpoint not found: $OLD_MODEL_KL_WEIGHTS_DIR" >&2; exit 1; }
        INTERLEAVE_ARGS+=(--moe-joint-replay-old-data-kd)
    fi
    if [ "$MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL" = "1" ]; then
        [ "$ENABLE_OLD_MODEL_KL" = "1" ] || { echo "ERROR: old-data hidden KL replay requires ENABLE_OLD_MODEL_KL=1" >&2; exit 1; }
        [ -n "$OLD_MODEL_KL_WEIGHTS_DIR" ] || { echo "ERROR: old-data hidden KL replay requires OLD_MODEL_KL_WEIGHTS_DIR" >&2; exit 1; }
        [ -d "$OLD_MODEL_KL_WEIGHTS_DIR" ] || { echo "ERROR: teacher checkpoint not found: $OLD_MODEL_KL_WEIGHTS_DIR" >&2; exit 1; }
        INTERLEAVE_ARGS+=(
            --moe-joint-replay-old-data-hidden-kl
            --moe-old-hidden-kl-coeff "$OLD_HIDDEN_KL_COEFF"
            --moe-old-hidden-kl-temperature "$OLD_HIDDEN_KL_TEMPERATURE"
            --moe-old-hidden-kl-layers "$OLD_HIDDEN_KL_LAYERS"
        )
        if [ "$OLD_HIDDEN_KL_COEFF_DECAY_STEPS" -gt 0 ]; then
            [ -n "$OLD_HIDDEN_KL_COEFF_START" ] || { echo "ERROR: hidden-KL scheduling requires OLD_HIDDEN_KL_COEFF_START" >&2; exit 1; }
            INTERLEAVE_ARGS+=(
                --moe-old-hidden-kl-coeff-start "$OLD_HIDDEN_KL_COEFF_START"
                --moe-old-hidden-kl-coeff-decay-steps "$OLD_HIDDEN_KL_COEFF_DECAY_STEPS"
            )
        fi
    fi
    if [ "$MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE" = "1" ]; then
        [ "$ENABLE_OLD_MODEL_KL" = "1" ] || { echo "ERROR: old-data hidden MSE replay requires ENABLE_OLD_MODEL_KL=1" >&2; exit 1; }
        [ -n "$OLD_MODEL_KL_WEIGHTS_DIR" ] || { echo "ERROR: old-data hidden MSE replay requires OLD_MODEL_KL_WEIGHTS_DIR" >&2; exit 1; }
        [ -d "$OLD_MODEL_KL_WEIGHTS_DIR" ] || { echo "ERROR: teacher checkpoint not found: $OLD_MODEL_KL_WEIGHTS_DIR" >&2; exit 1; }
        INTERLEAVE_ARGS+=(
            --moe-joint-replay-old-data-hidden-mse
            --moe-old-hidden-mse-coeff "$OLD_HIDDEN_MSE_COEFF"
            --moe-old-hidden-mse-layers "$OLD_HIDDEN_MSE_LAYERS"
        )
    fi
fi
if [ "$MOE_INTERLEAVE_CODE_STEPS" -gt 0 ]; then
    INTERLEAVE_ROUTER_DATA_PATH_DIRS=("$SSD_INTERLEAVE_WIKI_DATASET")
    if [ -n "$INTERLEAVE_SECONDARY_DATASET" ]; then
        INTERLEAVE_ROUTER_DATA_PATH_DIRS+=("$SSD_INTERLEAVE_SECONDARY_DATASET")
    fi
    INTERLEAVE_ROUTER_DATA_PATH_DIRS+=("$SSD_TRAIN_DATASET")
    case "$INTERLEAVE_ROUTER_DATA_WEIGHT_MODE" in
        equal_dataset)
            INTERLEAVE_ROUTER_DATA_PATH="$(build_equal_dataset_data_path "${INTERLEAVE_ROUTER_DATA_PATH_DIRS[@]}")"
            ;;
        equal_prefix)
            INTERLEAVE_ROUTER_DATA_PATH="$(build_data_path "${INTERLEAVE_ROUTER_DATA_PATH_DIRS[@]}")"
            ;;
        *)
            echo "ERROR: unsupported INTERLEAVE_ROUTER_DATA_WEIGHT_MODE=$INTERLEAVE_ROUTER_DATA_WEIGHT_MODE" >&2
            exit 1
            ;;
    esac
    INTERLEAVE_ARGS+=(
        --moe-interleave-code-steps "$MOE_INTERLEAVE_CODE_STEPS"
        --moe-interleave-router-steps "$MOE_INTERLEAVE_ROUTER_STEPS"
        --moe-interleave-code-total-steps "$MOE_INTERLEAVE_CODE_TOTAL_STEPS"
        --moe-interleave-router-data-path $INTERLEAVE_ROUTER_DATA_PATH
        --moe-interleave-code-lr "${MOE_INTERLEAVE_CODE_LR:-$LR}"
        --moe-interleave-code-min-lr "${MOE_INTERLEAVE_CODE_MIN_LR:-$MIN_LR}"
        --moe-interleave-router-lr "${MOE_INTERLEAVE_ROUTER_LR:-$LR}"
        --moe-interleave-router-min-lr "${MOE_INTERLEAVE_ROUTER_MIN_LR:-$MIN_LR}"
        --moe-interleave-code-aux-loss-coeff "${MOE_INTERLEAVE_CODE_AUX_LOSS_COEFF:-${MOE_AUX_LOSS_COEFF:-0.01}}"
        --moe-interleave-code-z-loss-coeff "${MOE_INTERLEAVE_CODE_Z_LOSS_COEFF:-${MOE_Z_LOSS_COEFF:-0.001}}"
        --moe-interleave-router-aux-loss-coeff "${MOE_INTERLEAVE_ROUTER_AUX_LOSS_COEFF:-0.0}"
        --moe-interleave-router-z-loss-coeff "${MOE_INTERLEAVE_ROUTER_Z_LOSS_COEFF:-0.0}"
    )
    if [ "$MOE_INTERLEAVE_ROUTER_AFTER_FINAL" = "1" ]; then
        INTERLEAVE_ARGS+=(--moe-interleave-router-after-final)
    fi
fi

SAVE_ARGS=(
    --log-interval "$LOG_INTERVAL"
    --tensorboard-log-interval "$TENSORBOARD_LOG_INTERVAL"
    --log-throughput
    --log-progress
    --save "$SSD_TARGET_WEIGHTS"
    --save-interval "$SAVE_INTERVAL"
    --eval-interval "$EVAL_INTERVAL"
    --tensorboard-dir "$SSD_TARGET_WEIGHTS"
    --moe-freeze-existing-experts
    --moe-freeze-existing-router
)
if [ "$NO_SAVE_OPTIM" = "1" ]; then
    SAVE_ARGS+=(--no-save-optim)
fi
if [ "$RECOVERY_SAVE_INTERVAL" -gt 0 ]; then
    SAVE_ARGS+=(
        --non-persistent-save-interval "$RECOVERY_SAVE_INTERVAL"
        --non-persistent-ckpt-type global
    )
fi

if [ "$RESUME_FROM_TARGET" = "1" ]; then
    SAVE_ARGS+=(
        --load "$SSD_TARGET_WEIGHTS"
        --moe-resume-from-num-experts "$SOURCE_NUM_EXPERTS"
    )
    if [ "$MOE_INTERLEAVE_CODE_STEPS" -gt 0 ] || [ "$NO_SAVE_OPTIM" = "1" ]; then
        SAVE_ARGS+=(--no-load-optim --no-load-rng)
    fi
elif [ "$LOAD_EXPANDED_SOURCE" = "1" ]; then
    # Source is already expanded to NUM_EXPERTS: load fresh (iteration 0) without
    # re-expanding, reapplying the new-experts+router freeze via resume-from.
    SAVE_ARGS+=(
        --load "$SSD_SOURCE_WEIGHTS"
        --no-load-optim
        --no-load-rng
        --finetune
        --moe-resume-from-num-experts "$SOURCE_NUM_EXPERTS"
    )
else
    SAVE_ARGS+=(
        --load "$SSD_SOURCE_WEIGHTS"
        --no-load-optim
        --no-load-rng
        --finetune
        --moe-expand-from-num-experts "$SOURCE_NUM_EXPERTS"
        --moe-resume-from-num-experts "$SOURCE_NUM_EXPERTS"
    )
fi

if [ "$RUN_INITIAL_VALID_EVAL" = "1" ]; then
    SAVE_ARGS+=(--run-initial-valid-eval)
fi

if [ "$TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY" = "1" ]; then
    SAVE_ARGS+=(--moe-train-new-experts-and-router-only)
    if [ "$TRAIN_ATTENTION_WITH_NEW_EXPERTS" = "1" ]; then
        SAVE_ARGS+=(--moe-train-attention-with-new-experts)
    fi
    if [ "$FREEZE_DENSE_ATTENTION_LORA_WITH_NEW_EXPERTS" = "1" ]; then
        SAVE_ARGS+=(--moe-freeze-dense-attention-lora-with-new-experts)
    fi
fi
if [ "$MOE_NEW_EXPERT_LR_RAMP_STEPS" -gt 0 ]; then
    SAVE_ARGS+=(--moe-new-expert-lr-ramp-steps "$MOE_NEW_EXPERT_LR_RAMP_STEPS")
fi
SAVE_ARGS+=(--moe-router-lr-multiplier "$MOE_ROUTER_LR_MULTIPLIER")
if [ "$MOE_SEPARATE_ROUTER_EXPERT_GRAD_CLIP" = "1" ]; then
    SAVE_ARGS+=(--moe-separate-router-expert-grad-clip)
fi
if [ "$MOE_ALLOW_PARTIAL_OPTIMIZER_STATE" = "1" ]; then
    SAVE_ARGS+=(--moe-allow-partial-optimizer-state)
fi
if [ "$LOG_ROUTER_GRAD_NORM_SOURCES" = "1" ]; then
    SAVE_ARGS+=(--log-router-grad-norm-sources)
fi
if [ "$TRAIN_ROUTER_USAGE_LOG_INTERVAL" -gt 0 ]; then
    SAVE_ARGS+=(--train-router-usage-log-interval "$TRAIN_ROUTER_USAGE_LOG_INTERVAL")
    if [ -n "$TRAIN_ROUTER_USAGE_NUM_EXISTING_EXPERTS" ]; then
        SAVE_ARGS+=(--train-router-usage-num-existing-experts "$TRAIN_ROUTER_USAGE_NUM_EXISTING_EXPERTS")
    fi
fi

if [ "$ENABLE_OLD_MODEL_KL" = "1" ] || [ "$MOE_EXPANSION_DISTILL_MODE" != "none" ] || [ "$FINGERPRINT_KD_ENABLE" = "1" ]; then
    OLD_MODEL_KL_LOAD_PATH="${OLD_MODEL_KL_WEIGHTS_DIR:-$SSD_SOURCE_WEIGHTS}"
    SAVE_ARGS+=(
        --moe-old-model-kl-load "$OLD_MODEL_KL_LOAD_PATH"
        --moe-old-model-kl-coeff "$OLD_MODEL_KL_COEFF"
        --moe-old-model-kl-temperature "$OLD_MODEL_KL_TEMPERATURE"
    )
    if [ -n "$OLD_MODEL_KL_NUM_EXPERTS" ]; then
        SAVE_ARGS+=(--moe-old-model-kl-num-experts "$OLD_MODEL_KL_NUM_EXPERTS")
    fi
fi

if [ "$FINGERPRINT_KD_ENABLE" = "1" ]; then
    [ -n "$FINGERPRINT_KD_BUNDLE" ] || { echo "ERROR: fingerprint KD requires FINGERPRINT_KD_BUNDLE" >&2; exit 1; }
    [ -f "$FINGERPRINT_KD_BUNDLE" ] || { echo "ERROR: fingerprint bundle not found: $FINGERPRINT_KD_BUNDLE" >&2; exit 1; }
    SAVE_ARGS+=(
        --fingerprint-kd-coeff "$FINGERPRINT_KD_COEFF"
        --fingerprint-kd-lm-loss-coeff "$FINGERPRINT_KD_LM_LOSS_COEFF"
        --fingerprint-kd-bundle "$FINGERPRINT_KD_BUNDLE"
        --fingerprint-kd-rank "$FINGERPRINT_KD_RANK"
        --fingerprint-kd-layers "$FINGERPRINT_KD_LAYERS"
        --fingerprint-kd-score-representation "$FINGERPRINT_KD_SCORE_REPRESENTATION"
        --fingerprint-kd-loss-representation "$FINGERPRINT_KD_LOSS_REPRESENTATION"
        --fingerprint-kd-gate-mode "$FINGERPRINT_KD_GATE_MODE"
        --fingerprint-kd-threshold "$FINGERPRINT_KD_THRESHOLD"
        --fingerprint-kd-soft-temperature "$FINGERPRINT_KD_SOFT_TEMPERATURE"
        --fingerprint-kd-weight-assignment "$FINGERPRINT_KD_WEIGHT_ASSIGNMENT"
    )
    if [ "$FINGERPRINT_KD_FORCE_ENABLE_ZERO_COEFF" = "1" ]; then
        SAVE_ARGS+=(--fingerprint-kd-force-enable-zero-coeff)
    fi
    if [ "$LOG_FINGERPRINT_GRAD_NORM_GROUPS" = "1" ]; then
        SAVE_ARGS+=(--log-fingerprint-grad-norm-groups)
    fi
fi

if [ "$MOE_EXPANSION_DISTILL_MODE" != "none" ]; then
    SAVE_ARGS+=(
        --moe-expansion-distill-mode "$MOE_EXPANSION_DISTILL_MODE"
        --moe-expansion-distill-lm-loss-coeff "$MOE_EXPANSION_DISTILL_LM_LOSS_COEFF"
        --moe-expansion-distill-hidden-mse-coeff "$MOE_EXPANSION_DISTILL_HIDDEN_MSE_COEFF"
        --moe-expansion-distill-router-kl-coeff "$MOE_EXPANSION_DISTILL_ROUTER_KL_COEFF"
        --moe-expansion-distill-hidden-layers "$MOE_EXPANSION_DISTILL_HIDDEN_LAYERS"
    )
fi

PROBE_ARGS=(
    --probe-name "$PROBE_NAME"
    --probe-eval-iters "$PROBE_EVAL_ITERS"
    --probe-eval-interval "$PROBE_EVAL_INTERVAL"
    --probe-step-offset "$PROBE_STEP_OFFSET"
    --probe-data-path $(build_data_path "$PROBE_DATASET")
)
if [ -n "$PROBE_MICRO_BATCH_SIZE" ]; then
    PROBE_ARGS+=(--probe-micro-batch-size "$PROBE_MICRO_BATCH_SIZE")
fi

if [ "$RUN_INITIAL_PROBE_EVAL" = "1" ]; then
    PROBE_ARGS+=(--run-initial-probe-eval)
fi

if [ -n "$SECONDARY_PROBE_DATASET" ]; then
    PROBE_ARGS+=(
        --secondary-probe-name "$SECONDARY_PROBE_NAME"
        --secondary-probe-eval-iters "$SECONDARY_PROBE_EVAL_ITERS"
        --secondary-probe-eval-interval "$SECONDARY_PROBE_EVAL_INTERVAL"
        --secondary-probe-step-offset "$SECONDARY_PROBE_STEP_OFFSET"
        --secondary-probe-data-path $(build_data_path "$SECONDARY_PROBE_DATASET")
    )
fi

if [ -n "$TERTIARY_PROBE_DATASET" ]; then
    PROBE_ARGS+=(
        --tertiary-probe-name "$TERTIARY_PROBE_NAME"
        --tertiary-probe-eval-iters "$TERTIARY_PROBE_EVAL_ITERS"
        --tertiary-probe-eval-interval "$TERTIARY_PROBE_EVAL_INTERVAL"
        --tertiary-probe-step-offset "$TERTIARY_PROBE_STEP_OFFSET"
        --tertiary-probe-data-path $(build_data_path "$TERTIARY_PROBE_DATASET")
    )
fi

WANDB_ARGS=()
if [ -n "$WANDB_PROJECT" ]; then
    WANDB_ARGS+=(
        --wandb-project "$WANDB_PROJECT"
        --wandb-exp-name "$WANDB_EXP_NAME"
        --wandb-save-dir "$WANDB_SAVE_DIR"
        --wandb-step-offset "$WANDB_STEP_OFFSET"
        --wandb-run-id "$WANDB_RUN_ID"
        --wandb-resume "$WANDB_RESUME"
    )
fi

DEBUG_TRAINABLE_ARGS=()
if [ "${DEBUG_TRAINABLE_PARAMS_AND_EXIT:-0}" = "1" ]; then
    DEBUG_TRAINABLE_ARGS+=(--debug-trainable-params-and-exit)
    if [ -n "${DEBUG_TRAINABLE_PARAMS_PATH:-}" ]; then
        DEBUG_TRAINABLE_ARGS+=(--debug-trainable-params-path "$DEBUG_TRAINABLE_PARAMS_PATH")
    fi
fi

if [ "$LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND" = "1" ] && [ -n "$WANDB_PROJECT" ]; then
    echo "logging source-model probe baseline at step $PROBE_STEP_OFFSET before expert expansion"
    "$PYTHON_BIN" analysis/log_source_probe_baseline_to_wandb.py \
        --source-run-dir "$SOURCE_WEIGHTS_DIR" \
        --project "$WANDB_PROJECT" \
        --run-name "$WANDB_EXP_NAME" \
        --run-id "$WANDB_RUN_ID" \
        --resume "$WANDB_RESUME" \
        --save-dir "$WANDB_SAVE_DIR" \
        --step "$PROBE_STEP_OFFSET" \
        --primary-source-candidates "$SOURCE_SECONDARY_PROBE_CANDIDATES" \
        --primary-target-name "$PROBE_NAME" \
        --secondary-source-candidates "$SOURCE_PRIMARY_PROBE_CANDIDATES" \
        --secondary-target-name "$SECONDARY_PROBE_NAME"
fi

cd Megatron-LM
(
    while true; do
        nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits >> "$GPU_LOG"
        sleep "$GPU_LOG_INTERVAL_SECONDS"
    done
) &
GPU_LOG_PID=$!

"$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nnodes 1 \
    --nproc_per_node "$NPROC_PER_NODE" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    pretrain_gpt.py \
    "${MODEL_ARGS[@]}" "${INFRA_ARGS[@]}" "${TRAIN_ARGS[@]}" \
    "${DATA_ARGS[@]}" "${INTERLEAVE_ARGS[@]}" "${SAVE_ARGS[@]}" \
    "${PROBE_ARGS[@]}" "${WANDB_ARGS[@]}" "${DEBUG_TRAINABLE_ARGS[@]}" \
    ${EXTRA_MEGATRON_ARGS:-}

kill "$GPU_LOG_PID" 2>/dev/null || true
if [ "$SSD_TARGET_WEIGHTS" != "$TRAIN_WEIGHTS" ]; then
    rsync -rlptD "$SSD_TARGET_WEIGHTS/" "$TRAIN_WEIGHTS/"
fi
SYNC_DONE=1

echo "a100 bf16 continual training complete. checkpoint at: $TRAIN_WEIGHTS"
