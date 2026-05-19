#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_LOG_CHECKPOINTS="${WANDB_LOG_CHECKPOINTS:-0}"
export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export STAGE1_REQUIRED_ITERS="${STAGE1_REQUIRED_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-300}"
export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-1}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1800}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export USE_GUARD="${USE_GUARD:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export BASE_STAGE_DIR="${BASE_STAGE_DIR:-$PROJECT_ROOT/.local/weights/a100/mha/shared-router-granularity-qkvo}"

# G2 setting: top-4, 8 -> 16 experts, FFN 352, QKVO full-rank LoRA rank 256.
export RUN_ID="${RUN_ID:-g2-top4-e8to16-ffn352-r256-wiki-to-code-shared-router-qkvo-mha-a100-bf16-mb72-teacher-student-router-kd-fullwiki-allrouter-kl0p1-1800}"
export STAGE1_WEIGHTS_DIR="${STAGE1_WEIGHTS_DIR:-$BASE_STAGE_DIR/wiki/g2-top4-e8-ffn352-r256-wiki-shared-router-qkvo-mha-a100-bf16-mb72-1800}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$BASE_STAGE_DIR/code/$RUN_ID}"

# Full-Wiki teacher-student router KD:
# - every Code iteration also consumes one Wiki global batch,
# - the frozen full Wiki teacher and expanded student are both kept loaded,
# - teacher router outputs use teacher hidden states, student router outputs use
#   student hidden states,
# - by default, the teacher distribution is zero-padded to 16 experts.
#   Set ROUTER_MEMORY_TEACHER_STUDENT_KL_EXISTING_EXPERTS_ONLY=1 to distill only
#   the old/wiki router rows by slicing student logits to the teacher expert count.
export ROUTER_MEMORY_KL_COEFF="${ROUTER_MEMORY_KL_COEFF:-0.1}"
export ROUTER_MEMORY_FORCE_ENABLE_ZERO_COEFF="${ROUTER_MEMORY_FORCE_ENABLE_ZERO_COEFF:-0}"
export ROUTER_MEMORY_FRACTION="${ROUTER_MEMORY_FRACTION:-1.0}"
export ROUTER_MEMORY_INTERVAL="${ROUTER_MEMORY_INTERVAL:-1}"
export ROUTER_MEMORY_DATASET="${ROUTER_MEMORY_DATASET:-$PROJECT_ROOT/data/wiki/train}"
export ROUTER_MEMORY_EVAL_DATASET="${ROUTER_MEMORY_EVAL_DATASET:-$ROUTER_MEMORY_DATASET}"
export ROUTER_MEMORY_EVAL_INTERVAL="${ROUTER_MEMORY_EVAL_INTERVAL:-0}"
export ROUTER_MEMORY_EVAL_ITERS="${ROUTER_MEMORY_EVAL_ITERS:-1}"
export ROUTER_MEMORY_TEACHER_STUDENT_KL=1
export ROUTER_MEMORY_TEACHER_STUDENT_KL_EXISTING_EXPERTS_ONLY="${ROUTER_MEMORY_TEACHER_STUDENT_KL_EXISTING_EXPERTS_ONLY:-0}"
export ROUTER_MEMORY_JOINT_UPDATE=1
export ROUTER_KL_EARLY_STOP_ENABLED=0
export ROUTER_KL_STOP_STEP=""
export SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS="${SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS:-1}"

export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"
export ATTN_LORA_GROUPED_GEMM="${ATTN_LORA_GROUPED_GEMM:-1}"
export MOE_PERMUTE_FUSION="${MOE_PERMUTE_FUSION:-0}"
export MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}"

is_completed() {
    local run_dir="$1"
    local latest_file="${run_dir}/latest_checkpointed_iteration.txt"
    [ -f "$latest_file" ] && [ "$(tr -d '\n\r[:space:]' < "$latest_file")" = "$TRAIN_ITERS" ]
}

has_checkpoint_at_least() {
    local run_dir="$1"
    local required_iters="$2"
    local latest_file="${run_dir}/latest_checkpointed_iteration.txt"
    [ -f "$latest_file" ] && [ "$(tr -d '\n\r[:space:]' < "$latest_file")" -ge "$required_iters" ]
}

if ! has_checkpoint_at_least "$STAGE1_WEIGHTS_DIR" "$STAGE1_REQUIRED_ITERS"; then
    echo "[ERROR] missing completed G2 wiki checkpoint >=${STAGE1_REQUIRED_ITERS}: ${STAGE1_WEIGHTS_DIR}" >&2
    exit 1
fi

if ! compgen -G "$ROUTER_MEMORY_DATASET/*.bin" >/dev/null; then
    echo "[ERROR] full Wiki train router-memory dataset not found: $ROUTER_MEMORY_DATASET" >&2
    exit 1
fi

if is_completed "$TRAIN_WEIGHTS"; then
    echo "[SKIP] already completed: $RUN_ID"
    exit 0
fi

echo "[CONFIG] G2 teacher-student router KD fullwiki"
echo "[CONFIG] code steps=${TRAIN_ITERS}, wiki KD interval=${ROUTER_MEMORY_INTERVAL}, fraction=${ROUTER_MEMORY_FRACTION}"
echo "[CONFIG] teacher/student full models stay loaded; joint update enabled"
echo "[CONFIG] teacher_student_existing_experts_only=${ROUTER_MEMORY_TEACHER_STUDENT_KL_EXISTING_EXPERTS_ONLY}"
echo "[CONFIG] wiki=${STAGE1_WEIGHTS_DIR}"
echo "[CONFIG] code=${TRAIN_WEIGHTS}"
echo "[CONFIG] router_memory_dataset=${ROUTER_MEMORY_DATASET}"

env \
    WANDB_MODE="$WANDB_MODE" \
    DIRECT_LOCAL_SAVE=1 \
    RUN_ID="$RUN_ID" \
    TRAIN_WEIGHTS="$TRAIN_WEIGHTS" \
    STAGE1_WEIGHTS_DIR="$STAGE1_WEIGHTS_DIR" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_LOG_CHECKPOINTS="$WANDB_LOG_CHECKPOINTS" \
    WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2 - teacher-student router KD fullwiki all-router kl0p1 - wiki to code}" \
    LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND="${LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND:-0}" \
    RUN_INITIAL_PROBE_EVAL="${RUN_INITIAL_PROBE_EVAL:-1}" \
    PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-100}" \
    SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-100}" \
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-72}" \
    GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
    TRAIN_ITERS="$TRAIN_ITERS" \
    SOURCE_REQUIRED_ITERS="$STAGE1_REQUIRED_ITERS" \
    SAVE_INTERVAL="$SAVE_INTERVAL" \
    SAVE_CHECKPOINTS="$SAVE_CHECKPOINTS" \
    EVAL_INTERVAL="$EVAL_INTERVAL" \
    LOG_INTERVAL="$LOG_INTERVAL" \
    SOURCE_NUM_EXPERTS=8 \
    NUM_EXPERTS=16 \
    MOE_ROUTER_TOPK=4 \
    MOE_FFN_HIDDEN_SIZE=352 \
    ENABLE_OLD_MODEL_KL=0 \
    OLD_MODEL_KL_COEFF=0.0 \
    OLD_MODEL_KL_TEMPERATURE=1.0 \
    ATTN_LORA_RANK=256 \
    ATTN_LORA_ALPHA=256 \
    ATTN_FULL_RANK_LORA_RANK=256 \
    ATTN_FULL_RANK_LORA_ALPHA=256 \
    ATTN_FULL_RANK_LORA_TARGETS=qkvo \
    ATTN_FULL_RANK_LORA_ACTIVE_TARGETS= \
    ATTN_LORA_GROUPED_GEMM="$ATTN_LORA_GROUPED_GEMM" \
    MOE_GROUPED_GEMM="$MOE_GROUPED_GEMM" \
    MOE_PERMUTE_FUSION="$MOE_PERMUTE_FUSION" \
    MOE_ROUTER_DTYPE="$MOE_ROUTER_DTYPE" \
    ROUTER_MEMORY_KL_COEFF="$ROUTER_MEMORY_KL_COEFF" \
    ROUTER_MEMORY_FORCE_ENABLE_ZERO_COEFF="$ROUTER_MEMORY_FORCE_ENABLE_ZERO_COEFF" \
    ROUTER_MEMORY_FRACTION="$ROUTER_MEMORY_FRACTION" \
    ROUTER_MEMORY_INTERVAL="$ROUTER_MEMORY_INTERVAL" \
    ROUTER_MEMORY_DATASET="$ROUTER_MEMORY_DATASET" \
    ROUTER_MEMORY_EVAL_DATASET="$ROUTER_MEMORY_EVAL_DATASET" \
    ROUTER_MEMORY_EVAL_INTERVAL="$ROUTER_MEMORY_EVAL_INTERVAL" \
    ROUTER_MEMORY_EVAL_ITERS="$ROUTER_MEMORY_EVAL_ITERS" \
    ROUTER_MEMORY_TEACHER_STUDENT_KL="$ROUTER_MEMORY_TEACHER_STUDENT_KL" \
    ROUTER_MEMORY_TEACHER_STUDENT_KL_EXISTING_EXPERTS_ONLY="$ROUTER_MEMORY_TEACHER_STUDENT_KL_EXISTING_EXPERTS_ONLY" \
    ROUTER_MEMORY_JOINT_UPDATE="$ROUTER_MEMORY_JOINT_UPDATE" \
    SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS="$SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS" \
    ROUTER_KL_EARLY_STOP_ENABLED="$ROUTER_KL_EARLY_STOP_ENABLED" \
    ROUTER_KL_STOP_STEP="$ROUTER_KL_STOP_STEP" \
    MASTER_PORT="${MASTER_PORT:-29732}" \
    USE_GUARD="$USE_GUARD" \
    SCRIPT_DIR="$SCRIPT_DIR" \
    bash -lc 'if [ "$USE_GUARD" = "1" ]; then exec "$SCRIPT_DIR/run_guarded_training.sh" bash "$SCRIPT_DIR/code_from_wiki_e6_fullrank_qkvo_expert_mha_a100_bf16.sh"; else exec bash "$SCRIPT_DIR/code_from_wiki_e6_fullrank_qkvo_expert_mha_a100_bf16.sh"; fi'

echo "[DONE] G2 teacher-student router KD fullwiki $(date)"
