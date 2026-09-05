#!/bin/bash
set -euo pipefail

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/../../.." && pwd)"
FLAME_ENV="${FLAME_ENV:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100}"
[ -x "$FLAME_ENV/bin/python" ] || { echo "[ERROR] H100 FLAME environment missing: $FLAME_ENV" >&2; exit 1; }
export PATH="$FLAME_ENV/bin:/usr/bin:/bin"
export PYTHON_BIN="${PYTHON_BIN:-$FLAME_ENV/bin/python}"
export PYTHONNOUSERSITE=1
export CUDA_HOME="${CUDA_HOME:-$FLAME_ENV}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export LOCAL_BASE="${LOCAL_BASE:-/data3/seonghyeonnoh/LLM-continual-learning-runs/local}"
export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/data3/seonghyeonnoh/LLM-continual-learning-runs/scratch}"
export G2_ROOT="${G2_ROOT:-$LOCAL_BASE/weights/a100/mha/g2-checkpoints}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"

WIKI_SOURCE="${WIKI_SOURCE:-}"
WIKI_SOURCE_STEP="${WIKI_SOURCE_STEP:-1800}"
KD_COEFF="${OLD_MODEL_KL_COEFF:-1.0}"
KD_TEMP="${OLD_MODEL_KL_TEMPERATURE:-1.0}"
RAMP_STEPS="${MOE_NEW_EXPERT_LR_RAMP_STEPS:-900}"

CODE_KD_ID="${CODE_KD_RUN_ID:-g2-e8to16-code-init-wiki-logits-kd-mb48-600}"
CODE_KD_OUT="$G2_ROOT/code/expansion_distill_init/$CODE_KD_ID"
CODE_ID="${CODE_RUN_ID:-g2-code-lm-wiki-olddata-kd-ramp900-mb96-1800}"
CODE_OUT="$G2_ROOT/code/joint_old_data_kd/$CODE_ID"
CONV_KD_ID="${CONV_KD_RUN_ID:-g2-e16to24-conv-init-wikicode-logits-kd-mb36-600}"
CONV_KD_OUT="$G2_ROOT/conversation/expansion_distill_init_olddata_kd/$CONV_KD_ID"
CONV_ID="${CONV_RUN_ID:-g2-conv-lm-wikicode-olddata-kd-ramp900-mb96-1800}"
CONV_OUT="$G2_ROOT/conversation/joint_old_data_kd/$CONV_ID"
LOG_ROOT="${LOG_DIR:-$LOCAL_BASE/logs/g2_4stage_old_data_kd}"

checkpoint_at() {
    local root="$1" expected="$2"
    [ -f "$root/latest_checkpointed_iteration.txt" ] &&
        [ "$(tr -d '[:space:]' < "$root/latest_checkpointed_iteration.txt")" = "$expected" ]
}

run_stage() {
    local name="$1" output="$2" expected="$3" log="$4"
    shift 4
    if checkpoint_at "$output" "$expected"; then
        echo "[SKIP] $name complete: $output"
        return
    fi
    echo "[START] $name"
    "$@" 2>&1 | tee "$log"
    checkpoint_at "$output" "$expected" || {
        echo "[ERROR] $name did not produce step $expected: $output" >&2
        exit 1
    }
}

echo "[EXPERIMENT] new-task LM + old-data output-logits KD in one optimizer update"
echo "[CONTROL] same expansion, freeze mask, all-router training, GBS, ramp, and data ratio as joint-LM replay"
echo "[ONLY CHANGE] old replay branch LM -> pure teacher-logits KD"
echo "[STAGES] CodeKD 600/MB48 -> Code 1800/MB96 -> ConvKD 600/MB36 -> Conv 1800/MB96"
echo "[OUTPUT] $CODE_KD_OUT"
echo "[OUTPUT] $CODE_OUT"
echo "[OUTPUT] $CONV_KD_OUT"
echo "[OUTPUT] $CONV_OUT"

if [ "${PLAN_ONLY:-0}" = "1" ]; then
    exit 0
fi

: "${WIKI_SOURCE:?set WIKI_SOURCE to the completed 8E Wiki checkpoint}"
checkpoint_at "$WIKI_SOURCE" "$WIKI_SOURCE_STEP" || {
    echo "[ERROR] Wiki source step $WIKI_SOURCE_STEP not found: $WIKI_SOURCE" >&2
    exit 1
}
mkdir -p "$LOG_ROOT"

run_stage code_expansion_kd "$CODE_KD_OUT" 600 "$LOG_ROOT/1_code_expansion_kd.log" \
    env SOURCE_WEIGHTS_DIR="$WIKI_SOURCE" SOURCE_REQUIRED_ITERS="$WIKI_SOURCE_STEP" \
        TRAIN_ITERS=600 MICRO_BATCH_SIZE=48 SAVE_INTERVAL=600 EVAL_INTERVAL=600 \
        OLD_MODEL_KL_COEFF="$KD_COEFF" OLD_MODEL_KL_TEMPERATURE="$KD_TEMP" \
        RUN_ID="$CODE_KD_ID" TRAIN_WEIGHTS="$CODE_KD_OUT" MASTER_PORT=29991 \
        bash "$D/run_g2_ffn_only_code_expert_distill_init_mha.sh" logits

run_stage code_old_data_kd "$CODE_OUT" 1800 "$LOG_ROOT/2_code_old_data_kd.log" \
    env SOURCE_WEIGHTS_DIR="$CODE_KD_OUT" SOURCE_REQUIRED_ITERS=600 \
        OLD_MODEL_KL_WEIGHTS_DIR="$WIKI_SOURCE" TRAIN_ITERS=1800 MICRO_BATCH_SIZE=96 \
        MOE_NEW_EXPERT_LR_RAMP_STEPS="$RAMP_STEPS" SAVE_INTERVAL=1800 EVAL_INTERVAL=600 \
        OLD_MODEL_KL_COEFF="$KD_COEFF" OLD_MODEL_KL_TEMPERATURE="$KD_TEMP" \
        RUN_ID="$CODE_ID" TRAIN_WEIGHTS="$CODE_OUT" MASTER_PORT=29992 \
        bash "$D/run_g2_ffn_only_code_wiki_joint_old_data_kd_allrouter_mha.sh"

run_stage conversation_expansion_kd "$CONV_KD_OUT" 600 "$LOG_ROOT/3_conversation_expansion_kd.log" \
    env SOURCE_WEIGHTS_DIR="$CODE_OUT" SOURCE_REQUIRED_ITERS=1800 \
        TRAIN_ITERS=600 MICRO_BATCH_SIZE=36 SAVE_INTERVAL=600 EVAL_INTERVAL=600 \
        OLD_MODEL_KL_COEFF="$KD_COEFF" OLD_MODEL_KL_TEMPERATURE="$KD_TEMP" \
        STAGE_DIR_NAME=a100/mha/g2-checkpoints/conversation/expansion_distill_init_olddata_kd \
        RUN_ID="$CONV_KD_ID" TRAIN_WEIGHTS="$CONV_KD_OUT" MASTER_PORT=29993 \
        bash "$D/run_g2_ffn_only_conversation_expert_distill_init_wikicode_mha.sh"

run_stage conversation_old_data_kd "$CONV_OUT" 1800 "$LOG_ROOT/4_conversation_old_data_kd.log" \
    env SOURCE_WEIGHTS_DIR="$CONV_KD_OUT" SOURCE_REQUIRED_ITERS=600 \
        OLD_MODEL_KL_WEIGHTS_DIR="$CODE_OUT" TRAIN_ITERS=1800 MICRO_BATCH_SIZE=96 \
        MOE_NEW_EXPERT_LR_RAMP_STEPS="$RAMP_STEPS" SAVE_INTERVAL=1800 EVAL_INTERVAL=600 \
        OLD_MODEL_KL_COEFF="$KD_COEFF" OLD_MODEL_KL_TEMPERATURE="$KD_TEMP" \
        RUN_ID="$CONV_ID" TRAIN_WEIGHTS="$CONV_OUT" MASTER_PORT=29994 \
        bash "$D/run_g2_ffn_only_conversation_wikicode_joint_old_data_kd_allrouter_mha.sh"

echo "[DONE] $CONV_OUT"
