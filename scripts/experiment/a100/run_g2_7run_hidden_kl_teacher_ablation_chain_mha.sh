#!/bin/bash
set -euo pipefail

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/../../.." && pwd)"
FLAME_ENV="${FLAME_ENV:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100}"
[ -x "$FLAME_ENV/bin/python" ] || { echo "[ERROR] H100 FLAME environment missing: $FLAME_ENV" >&2; exit 1; }
export PATH="$FLAME_ENV/bin:/usr/bin:/bin"
export PYTHON_BIN="${PYTHON_BIN:-$FLAME_ENV/bin/python}"
export PYTHONNOUSERSITE=1 CUDA_HOME="${CUDA_HOME:-$FLAME_ENV}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export WANDB_MODE="${WANDB_MODE:-offline}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}" GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-/data3/seonghyeonnoh/LLM-continual-learning-models/pythia-12b-tokenizer}"
export LOCAL_BASE="${LOCAL_BASE:-/data3/seonghyeonnoh/LLM-continual-learning-runs/local}"
export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/data3/seonghyeonnoh/LLM-continual-learning-runs/scratch}"
export G2_ROOT="${G2_ROOT:-$LOCAL_BASE/weights/a100/mha/g2-checkpoints}"

WIKI_SOURCE="${WIKI_SOURCE:-/data3/seonghyeonnoh/LLM-continual-learning-runs/wiki/g2-wiki-e8-ffn352-top4-h100-mb72-1800}"
WIKI_SOURCE_STEP="${WIKI_SOURCE_STEP:-1800}"
EXPAND_STEPS="${EXPAND_STEPS:-600}"
PHASE_STEPS="${PHASE_STEPS:-1800}"
CODE_EXPAND_MB="${CODE_EXPAND_MB:-36}"
CODE_PHASE_MB="${CODE_PHASE_MB:-36}"
CONV_EXPAND_MB="${CONV_EXPAND_MB:-24}"
CONV_PHASE_MB="${CONV_PHASE_MB:-24}"
HIDDEN_KL_COEFF="${OLD_HIDDEN_KL_COEFF:-1.0}"
HIDDEN_KL_TEMP="${OLD_HIDDEN_KL_TEMPERATURE:-1.0}"
HIDDEN_KL_LAYERS="${OLD_HIDDEN_KL_LAYERS:-all_but_last}"
RAMP_STEPS="${MOE_NEW_EXPERT_LR_RAMP_STEPS:-900}"
LOG_ROOT="${CHAIN_LOG_DIR:-$LOCAL_BASE/logs/g2_7run_hidden_kl_teacher_ablation}"

R1_ID="${R1_RUN_ID:-r1-code-expand-wiki-kd-e8to16-mb${CODE_EXPAND_MB}-${EXPAND_STEPS}}"
R1_OUT="$G2_ROOT/code/expansion_distill_init/$R1_ID"
R2_ID="${R2_RUN_ID:-r2-code-hiddenkl-teacher-pre8-mb${CODE_PHASE_MB}-${PHASE_STEPS}}"
R2_OUT="$G2_ROOT/code/joint_old_data_hidden_kl/pre_expansion_teacher/$R2_ID"
R3_ID="${R3_RUN_ID:-r3-conv-expand-wikicode-kd-prebranch-e16to24-mb${CONV_EXPAND_MB}-${EXPAND_STEPS}}"
R3_OUT="$G2_ROOT/conversation/expansion_distill_init_hidden_kl_prebranch/$R3_ID"
R4_ID="${R4_RUN_ID:-r4-conv-hiddenkl-teacher-pre16-mb${CONV_PHASE_MB}-${PHASE_STEPS}}"
R4_OUT="$G2_ROOT/conversation/joint_old_data_hidden_kl/pre_expansion_teacher/$R4_ID"
R5_ID="${R5_RUN_ID:-r5-code-hiddenkl-teacher-post16-mb${CODE_PHASE_MB}-${PHASE_STEPS}}"
R5_OUT="$G2_ROOT/code/joint_old_data_hidden_kl/post_kd_teacher/$R5_ID"
R6_ID="${R6_RUN_ID:-r6-conv-expand-wikicode-kd-postbranch-e16to24-mb${CONV_EXPAND_MB}-${EXPAND_STEPS}}"
R6_OUT="$G2_ROOT/conversation/expansion_distill_init_hidden_kl_postbranch/$R6_ID"
R7_ID="${R7_RUN_ID:-r7-conv-hiddenkl-teacher-post24-mb${CONV_PHASE_MB}-${PHASE_STEPS}}"
R7_OUT="$G2_ROOT/conversation/joint_old_data_hidden_kl/post_kd_teacher/$R7_ID"

checkpoint_at() {
    [ -f "$1/latest_checkpointed_iteration.txt" ] &&
        [ "$(tr -d '[:space:]' < "$1/latest_checkpointed_iteration.txt")" = "$2" ]
}
run_stage() {
    local name="$1" output="$2" expected="$3" log="$4"; shift 4
    if checkpoint_at "$output" "$expected"; then echo "[SKIP] $name complete: $output"; return; fi
    echo "[START] $name"
    "$@" 2>&1 | tee "$log"
    checkpoint_at "$output" "$expected" || { echo "[ERROR] $name missing final step $expected: $output" >&2; exit 1; }
}

cat <<PLAN
[EXPERIMENT] 7-run layer-hidden-KL teacher ablation
[COMMON] R1 Wiki 8E -> Code-init 16E expansion KD on Wiki
[PRE]    R2 Code phase teacher=Wiki 8E -> R3 16E->24E KD on Wiki+Code -> R4 Conv phase teacher=R2 16E
[POST]   R5 Code phase teacher=R1 post-KD 16E -> R6 16E->24E KD on Wiki+Code -> R7 Conv phase teacher=R6 post-KD 24E
[LOSS]   primary=new-task LM; replay=old-data per-layer hidden KL; replay gradients=all router rows only
[DATA]   Code phase Code:Wiki=1:1; Conv phase Conversation:Wiki:Code=1:0.5:0.5
[SAVE]   final checkpoint only for every run
[R1] $R1_OUT
[R2] $R2_OUT
[R3] $R3_OUT
[R4] $R4_OUT
[R5] $R5_OUT
[R6] $R6_OUT
[R7] $R7_OUT
PLAN
[ "${PLAN_ONLY:-0}" != "1" ] || exit 0

checkpoint_at "$WIKI_SOURCE" "$WIKI_SOURCE_STEP" || { echo "[ERROR] Wiki final checkpoint missing: $WIKI_SOURCE" >&2; exit 1; }
mkdir -p "$LOG_ROOT"

run_stage r1_code_expansion "$R1_OUT" "$EXPAND_STEPS" "$LOG_ROOT/r1.log" env \
    SOURCE_WEIGHTS_DIR="$WIKI_SOURCE" SOURCE_REQUIRED_ITERS="$WIKI_SOURCE_STEP" \
    TRAIN_ITERS="$EXPAND_STEPS" MICRO_BATCH_SIZE="$CODE_EXPAND_MB" SAVE_INTERVAL="$EXPAND_STEPS" EVAL_INTERVAL="$EXPAND_STEPS" \
    RUN_ID="$R1_ID" TRAIN_WEIGHTS="$R1_OUT" MASTER_PORT=29871 \
    bash "$D/run_g2_ffn_only_code_expert_distill_init_mha.sh" logits

run_stage r2_code_pre_teacher "$R2_OUT" "$PHASE_STEPS" "$LOG_ROOT/r2.log" env \
    SOURCE_WEIGHTS_DIR="$R1_OUT" SOURCE_REQUIRED_ITERS="$EXPAND_STEPS" \
    OLD_MODEL_KL_WEIGHTS_DIR="$WIKI_SOURCE" OLD_MODEL_KL_NUM_EXPERTS=8 \
    OLD_HIDDEN_KL_COEFF="$HIDDEN_KL_COEFF" OLD_HIDDEN_KL_TEMPERATURE="$HIDDEN_KL_TEMP" OLD_HIDDEN_KL_LAYERS="$HIDDEN_KL_LAYERS" \
    TRAIN_ITERS="$PHASE_STEPS" MICRO_BATCH_SIZE="$CODE_PHASE_MB" SAVE_INTERVAL="$PHASE_STEPS" \
    MOE_NEW_EXPERT_LR_RAMP_STEPS="$RAMP_STEPS" RUN_ID="$R2_ID" TRAIN_WEIGHTS="$R2_OUT" MASTER_PORT=29872 \
    bash "$D/run_g2_ffn_only_code_wiki_joint_old_data_hidden_kl_allrouter_mha.sh"

run_stage r3_conv_expansion_pre_branch "$R3_OUT" "$EXPAND_STEPS" "$LOG_ROOT/r3.log" env \
    SOURCE_WEIGHTS_DIR="$R2_OUT" SOURCE_REQUIRED_ITERS="$PHASE_STEPS" \
    TRAIN_ITERS="$EXPAND_STEPS" MICRO_BATCH_SIZE="$CONV_EXPAND_MB" SAVE_INTERVAL="$EXPAND_STEPS" EVAL_INTERVAL="$EXPAND_STEPS" \
    STAGE_DIR_NAME=a100/mha/g2-checkpoints/conversation/expansion_distill_init_hidden_kl_prebranch \
    RUN_ID="$R3_ID" TRAIN_WEIGHTS="$R3_OUT" MASTER_PORT=29873 \
    bash "$D/run_g2_ffn_only_conversation_expert_distill_init_wikicode_mha.sh"

run_stage r4_conv_pre_teacher "$R4_OUT" "$PHASE_STEPS" "$LOG_ROOT/r4.log" env \
    SOURCE_WEIGHTS_DIR="$R3_OUT" SOURCE_REQUIRED_ITERS="$EXPAND_STEPS" \
    OLD_MODEL_KL_WEIGHTS_DIR="$R2_OUT" OLD_MODEL_KL_NUM_EXPERTS=16 \
    OLD_HIDDEN_KL_COEFF="$HIDDEN_KL_COEFF" OLD_HIDDEN_KL_TEMPERATURE="$HIDDEN_KL_TEMP" OLD_HIDDEN_KL_LAYERS="$HIDDEN_KL_LAYERS" \
    TRAIN_ITERS="$PHASE_STEPS" MICRO_BATCH_SIZE="$CONV_PHASE_MB" SAVE_INTERVAL="$PHASE_STEPS" \
    MOE_NEW_EXPERT_LR_RAMP_STEPS="$RAMP_STEPS" RUN_ID="$R4_ID" TRAIN_WEIGHTS="$R4_OUT" MASTER_PORT=29874 \
    bash "$D/run_g2_ffn_only_conversation_wikicode_joint_old_data_hidden_kl_allrouter_mha.sh"

run_stage r5_code_post_teacher "$R5_OUT" "$PHASE_STEPS" "$LOG_ROOT/r5.log" env \
    SOURCE_WEIGHTS_DIR="$R1_OUT" SOURCE_REQUIRED_ITERS="$EXPAND_STEPS" \
    OLD_MODEL_KL_WEIGHTS_DIR="$R1_OUT" OLD_MODEL_KL_NUM_EXPERTS=16 \
    OLD_HIDDEN_KL_COEFF="$HIDDEN_KL_COEFF" OLD_HIDDEN_KL_TEMPERATURE="$HIDDEN_KL_TEMP" OLD_HIDDEN_KL_LAYERS="$HIDDEN_KL_LAYERS" \
    TRAIN_ITERS="$PHASE_STEPS" MICRO_BATCH_SIZE="$CODE_PHASE_MB" SAVE_INTERVAL="$PHASE_STEPS" \
    MOE_NEW_EXPERT_LR_RAMP_STEPS="$RAMP_STEPS" RUN_ID="$R5_ID" TRAIN_WEIGHTS="$R5_OUT" MASTER_PORT=29875 \
    bash "$D/run_g2_ffn_only_code_wiki_joint_old_data_hidden_kl_allrouter_mha.sh"

run_stage r6_conv_expansion_post_branch "$R6_OUT" "$EXPAND_STEPS" "$LOG_ROOT/r6.log" env \
    SOURCE_WEIGHTS_DIR="$R5_OUT" SOURCE_REQUIRED_ITERS="$PHASE_STEPS" \
    TRAIN_ITERS="$EXPAND_STEPS" MICRO_BATCH_SIZE="$CONV_EXPAND_MB" SAVE_INTERVAL="$EXPAND_STEPS" EVAL_INTERVAL="$EXPAND_STEPS" \
    STAGE_DIR_NAME=a100/mha/g2-checkpoints/conversation/expansion_distill_init_hidden_kl_postbranch \
    RUN_ID="$R6_ID" TRAIN_WEIGHTS="$R6_OUT" MASTER_PORT=29876 \
    bash "$D/run_g2_ffn_only_conversation_expert_distill_init_wikicode_mha.sh"

run_stage r7_conv_post_teacher "$R7_OUT" "$PHASE_STEPS" "$LOG_ROOT/r7.log" env \
    SOURCE_WEIGHTS_DIR="$R6_OUT" SOURCE_REQUIRED_ITERS="$EXPAND_STEPS" \
    OLD_MODEL_KL_WEIGHTS_DIR="$R6_OUT" OLD_MODEL_KL_NUM_EXPERTS=24 \
    OLD_HIDDEN_KL_COEFF="$HIDDEN_KL_COEFF" OLD_HIDDEN_KL_TEMPERATURE="$HIDDEN_KL_TEMP" OLD_HIDDEN_KL_LAYERS="$HIDDEN_KL_LAYERS" \
    TRAIN_ITERS="$PHASE_STEPS" MICRO_BATCH_SIZE="$CONV_PHASE_MB" SAVE_INTERVAL="$PHASE_STEPS" \
    MOE_NEW_EXPERT_LR_RAMP_STEPS="$RAMP_STEPS" RUN_ID="$R7_ID" TRAIN_WEIGHTS="$R7_OUT" MASTER_PORT=29877 \
    bash "$D/run_g2_ffn_only_conversation_wikicode_joint_old_data_hidden_kl_allrouter_mha.sh"

echo "[DONE] pre-expansion-teacher final: $R4_OUT"
echo "[DONE] post-KD-teacher final: $R7_OUT"
