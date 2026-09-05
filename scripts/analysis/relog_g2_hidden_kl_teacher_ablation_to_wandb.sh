#!/bin/bash
set -euo pipefail

MODE="${1:-dry-run}"
case "$MODE" in
    dry-run|upload) ;;
    *) echo "usage: $0 [dry-run|upload]" >&2; exit 2 ;;
esac

R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/seonghyeonnoh/yemokoo/.bootstrap-auth/bin/python}"
LOCAL_BASE="${LOCAL_BASE:-/data3/seonghyeonnoh/LLM-continual-learning-runs/local}"
G="$LOCAL_BASE/weights/a100/mha/g2-checkpoints"

WIKI="${WIKI_SOURCE:-/data3/seonghyeonnoh/LLM-continual-learning-runs/wiki/g2-wiki-e8-ffn352-top4-h100-mb72-1800}"
TEACHER_LOG="$WIKI/logs/wiki_a.log"
R1="$G/code/expansion_distill_init/r1-code-expand-wiki-kd-e8to16-mb36-600"
R1_LOG="$R1/logs/g2_ffn_only_code_expert_logits_init_freeze.log"

POST_R2="$G/code/joint_old_data_hidden_kl/post_kd_teacher_c10_normfix/r2-code-hiddenkl-c10-normfix-teacher-post16-mb48-1800"
POST_R3="$G/conversation/expansion_distill_init_hidden_kl_postbranch_c10_normfix/r3-conv-expand-c10-normfix-postbranch-e16to24-mb32-600"
POST_R4="$G/conversation/joint_old_data_hidden_kl/post_kd_teacher_c10_normfix/r4-conv-hiddenkl-c10-normfix-teacher-post24-mb36-1800"

PRE_R5="$G/code/joint_old_data_hidden_kl/pre_expansion_teacher_c10_normfix/r5-code-hiddenkl-c10-normfix-teacher-pre8-mb48-1800"
PRE_R6="$G/conversation/expansion_distill_init_hidden_kl_prebranch_c10_normfix/r6-conv-expand-c10-normfix-prebranch-e16to24-mb32-600"
PRE_R7="$G/conversation/joint_old_data_hidden_kl/pre_expansion_teacher_c10_normfix/r7-conv-hiddenkl-c10-normfix-teacher-pre16-mb36-1800"

PARSER="$R/scripts/analysis/relog_expansion_distill_pipeline_to_wandb.py"
STAMP="$(date +%Y%m%d-%H%M%S)"
GROUP="${WANDB_GROUP:-G2 hidden-KL c10 teacher ablation Wiki-Code-Conversation}"
COMMON_TAGS=(g2 ffn-only hidden-kl c10 all-but-last teacher-ablation h100)
DRY_ARGS=()
if [ "$MODE" = dry-run ]; then
    DRY_ARGS+=(--dry-run)
else
    export WANDB_MODE=online WANDB_DISABLE_CODE=true WANDB_CONSOLE=off WANDB_INIT_TIMEOUT=300
fi

run_branch() {
    local branch="$1" code_stage="$2" conv_kd_stage="$3" conv_stage="$4" run_name="$5"
    local code_log conv_kd_log conv_log
    code_log="$(find "$code_stage/logs" -maxdepth 1 -type f -name '*.log' | head -n 1)"
    conv_kd_log="$(find "$conv_kd_stage/logs" -maxdepth 1 -type f -name '*.log' | head -n 1)"
    conv_log="$(find "$conv_stage/logs" -maxdepth 1 -type f -name '*.log' | head -n 1)"

    "$PYTHON_BIN" -u "$PARSER" \
        --teacher-dir "$WIKI" \
        --distill-dir "$R1" \
        --code-dir "$code_stage" \
        --retune-dir "$conv_kd_stage" \
        --final-dir "$conv_stage" \
        --teacher-log "$TEACHER_LOG" \
        --distill-log "$R1_LOG" \
        --code-log "$code_log" \
        --retune-log "$conv_kd_log" \
        --final-log "$conv_log" \
        --teacher-step 1800 \
        --distill-iters 600 \
        --code-iters 1800 \
        --retune-iters 600 \
        --final-iters 1800 \
        --stage-labels code_kd code_hidden_kl conversation_kd conversation_hidden_kl \
        --run-id "g2-hiddenkl-c10-${branch}-${STAMP}" \
        --run-name "$run_name" \
        --group "$GROUP" \
        --tags "${COMMON_TAGS[@]}" "$branch" \
        "${DRY_ARGS[@]}"
}

run_branch \
    post-kd-teacher \
    "$POST_R2" "$POST_R3" "$POST_R4" \
    "G2 Hidden-KL c10 - Post-KD teacher - Wiki→Code→Conversation"

run_branch \
    pre-expansion-teacher \
    "$PRE_R5" "$PRE_R6" "$PRE_R7" \
    "G2 Hidden-KL c10 - Pre-expansion teacher - Wiki→Code→Conversation"
