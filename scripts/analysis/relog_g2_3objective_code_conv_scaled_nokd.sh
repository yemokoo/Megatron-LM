#!/bin/bash
set -euo pipefail

R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_BASE="${LOCAL_BASE:-/data3/seonghyeonnoh/LLM-continual-learning-runs/local}"
G="$LOCAL_BASE/weights/a100/mha/g2-checkpoints"
L="$LOCAL_BASE/logs/g2_9stage_old_replay_3objective_c10_l2to9_postkd_probe3i100"
PYTHON_BIN="${PYTHON_BIN:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin/python}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_ENTITY="${WANDB_ENTITY:-yemoyemo010831-korea-university}"

DRY=()
[ "${DRY_RUN:-0}" = "1" ] && DRY=(--dry-run)

upload() {
    local objective="$1" run_id="$2" run_name="$3" code_dir="$4" conv_dir="$5" code_log="$6" conv_log="$7"
    "$PYTHON_BIN" "$R/scripts/analysis/relog_g2_code_conv_scaled_nokd_to_wandb.py" \
        --objective "$objective" --run-id "$run_id" --run-name "$run_name" \
        --code-dir "$code_dir" --conv-dir "$conv_dir" \
        "${DRY[@]}"
}

upload hidden_kl \
    g2-hkl-c10-l2to9-code1800to5400-conv5400to9000-nokd \
    "G2 hidden-KL c10 L2-L9 | Code 1800-5400 + Conv 5400-9000 | no KD curve" \
    "$G/code/joint_old_data_hidden_kl/post_kd_teacher_c10_fixed_l2to9/r2-code-hiddenkl-c10-fixed-l2to9-post16-mb48-1800-probe3i100" \
    "$G/conversation/joint_old_data_hidden_kl/post_kd_teacher_c10_fixed_l2to9/r4-conv-hiddenkl-c10-fixed-l2to9-post24-mb36-1800-probe3i100-wandbstepfix" \
    "$L/hidden_kl/r2.log" "$L/hidden_kl/r4.log"

upload hidden_mse \
    g2-hmse-c10-l2to9-code1800to5400-conv5400to9000-nokd \
    "G2 hidden-MSE c10 L2-L9 | Code 1800-5400 + Conv 5400-9000 | no KD curve" \
    "$G/code/joint_old_data_hidden_mse/post_kd_teacher_c10_fixed_l2to9/r2-code-hiddenmse-c10-fixed-l2to9-post16-mb48-1800-probe3i100" \
    "$G/conversation/joint_old_data_hidden_mse/post_kd_teacher_c10_fixed_l2to9/r4-conv-hiddenmse-c10-fixed-l2to9-post24-mb36-1800-probe3i100" \
    "$L/hidden_mse/r2.log" "$L/hidden_mse/r4.log"

upload vocab_kl \
    g2-vkl-c10-code1800to5400-conv5400to9000-nokd \
    "G2 vocabulary-KL c10 | Code 1800-5400 + Conv 5400-9000 | no KD curve" \
    "$G/code/joint_old_data_kd/post_kd_teacher_c10_fixed/r2-code-vocabkl-c10-fixed-post16-mb48-1800-probe3i100" \
    "$G/conversation/joint_old_data_kd/post_kd_teacher_c10_fixed/r4-conv-vocabkl-c10-fixed-post24-mb36-1800-probe3i100" \
    "$L/vocab_kl/r2.log" "$L/vocab_kl/r4.log"
