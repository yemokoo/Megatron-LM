#!/bin/bash
set -euo pipefail

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FLAME_ENV="${FLAME_ENV:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100}"
[ -x "$FLAME_ENV/bin/python" ] || { echo "[ERROR] missing environment: $FLAME_ENV" >&2; exit 1; }
export PATH="$FLAME_ENV/bin:/usr/bin:/bin"
export PYTHON_BIN="${PYTHON_BIN:-$FLAME_ENV/bin/python}"
export PYTHONNOUSERSITE=1 CUDA_HOME="${CUDA_HOME:-$FLAME_ENV}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export WANDB_MODE="${WANDB_MODE:-offline}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_ENTITY="${WANDB_ENTITY:-yemoyemo010831-korea-university}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}" GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-100}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-100}"
export TERTIARY_PROBE_EVAL_INTERVAL="${TERTIARY_PROBE_EVAL_INTERVAL:-100}"
export PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-25}"
export SECONDARY_PROBE_EVAL_ITERS="${SECONDARY_PROBE_EVAL_ITERS:-25}"
export TERTIARY_PROBE_EVAL_ITERS="${TERTIARY_PROBE_EVAL_ITERS:-25}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-/data3/seonghyeonnoh/LLM-continual-learning-models/pythia-12b-tokenizer}"
export LOCAL_BASE="${LOCAL_BASE:-/data3/seonghyeonnoh/LLM-continual-learning-runs/local}"
export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/data3/seonghyeonnoh/LLM-continual-learning-runs/scratch}"

G2_ROOT="${G2_ROOT:-$LOCAL_BASE/weights/a100/mha/g2-checkpoints}"
R1_SOURCE="${R1_SOURCE:-$G2_ROOT/code/expansion_distill_init/r1-code-expand-wiki-kd-e8to16-mb36-600}"
R1_STEP="${R1_STEP:-600}"
EXPAND_STEPS="${EXPAND_STEPS:-600}"
PHASE_STEPS="${PHASE_STEPS:-1800}"
CODE_PHASE_MB="${CODE_PHASE_MB:-48}"
CONV_EXPAND_MB="${CONV_EXPAND_MB:-32}"
CONV_PHASE_MB="${CONV_PHASE_MB:-36}"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-2,3,4,5,6,7,8,9}"
CONTINUAL_COEFF="${CONTINUAL_COEFF:-10}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"
OBJECTIVE_FILTER="${OBJECTIVE_FILTER:-all}"
case "$OBJECTIVE_FILTER" in
    all|hidden_mse|hidden_kl|vocab_kl) ;;
    *) echo "[ERROR] invalid OBJECTIVE_FILTER=$OBJECTIVE_FILTER" >&2; exit 2 ;;
esac
R2_METRIC_OFFSET=1800
# W&B resumes at last_logged_step + 1.  Give every stage a disjoint boundary
# so the initial metrics of the next stage are not rejected as out of order.
R3_METRIC_OFFSET="$((R2_METRIC_OFFSET + PHASE_STEPS + 1))"
R4_METRIC_OFFSET="$((R3_METRIC_OFFSET + EXPAND_STEPS + 1))"
HKL_WANDB_RUN_ID="${HKL_WANDB_RUN_ID:-g2-3objective-hidden-kl-c10-l2to9-probe3i100}"
HMSE_WANDB_RUN_ID="${HMSE_WANDB_RUN_ID:-g2-3objective-hidden-mse-c10-l2to9-probe3i100}"
VKL_WANDB_RUN_ID="${VKL_WANDB_RUN_ID:-g2-3objective-vocab-kl-c10-probe3i100}"
LOG_ROOT="${CHAIN_LOG_DIR:-$LOCAL_BASE/logs/g2_9stage_old_replay_3objective_c10_l2to9_postkd_probe3i100}"

HKL_R2_ID="${HKL_R2_ID:-r2-code-hiddenkl-c10-fixed-l2to9-post16-mb48-1800-probe3i100}"
HKL_R3_ID="${HKL_R3_ID:-r3-conv-expand-outputkd-c1-hiddenkl-branch-e16to24-mb32-600-probe3i100-wandbstepfix}"
HKL_R4_ID="${HKL_R4_ID:-r4-conv-hiddenkl-c10-fixed-l2to9-post24-mb36-1800-probe3i100-wandbstepfix}"
HMSE_R2_ID="${HMSE_R2_ID:-r2-code-hiddenmse-c10-fixed-l2to9-post16-mb48-1800-probe3i100}"
HMSE_R3_ID="${HMSE_R3_ID:-r3-conv-expand-outputkd-c1-hiddenmse-branch-e16to24-mb32-600-probe3i100}"
HMSE_R4_ID="${HMSE_R4_ID:-r4-conv-hiddenmse-c10-fixed-l2to9-post24-mb36-1800-probe3i100}"
VKL_R2_ID="${VKL_R2_ID:-r2-code-vocabkl-c10-fixed-post16-mb48-1800-probe3i100}"
VKL_R3_ID="${VKL_R3_ID:-r3-conv-expand-outputkd-c1-vocabkl-branch-e16to24-mb32-600-probe3i100}"
VKL_R4_ID="${VKL_R4_ID:-r4-conv-vocabkl-c10-fixed-post24-mb36-1800-probe3i100}"

HKL_R2_OUT="$G2_ROOT/code/joint_old_data_hidden_kl/post_kd_teacher_c10_fixed_l2to9/$HKL_R2_ID"
HKL_R3_OUT="$G2_ROOT/conversation/expansion_distill_init_3objective_c10_l2to9/hidden_kl/$HKL_R3_ID"
HKL_R4_OUT="$G2_ROOT/conversation/joint_old_data_hidden_kl/post_kd_teacher_c10_fixed_l2to9/$HKL_R4_ID"
HMSE_R2_OUT="$G2_ROOT/code/joint_old_data_hidden_mse/post_kd_teacher_c10_fixed_l2to9/$HMSE_R2_ID"
HMSE_R3_OUT="$G2_ROOT/conversation/expansion_distill_init_3objective_c10_l2to9/hidden_mse/$HMSE_R3_ID"
HMSE_R4_OUT="$G2_ROOT/conversation/joint_old_data_hidden_mse/post_kd_teacher_c10_fixed_l2to9/$HMSE_R4_ID"
VKL_R2_OUT="$G2_ROOT/code/joint_old_data_kd/post_kd_teacher_c10_fixed/$VKL_R2_ID"
VKL_R3_OUT="$G2_ROOT/conversation/expansion_distill_init_3objective_c10_l2to9/vocab_kl/$VKL_R3_ID"
VKL_R4_OUT="$G2_ROOT/conversation/joint_old_data_kd/post_kd_teacher_c10_fixed/$VKL_R4_ID"

checkpoint_at() {
    [ -f "$1/latest_checkpointed_iteration.txt" ] &&
        [ "$(tr -d '[:space:]' < "$1/latest_checkpointed_iteration.txt")" = "$2" ]
}

run_stage() {
    local name="$1" output="$2" expected="$3" log="$4"
    shift 4
    if checkpoint_at "$output" "$expected"; then
        echo "[SKIP] $name complete: $output"
        return
    fi
    echo "[START] $name -> $output"
    "$@" 2>&1 | tee "$log"
    checkpoint_at "$output" "$expected" || {
        echo "[ERROR] $name missing final step $expected: $output" >&2
        exit 1
    }
    echo "[DONE] $name"
}

cat <<PLAN
[EXPERIMENT] 9-stage post-KD-teacher replay objective comparison
[COMMON R1] reuse=$R1_SOURCE step=$R1_STEP (8E -> 16E output-KD init)
[BRANCHES] hidden MSE | hidden softmax KL | final-vocabulary KL
[OBJECTIVE FILTER] $OBJECTIVE_FILTER
[STAGES] each branch: R2 Code continual $PHASE_STEPS -> R3 Conversation expansion KD $EXPAND_STEPS -> R4 Conversation continual $PHASE_STEPS
[CONTINUAL] coeff=$CONTINUAL_COEFF fixed, temperature=$DISTILL_TEMPERATURE, hidden_layers=$HIDDEN_LAYERS
[OPTIMIZER] expert_ramp=0 router_lr_multiplier=1.0 gradient_source_logging=1
[R3] canonical final-vocabulary output KD coeff=1 temperature=1, no LM/aux/z
[DATA] Code phase Code:Wiki=1:1; Conversation phase Conversation:Wiki:Code=1:0.5:0.5
[BATCH] global=$GLOBAL_BATCH_SIZE code_phase_mb=$CODE_PHASE_MB conv_expand_mb=$CONV_EXPAND_MB conv_phase_mb=$CONV_PHASE_MB
[PROBE] Code + Wiki + Conversation at local step 0 and every $PROBE_EVAL_INTERVAL steps; iters=$PROBE_EVAL_ITERS each
[WANDB] one continuous run per objective; R2=$R2_METRIC_OFFSET..$((R2_METRIC_OFFSET + PHASE_STEPS)) R3=$R3_METRIC_OFFSET..$((R3_METRIC_OFFSET + EXPAND_STEPS)) R4=$R4_METRIC_OFFSET..$((R4_METRIC_OFFSET + PHASE_STEPS))

 1. hidden-KL R2:  $HKL_R2_OUT
 2. hidden-KL R3:  $HKL_R3_OUT
 3. hidden-KL R4:  $HKL_R4_OUT
 4. hidden-MSE R2: $HMSE_R2_OUT
 5. hidden-MSE R3: $HMSE_R3_OUT
 6. hidden-MSE R4: $HMSE_R4_OUT
 7. vocab-KL R2:   $VKL_R2_OUT
 8. vocab-KL R3:   $VKL_R3_OUT
 9. vocab-KL R4:   $VKL_R4_OUT
PLAN

[ "${PLAN_ONLY:-0}" != "1" ] || exit 0

checkpoint_at "$R1_SOURCE" "$R1_STEP" || {
    echo "[ERROR] common R1 checkpoint must be at step $R1_STEP: $R1_SOURCE" >&2
    exit 1
}
mkdir -p "$LOG_ROOT/hidden_kl" "$LOG_ROOT/hidden_mse" "$LOG_ROOT/vocab_kl"

# Branch A: layer-output hidden softmax KL on explicit MoE layers 2..9.
if [[ "$OBJECTIVE_FILTER" == "all" || "$OBJECTIVE_FILTER" == "hidden_kl" ]]; then
run_stage hidden_kl_r2_code "$HKL_R2_OUT" "$PHASE_STEPS" "$LOG_ROOT/hidden_kl/r2.log" env \
    SOURCE_WEIGHTS_DIR="$R1_SOURCE" SOURCE_REQUIRED_ITERS="$R1_STEP" \
    OLD_MODEL_KL_WEIGHTS_DIR="$R1_SOURCE" OLD_MODEL_KL_NUM_EXPERTS=16 \
    OLD_MODEL_KL_COEFF=0 \
    OLD_HIDDEN_KL_COEFF="$CONTINUAL_COEFF" OLD_HIDDEN_KL_COEFF_START= OLD_HIDDEN_KL_COEFF_DECAY_STEPS=0 \
    OLD_HIDDEN_KL_TEMPERATURE="$DISTILL_TEMPERATURE" OLD_HIDDEN_KL_LAYERS="$HIDDEN_LAYERS" \
    OLD_MODEL_KL_TEMPERATURE="$DISTILL_TEMPERATURE" \
    TRAIN_ITERS="$PHASE_STEPS" MICRO_BATCH_SIZE="$CODE_PHASE_MB" SAVE_INTERVAL="$PHASE_STEPS" \
    WANDB_RUN_ID="$HKL_WANDB_RUN_ID" WANDB_EXP_NAME="G2 hidden-KL c10 L2-L9" WANDB_RESUME=allow \
    WANDB_STEP_OFFSET="$R2_METRIC_OFFSET" PROBE_STEP_OFFSET="$R2_METRIC_OFFSET" SECONDARY_PROBE_STEP_OFFSET="$R2_METRIC_OFFSET" TERTIARY_PROBE_STEP_OFFSET="$R2_METRIC_OFFSET" \
    MOE_NEW_EXPERT_LR_RAMP_STEPS=0 MOE_ROUTER_LR_MULTIPLIER=1.0 LOG_ROUTER_GRAD_NORM_SOURCES=1 \
    RUN_ID="$HKL_R2_ID" TRAIN_WEIGHTS="$HKL_R2_OUT" MASTER_PORT=29811 \
    bash "$D/run_g2_ffn_only_code_wiki_joint_old_data_hidden_kl_allrouter_mha.sh"

run_stage hidden_kl_r3_conv_expansion "$HKL_R3_OUT" "$EXPAND_STEPS" "$LOG_ROOT/hidden_kl/r3.log" env \
    SOURCE_WEIGHTS_DIR="$HKL_R2_OUT" SOURCE_REQUIRED_ITERS="$PHASE_STEPS" OLD_MODEL_KL_NUM_EXPERTS=16 \
    OLD_MODEL_KL_COEFF=1 OLD_MODEL_KL_TEMPERATURE=1.0 \
    MOE_JOINT_REPLAY_LM=0 MOE_JOINT_REPLAY_OLD_DATA_KD=0 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=0 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=0 \
    TRAIN_ITERS="$EXPAND_STEPS" MICRO_BATCH_SIZE="$CONV_EXPAND_MB" SAVE_INTERVAL="$EXPAND_STEPS" EVAL_INTERVAL="$EXPAND_STEPS" \
    WANDB_RUN_ID="$HKL_WANDB_RUN_ID" WANDB_EXP_NAME="G2 hidden-KL c10 L2-L9" WANDB_RESUME=allow \
    WANDB_STEP_OFFSET="$R3_METRIC_OFFSET" PROBE_STEP_OFFSET="$R3_METRIC_OFFSET" SECONDARY_PROBE_STEP_OFFSET="$R3_METRIC_OFFSET" TERTIARY_PROBE_STEP_OFFSET="$R3_METRIC_OFFSET" \
    MOE_NEW_EXPERT_LR_RAMP_STEPS=0 MOE_ROUTER_LR_MULTIPLIER=1.0 LOG_ROUTER_GRAD_NORM_SOURCES=1 \
    STAGE_DIR_NAME=a100/mha/g2-checkpoints/conversation/expansion_distill_init_3objective_c10_l2to9/hidden_kl \
    RUN_ID="$HKL_R3_ID" TRAIN_WEIGHTS="$HKL_R3_OUT" MASTER_PORT=29812 \
    bash "$D/run_g2_ffn_only_conversation_expert_distill_init_wikicode_mha.sh"

run_stage hidden_kl_r4_conv "$HKL_R4_OUT" "$PHASE_STEPS" "$LOG_ROOT/hidden_kl/r4.log" env \
    SOURCE_WEIGHTS_DIR="$HKL_R3_OUT" SOURCE_REQUIRED_ITERS="$EXPAND_STEPS" \
    OLD_MODEL_KL_WEIGHTS_DIR="$HKL_R3_OUT" OLD_MODEL_KL_NUM_EXPERTS=24 \
    OLD_MODEL_KL_COEFF=0 \
    OLD_HIDDEN_KL_COEFF="$CONTINUAL_COEFF" OLD_HIDDEN_KL_COEFF_START= OLD_HIDDEN_KL_COEFF_DECAY_STEPS=0 \
    OLD_HIDDEN_KL_TEMPERATURE="$DISTILL_TEMPERATURE" OLD_HIDDEN_KL_LAYERS="$HIDDEN_LAYERS" \
    OLD_MODEL_KL_TEMPERATURE="$DISTILL_TEMPERATURE" \
    TRAIN_ITERS="$PHASE_STEPS" MICRO_BATCH_SIZE="$CONV_PHASE_MB" SAVE_INTERVAL="$PHASE_STEPS" \
    WANDB_RUN_ID="$HKL_WANDB_RUN_ID" WANDB_EXP_NAME="G2 hidden-KL c10 L2-L9" WANDB_RESUME=allow \
    WANDB_STEP_OFFSET="$R4_METRIC_OFFSET" PROBE_STEP_OFFSET="$R4_METRIC_OFFSET" SECONDARY_PROBE_STEP_OFFSET="$R4_METRIC_OFFSET" TERTIARY_PROBE_STEP_OFFSET="$R4_METRIC_OFFSET" \
    MOE_NEW_EXPERT_LR_RAMP_STEPS=0 MOE_ROUTER_LR_MULTIPLIER=1.0 LOG_ROUTER_GRAD_NORM_SOURCES=1 \
    RUN_ID="$HKL_R4_ID" TRAIN_WEIGHTS="$HKL_R4_OUT" MASTER_PORT=29813 \
    bash "$D/run_g2_ffn_only_conversation_wikicode_joint_old_data_hidden_kl_allrouter_mha.sh"
fi

# Branch B: layer-output hidden MSE on the same explicit layers.
if [[ "$OBJECTIVE_FILTER" == "all" || "$OBJECTIVE_FILTER" == "hidden_mse" ]]; then
run_stage hidden_mse_r2_code "$HMSE_R2_OUT" "$PHASE_STEPS" "$LOG_ROOT/hidden_mse/r2.log" env \
    SOURCE_WEIGHTS_DIR="$R1_SOURCE" SOURCE_REQUIRED_ITERS="$R1_STEP" \
    OLD_MODEL_KL_WEIGHTS_DIR="$R1_SOURCE" OLD_MODEL_KL_NUM_EXPERTS=16 \
    OLD_MODEL_KL_COEFF=0 \
    OLD_HIDDEN_MSE_COEFF="$CONTINUAL_COEFF" OLD_HIDDEN_MSE_LAYERS="$HIDDEN_LAYERS" \
    OLD_MODEL_KL_TEMPERATURE="$DISTILL_TEMPERATURE" \
    TRAIN_ITERS="$PHASE_STEPS" MICRO_BATCH_SIZE="$CODE_PHASE_MB" SAVE_INTERVAL="$PHASE_STEPS" \
    WANDB_RUN_ID="$HMSE_WANDB_RUN_ID" WANDB_EXP_NAME="G2 hidden-MSE c10 L2-L9" WANDB_RESUME=allow \
    WANDB_STEP_OFFSET="$R2_METRIC_OFFSET" PROBE_STEP_OFFSET="$R2_METRIC_OFFSET" SECONDARY_PROBE_STEP_OFFSET="$R2_METRIC_OFFSET" TERTIARY_PROBE_STEP_OFFSET="$R2_METRIC_OFFSET" \
    MOE_NEW_EXPERT_LR_RAMP_STEPS=0 MOE_ROUTER_LR_MULTIPLIER=1.0 LOG_ROUTER_GRAD_NORM_SOURCES=1 \
    RUN_ID="$HMSE_R2_ID" TRAIN_WEIGHTS="$HMSE_R2_OUT" MASTER_PORT=29821 \
    bash "$D/run_g2_ffn_only_code_wiki_joint_old_data_hidden_mse_allrouter_mha.sh"

run_stage hidden_mse_r3_conv_expansion "$HMSE_R3_OUT" "$EXPAND_STEPS" "$LOG_ROOT/hidden_mse/r3.log" env \
    SOURCE_WEIGHTS_DIR="$HMSE_R2_OUT" SOURCE_REQUIRED_ITERS="$PHASE_STEPS" OLD_MODEL_KL_NUM_EXPERTS=16 \
    OLD_MODEL_KL_COEFF=1 OLD_MODEL_KL_TEMPERATURE=1.0 \
    MOE_JOINT_REPLAY_LM=0 MOE_JOINT_REPLAY_OLD_DATA_KD=0 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=0 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=0 \
    TRAIN_ITERS="$EXPAND_STEPS" MICRO_BATCH_SIZE="$CONV_EXPAND_MB" SAVE_INTERVAL="$EXPAND_STEPS" EVAL_INTERVAL="$EXPAND_STEPS" \
    WANDB_RUN_ID="$HMSE_WANDB_RUN_ID" WANDB_EXP_NAME="G2 hidden-MSE c10 L2-L9" WANDB_RESUME=allow \
    WANDB_STEP_OFFSET="$R3_METRIC_OFFSET" PROBE_STEP_OFFSET="$R3_METRIC_OFFSET" SECONDARY_PROBE_STEP_OFFSET="$R3_METRIC_OFFSET" TERTIARY_PROBE_STEP_OFFSET="$R3_METRIC_OFFSET" \
    MOE_NEW_EXPERT_LR_RAMP_STEPS=0 MOE_ROUTER_LR_MULTIPLIER=1.0 LOG_ROUTER_GRAD_NORM_SOURCES=1 \
    STAGE_DIR_NAME=a100/mha/g2-checkpoints/conversation/expansion_distill_init_3objective_c10_l2to9/hidden_mse \
    RUN_ID="$HMSE_R3_ID" TRAIN_WEIGHTS="$HMSE_R3_OUT" MASTER_PORT=29822 \
    bash "$D/run_g2_ffn_only_conversation_expert_distill_init_wikicode_mha.sh"

run_stage hidden_mse_r4_conv "$HMSE_R4_OUT" "$PHASE_STEPS" "$LOG_ROOT/hidden_mse/r4.log" env \
    SOURCE_WEIGHTS_DIR="$HMSE_R3_OUT" SOURCE_REQUIRED_ITERS="$EXPAND_STEPS" \
    OLD_MODEL_KL_WEIGHTS_DIR="$HMSE_R3_OUT" OLD_MODEL_KL_NUM_EXPERTS=24 \
    OLD_MODEL_KL_COEFF=0 \
    OLD_HIDDEN_MSE_COEFF="$CONTINUAL_COEFF" OLD_HIDDEN_MSE_LAYERS="$HIDDEN_LAYERS" \
    OLD_MODEL_KL_TEMPERATURE="$DISTILL_TEMPERATURE" \
    TRAIN_ITERS="$PHASE_STEPS" MICRO_BATCH_SIZE="$CONV_PHASE_MB" SAVE_INTERVAL="$PHASE_STEPS" \
    WANDB_RUN_ID="$HMSE_WANDB_RUN_ID" WANDB_EXP_NAME="G2 hidden-MSE c10 L2-L9" WANDB_RESUME=allow \
    WANDB_STEP_OFFSET="$R4_METRIC_OFFSET" PROBE_STEP_OFFSET="$R4_METRIC_OFFSET" SECONDARY_PROBE_STEP_OFFSET="$R4_METRIC_OFFSET" TERTIARY_PROBE_STEP_OFFSET="$R4_METRIC_OFFSET" \
    MOE_NEW_EXPERT_LR_RAMP_STEPS=0 MOE_ROUTER_LR_MULTIPLIER=1.0 LOG_ROUTER_GRAD_NORM_SOURCES=1 \
    RUN_ID="$HMSE_R4_ID" TRAIN_WEIGHTS="$HMSE_R4_OUT" MASTER_PORT=29823 \
    bash "$D/run_g2_ffn_only_conversation_wikicode_joint_old_data_hidden_mse_allrouter_mha.sh"
fi

# Branch C: final-vocabulary teacher/student KL.
if [[ "$OBJECTIVE_FILTER" == "all" || "$OBJECTIVE_FILTER" == "vocab_kl" ]]; then
run_stage vocab_kl_r2_code "$VKL_R2_OUT" "$PHASE_STEPS" "$LOG_ROOT/vocab_kl/r2.log" env \
    SOURCE_WEIGHTS_DIR="$R1_SOURCE" SOURCE_REQUIRED_ITERS="$R1_STEP" \
    OLD_MODEL_KL_WEIGHTS_DIR="$R1_SOURCE" OLD_MODEL_KL_NUM_EXPERTS=16 \
    OLD_MODEL_KL_COEFF="$CONTINUAL_COEFF" OLD_MODEL_KL_TEMPERATURE="$DISTILL_TEMPERATURE" \
    TRAIN_ITERS="$PHASE_STEPS" MICRO_BATCH_SIZE="$CODE_PHASE_MB" SAVE_INTERVAL="$PHASE_STEPS" \
    WANDB_RUN_ID="$VKL_WANDB_RUN_ID" WANDB_EXP_NAME="G2 vocabulary-KL c10" WANDB_RESUME=allow \
    WANDB_STEP_OFFSET="$R2_METRIC_OFFSET" PROBE_STEP_OFFSET="$R2_METRIC_OFFSET" SECONDARY_PROBE_STEP_OFFSET="$R2_METRIC_OFFSET" TERTIARY_PROBE_STEP_OFFSET="$R2_METRIC_OFFSET" \
    MOE_NEW_EXPERT_LR_RAMP_STEPS=0 MOE_ROUTER_LR_MULTIPLIER=1.0 LOG_ROUTER_GRAD_NORM_SOURCES=1 \
    RUN_ID="$VKL_R2_ID" TRAIN_WEIGHTS="$VKL_R2_OUT" MASTER_PORT=29831 \
    bash "$D/run_g2_ffn_only_code_wiki_joint_old_data_kd_allrouter_mha.sh"

run_stage vocab_kl_r3_conv_expansion "$VKL_R3_OUT" "$EXPAND_STEPS" "$LOG_ROOT/vocab_kl/r3.log" env \
    SOURCE_WEIGHTS_DIR="$VKL_R2_OUT" SOURCE_REQUIRED_ITERS="$PHASE_STEPS" OLD_MODEL_KL_NUM_EXPERTS=16 \
    OLD_MODEL_KL_COEFF=1 OLD_MODEL_KL_TEMPERATURE=1.0 \
    MOE_JOINT_REPLAY_LM=0 MOE_JOINT_REPLAY_OLD_DATA_KD=0 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=0 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=0 \
    TRAIN_ITERS="$EXPAND_STEPS" MICRO_BATCH_SIZE="$CONV_EXPAND_MB" SAVE_INTERVAL="$EXPAND_STEPS" EVAL_INTERVAL="$EXPAND_STEPS" \
    WANDB_RUN_ID="$VKL_WANDB_RUN_ID" WANDB_EXP_NAME="G2 vocabulary-KL c10" WANDB_RESUME=allow \
    WANDB_STEP_OFFSET="$R3_METRIC_OFFSET" PROBE_STEP_OFFSET="$R3_METRIC_OFFSET" SECONDARY_PROBE_STEP_OFFSET="$R3_METRIC_OFFSET" TERTIARY_PROBE_STEP_OFFSET="$R3_METRIC_OFFSET" \
    MOE_NEW_EXPERT_LR_RAMP_STEPS=0 MOE_ROUTER_LR_MULTIPLIER=1.0 LOG_ROUTER_GRAD_NORM_SOURCES=1 \
    STAGE_DIR_NAME=a100/mha/g2-checkpoints/conversation/expansion_distill_init_3objective_c10_l2to9/vocab_kl \
    RUN_ID="$VKL_R3_ID" TRAIN_WEIGHTS="$VKL_R3_OUT" MASTER_PORT=29832 \
    bash "$D/run_g2_ffn_only_conversation_expert_distill_init_wikicode_mha.sh"

run_stage vocab_kl_r4_conv "$VKL_R4_OUT" "$PHASE_STEPS" "$LOG_ROOT/vocab_kl/r4.log" env \
    SOURCE_WEIGHTS_DIR="$VKL_R3_OUT" SOURCE_REQUIRED_ITERS="$EXPAND_STEPS" \
    OLD_MODEL_KL_WEIGHTS_DIR="$VKL_R3_OUT" OLD_MODEL_KL_NUM_EXPERTS=24 \
    OLD_MODEL_KL_COEFF="$CONTINUAL_COEFF" OLD_MODEL_KL_TEMPERATURE="$DISTILL_TEMPERATURE" \
    TRAIN_ITERS="$PHASE_STEPS" MICRO_BATCH_SIZE="$CONV_PHASE_MB" SAVE_INTERVAL="$PHASE_STEPS" \
    WANDB_RUN_ID="$VKL_WANDB_RUN_ID" WANDB_EXP_NAME="G2 vocabulary-KL c10" WANDB_RESUME=allow \
    WANDB_STEP_OFFSET="$R4_METRIC_OFFSET" PROBE_STEP_OFFSET="$R4_METRIC_OFFSET" SECONDARY_PROBE_STEP_OFFSET="$R4_METRIC_OFFSET" TERTIARY_PROBE_STEP_OFFSET="$R4_METRIC_OFFSET" \
    MOE_NEW_EXPERT_LR_RAMP_STEPS=0 MOE_ROUTER_LR_MULTIPLIER=1.0 LOG_ROUTER_GRAD_NORM_SOURCES=1 \
    RUN_ID="$VKL_R4_ID" TRAIN_WEIGHTS="$VKL_R4_OUT" MASTER_PORT=29833 \
    bash "$D/run_g2_ffn_only_conversation_wikicode_joint_old_data_kd_allrouter_mha.sh"
fi

echo "[ALL DONE] 9-stage hidden-KL, hidden-MSE, final-vocabulary-KL comparison"
