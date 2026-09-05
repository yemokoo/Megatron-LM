#!/bin/bash
set -euo pipefail

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/../../.." && pwd)"
source "$D/common.sh"

export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-$R/.local/models/pythia-12b-tokenizer}"
export SOURCE_TASK=code TARGET_TASK=conversation FREEZE_SHARED=1
export TRAIN_ATTENTION_WITH_NEW_EXPERTS=0 FREEZE_DENSE_ATTENTION_LORA_WITH_NEW_EXPERTS=1
export TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY=1 LOAD_EXPANDED_SOURCE=1
export LOCAL_BASE="${LOCAL_BASE:-$R/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export G2_ROOT="${G2_ROOT:-$LOCAL_WEIGHTS/a100/mha/g2-checkpoints}"
export SOURCE_NUM_EXPERTS=16 NUM_EXPERTS=24 MOE_ROUTER_TOPK=4 MOE_FFN_HIDDEN_SIZE=352 NUM_QUERY_GROUPS=16
: "${SOURCE_WEIGHTS_DIR:?set SOURCE_WEIGHTS_DIR to the 24E Conversation expansion-KD checkpoint}"
: "${OLD_MODEL_KL_WEIGHTS_DIR:?set OLD_MODEL_KL_WEIGHTS_DIR to the frozen hidden-MSE teacher}"
export SOURCE_REQUIRED_ITERS="${SOURCE_REQUIRED_ITERS:-600}"
export MODEL_CONFIG_SCRIPT=scripts/experiment/a100/flame-moe-bf16-no-shared.sh
export TRAIN_ITERS="${TRAIN_ITERS:-1800}" MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-36}" GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"

# Primary is Conversation LM. Replay is an equal Wiki/Code blend with pure
# layer-output MSE; only replay router gradients survive.
export MOE_JOINT_REPLAY_LM=1 MOE_JOINT_REPLAY_OLD_DATA_KD=0
export MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=0
export MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=1
export JOINT_REPLAY_DATASET="${JOINT_REPLAY_DATASET-$(dataset_dir_for_task wiki)}"
export JOINT_REPLAY_SECONDARY_DATASET="${JOINT_REPLAY_SECONDARY_DATASET-$(dataset_dir_for_task code)}"
export JOINT_REPLAY_DATA_WEIGHT_MODE=equal_dataset
export ENABLE_OLD_MODEL_KL=1 OLD_MODEL_KL_COEFF="${OLD_MODEL_KL_COEFF:-0.0}"
export OLD_HIDDEN_MSE_COEFF="${OLD_HIDDEN_MSE_COEFF:-1.0}"
export OLD_HIDDEN_MSE_LAYERS="${OLD_HIDDEN_MSE_LAYERS:-2,3,4,5,6,7,8,9}"
export MOE_EXPANSION_DISTILL_MODE=none
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.01}" MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.001}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-$TRAIN_ITERS}" EVAL_INTERVAL="${EVAL_INTERVAL:-600}" LOG_INTERVAL="${LOG_INTERVAL:-20}"
export NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-1}" LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND=0
export PROBE_DATASET="${PROBE_DATASET:-$(probe_dir_for_task conversation)}" PROBE_NAME=conversation_probe
export SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET:-$(probe_dir_for_task wiki)}" SECONDARY_PROBE_NAME=wiki_probe
export TERTIARY_PROBE_DATASET="${TERTIARY_PROBE_DATASET:-$(probe_dir_for_task code)}" TERTIARY_PROBE_NAME=code_probe
export TERTIARY_PROBE_EVAL_INTERVAL="${TERTIARY_PROBE_EVAL_INTERVAL:-100}"
export STAGE_DIR_NAME=a100/mha/g2-checkpoints/conversation/joint_old_data_hidden_mse
export RUN_ID="${RUN_ID:-g2-conv-lm-wikicode-hidden-mse-allrouter-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$G2_ROOT/conversation/joint_old_data_hidden_mse/$RUN_ID}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2 Conversation LM + WikiCode layer-hidden MSE all-router}"

echo "[CONFIG] primary=Conversation LM; replay=Wiki+Code 1:1 layer-output hidden MSE"
echo "[CONFIG] hidden_mse_coeff=$OLD_HIDDEN_MSE_COEFF layers=$OLD_HIDDEN_MSE_LAYERS"
echo "[CONFIG] trainable=new experts + all 24 router rows; replay gradients=all router rows only"
echo "[CONFIG] student=$SOURCE_WEIGHTS_DIR teacher=$OLD_MODEL_KL_WEIGHTS_DIR teacher_experts=${OLD_MODEL_KL_NUM_EXPERTS:-16}"

if [ "${PLAN_ONLY:-0}" = "1" ]; then
    exit 0
fi

exec bash "$D/run_continual_moe_a100_bf16.sh"
