#!/bin/bash
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/../../.." && pwd)"
source "$D/common.sh"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-$R/.local/models/pythia-12b-tokenizer}"
export SOURCE_TASK=wiki TARGET_TASK=code FREEZE_SHARED=1
export TRAIN_ATTENTION_WITH_NEW_EXPERTS=0 FREEZE_DENSE_ATTENTION_LORA_WITH_NEW_EXPERTS=1
export TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY=1
export LOAD_EXPANDED_SOURCE="${LOAD_EXPANDED_SOURCE:-1}"
export LOCAL_BASE="${LOCAL_BASE:-$R/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export G2_ROOT="${G2_ROOT:-$LOCAL_WEIGHTS/a100/mha/g2-checkpoints}"
export SOURCE_NUM_EXPERTS=8 NUM_EXPERTS=16 MOE_ROUTER_TOPK=4 MOE_FFN_HIDDEN_SIZE=352 NUM_QUERY_GROUPS=16
export SOURCE_WEIGHTS_DIR="${SOURCE_WEIGHTS_DIR:-$G2_ROOT/code/expansion_distill_init/g2-ffn-only-e8to16-code-expert-init-logits-wiki-distill-mha-a100-bf16-mb48-1800}"
export SOURCE_REQUIRED_ITERS="${SOURCE_REQUIRED_ITERS:-1800}"
export MODEL_CONFIG_SCRIPT=scripts/experiment/a100/flame-moe-bf16-no-shared.sh
export TRAIN_ITERS="${TRAIN_ITERS:-1800}" MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-64}" GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
# task-only 진단용: replay 를 끌 수 있게 개방 (기본값은 기존과 동일하게 1)
export MOE_JOINT_REPLAY_LM="${MOE_JOINT_REPLAY_LM:-1}"
export JOINT_REPLAY_DATASET="${JOINT_REPLAY_DATASET:-$(dataset_dir_for_task wiki)}"
export JOINT_REPLAY_DATA_WEIGHT_MODE=equal_dataset
export ENABLE_OLD_MODEL_KL=0 OLD_MODEL_KL_COEFF=0 MOE_EXPANSION_DISTILL_MODE=none
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.01}" MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.001}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-600}" EVAL_INTERVAL="${EVAL_INTERVAL:-$TRAIN_ITERS}" LOG_INTERVAL="${LOG_INTERVAL:-20}"
export NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-1}"
export LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND="${LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND:-0}"
export PROBE_DATASET="$(probe_dir_for_task code)" PROBE_NAME=code_probe
export SECONDARY_PROBE_DATASET="$(probe_dir_for_task wiki)" SECONDARY_PROBE_NAME=wiki_probe
export STAGE_DIR_NAME=a100/mha/g2-checkpoints/code/joint_lm_replay
export RUN_ID="${RUN_ID:-g2-ffn-only-code-wiki-joint-lm-allrouter-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$G2_ROOT/code/joint_lm_replay/$RUN_ID}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2 FFN-only Code+Wiki joint LM all-router}"
exec bash "$D/run_continual_moe_a100_bf16.sh"
