#!/bin/bash
# 1-phase conversation stage (16->24 KD-init source, wiki+code LM replay, all-router) with
# MoE-LPR supervised old-expert group loss applied INSIDE the old-data replay pass.
# Per-task ranges wiki->0:8, code->8:16 (group 0:16 if TASK_RANGES is emptied). Otherwise identical to
# run_g2_ffn_only_conversation_wikicode_joint_lm_allrouter_mha.sh.
export TRAIN_ENTRY=pretrain_gpt_lprjoint.py
export JOINT_REPLAY_LPR_COEFF="${JOINT_REPLAY_LPR_COEFF:-0.1}"
export JOINT_REPLAY_LPR_OLD_EXPERTS="${JOINT_REPLAY_LPR_OLD_EXPERTS:-16}"
# Per-task forcing (MoE-LPR style): replay data path order is wiki (1 prefix) then code (1 prefix),
# so dataset_id 0 -> experts 0:8 and dataset_id 1 -> experts 8:16.
export JOINT_REPLAY_LPR_TASK_RANGES="${JOINT_REPLAY_LPR_TASK_RANGES:-0:8,8:16}"
export JOINT_REPLAY_LPR_PREFIX_COUNTS="${JOINT_REPLAY_LPR_PREFIX_COUNTS:-1,1}"
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/../../.." && pwd)"
source "$D/common.sh"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-$R/.local/models/pythia-12b-tokenizer}"
export SOURCE_TASK=code TARGET_TASK=conversation FREEZE_SHARED=1
export TRAIN_ATTENTION_WITH_NEW_EXPERTS=0 FREEZE_DENSE_ATTENTION_LORA_WITH_NEW_EXPERTS=1
export TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY=1
export LOAD_EXPANDED_SOURCE="${LOAD_EXPANDED_SOURCE:-1}"
export LOCAL_BASE="${LOCAL_BASE:-$R/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export G2_ROOT="${G2_ROOT:-$LOCAL_WEIGHTS/a100/mha/g2-checkpoints}"
export SOURCE_NUM_EXPERTS=16 NUM_EXPERTS=24 MOE_ROUTER_TOPK=4 MOE_FFN_HIDDEN_SIZE=352 NUM_QUERY_GROUPS=16
: "${SOURCE_WEIGHTS_DIR:?set SOURCE_WEIGHTS_DIR to the 16-to-24 KD checkpoint}"
export SOURCE_REQUIRED_ITERS="${SOURCE_REQUIRED_ITERS:-1800}"
export MODEL_CONFIG_SCRIPT=scripts/experiment/a100/flame-moe-bf16-no-shared.sh
export TRAIN_ITERS="${TRAIN_ITERS:-1800}" MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-64}" GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export MOE_JOINT_REPLAY_LM=1
export JOINT_REPLAY_DATASET="${JOINT_REPLAY_DATASET-$(dataset_dir_for_task wiki)}"
export JOINT_REPLAY_SECONDARY_DATASET="${JOINT_REPLAY_SECONDARY_DATASET-$(dataset_dir_for_task code)}"
export JOINT_REPLAY_DATA_WEIGHT_MODE=equal_dataset
export ENABLE_OLD_MODEL_KL=0 OLD_MODEL_KL_COEFF=0 MOE_EXPANSION_DISTILL_MODE=none
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.01}" MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.001}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-600}" EVAL_INTERVAL="${EVAL_INTERVAL:-$TRAIN_ITERS}" LOG_INTERVAL="${LOG_INTERVAL:-20}"
export NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-1}"
export LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND="${LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND:-0}"
export STAGE_DIR_NAME=a100/mha/g2-checkpoints/conversation/joint_lm_replay_lpr
export RUN_ID="${RUN_ID:-g2-ffn-only-conv-wikicode-joint-lm-lpr${JOINT_REPLAY_LPR_COEFF}-allrouter-112-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$G2_ROOT/conversation/joint_lm_replay_lpr/$RUN_ID}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-G2 FFN-only Conversation+WikiCode joint LM 1:1:2}"
exec bash "$D/run_continual_moe_a100_bf16_lprjoint.sh"
