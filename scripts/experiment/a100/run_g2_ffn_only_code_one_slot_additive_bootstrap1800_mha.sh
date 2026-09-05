#!/bin/bash
set -euo pipefail

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/../../.." && pwd)"
source "$D/common.sh"

export PYTHON_BIN="${FLAME_PYTHON_BIN:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin/python}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

export SOURCE_TASK=wiki TARGET_TASK=code FREEZE_SHARED=1
export TRAIN_ATTENTION_WITH_NEW_EXPERTS=0 FREEZE_DENSE_ATTENTION_LORA_WITH_NEW_EXPERTS=1
export TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY=1 LOAD_EXPANDED_SOURCE=1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export LOCAL_BASE="${LOCAL_BASE:-$R/.local}"
export LOCAL_DATASET="${LOCAL_DATASET:-$LOCAL_BASE/dataset}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export FLAME_DATA_ROOT="${FLAME_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup}"
export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/flame_code_bootstrap_20260810/scratch_one_slot_additive}"
export DIRECT_LOCAL_SAVE=1

export SOURCE_WEIGHTS_DIR="${SOURCE_WEIGHTS_DIR:-/data2/seonghyeonnoh/LLM-continual-learning-runs/g2_olddata_kd_9run_20260808/01_common_kd_init/code_e8_to_e16_wiki_kd_step600}"
export SOURCE_REQUIRED_ITERS="${SOURCE_REQUIRED_ITERS:-600}"
export SOURCE_NUM_EXPERTS=8 NUM_EXPERTS=16
export MOE_ROUTER_TOPK=4 MOE_FFN_HIDDEN_SIZE=352 NUM_QUERY_GROUPS=16
export MODEL_CONFIG_SCRIPT=scripts/experiment/a100/flame-moe-bf16-no-shared.sh

export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-48}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export MOE_JOINT_REPLAY_LM=1
export MOE_JOINT_REPLAY_OLD_DATA_KD=0
export MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=0
export MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=0
export JOINT_REPLAY_DATASET="${JOINT_REPLAY_DATASET:-$(dataset_dir_for_task wiki)}"
export JOINT_REPLAY_DATA_WEIGHT_MODE=equal_dataset

# Preserve the ordinary V2-style natural Code update. For the first 200 steps,
# add only 0.1x expert gradient from a shadow pass that gives at least 50% of
# tokens one new-group slot in top-4. The quota pass never changes router grads.
export MOE_JOINT_NEW_EXPERT_QUOTA=0
export MOE_JOINT_NEW_EXPERT_QUOTA_SCHEDULE="${MOE_JOINT_NEW_EXPERT_QUOTA_SCHEDULE:-200:0.5}"
export MOE_JOINT_NEW_EXPERT_QUOTA_MIN_NEW_SLOTS="${MOE_JOINT_NEW_EXPERT_QUOTA_MIN_NEW_SLOTS:-1}"
export MOE_JOINT_NEW_EXPERT_QUOTA_LOSS_COEFF="${MOE_JOINT_NEW_EXPERT_QUOTA_LOSS_COEFF:-0.1}"
export MOE_JOINT_NEW_EXPERT_QUOTA_PRESERVE_NATURAL_GRADS=1
export MOE_SEPARATE_ROUTER_EXPERT_GRAD_CLIP=1
export MOE_ALLOW_PARTIAL_OPTIMIZER_STATE=0

export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.01}"
export MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.001}"
export MOE_EXPANSION_DISTILL_MODE=none ENABLE_OLD_MODEL_KL=0
export NO_SAVE_OPTIM=1
export SAVE_INTERVAL="${SAVE_INTERVAL:-$TRAIN_ITERS}"
export RECOVERY_SAVE_INTERVAL="${RECOVERY_SAVE_INTERVAL:-300}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-$TRAIN_ITERS}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export RUN_INITIAL_PROBE_EVAL="${RUN_INITIAL_PROBE_EVAL:-1}"
export RUN_INITIAL_VALID_EVAL=0
export LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND=0
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-100}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-100}"
export PROBE_DATASET="${PROBE_DATASET:-$(probe_dir_for_task code)}" PROBE_NAME=code_probe
export SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET:-$(probe_dir_for_task wiki)}" SECONDARY_PROBE_NAME=wiki_probe
export WANDB_MODE=offline WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"

export RUN_ID="${RUN_ID:-g2-ffn-only-code1800-one-slot-additive-bootstrap-q50s200-b01-persistent-opt}"
export STAGE_DIR_NAME=a100/mha/g2-checkpoints/code/one_slot_additive_bootstrap
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-/data2/seonghyeonnoh/LLM-continual-learning-runs/flame_code_bootstrap_20260810/$RUN_ID}"
export RUN_LOG="${RUN_LOG:-$TRAIN_WEIGHTS/logs/run.log}"
export RESUME_CONTINUAL_FROM_TRAIN_WEIGHTS=1

echo "[CONFIG] source=post-Wiki KD-init $SOURCE_WEIGHTS_DIR"
echo "[CONFIG] natural Code gradients preserved for new experts + all router rows"
echo "[CONFIG] one-slot shadow bootstrap=$MOE_JOINT_NEW_EXPERT_QUOTA_SCHEDULE coeff=$MOE_JOINT_NEW_EXPERT_QUOTA_LOSS_COEFF"
echo "[CONFIG] Wiki router replay on every update; persistent in-memory optimizer"
echo "[CONFIG] gpus=$CUDA_VISIBLE_DEVICES mb=$MICRO_BATCH_SIZE gbs=$GLOBAL_BATCH_SIZE steps=$TRAIN_ITERS"

if [ "${PLAN_ONLY:-0}" = "1" ]; then
    exit 0
fi

exec bash "$D/run_continual_moe_a100_bf16.sh"
