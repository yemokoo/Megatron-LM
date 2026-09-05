#!/bin/bash
set -euo pipefail

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/../../.." && pwd)"
source "$D/common.sh"

# Structural smoke for the backbone-frozen FFN-only FLAME MoE path:
#   natural Code LM -> router only
#   quota Code LM   -> new experts only
#   Wiki replay LM  -> router only
# followed by exactly one persistent optimizer step.
export SOURCE_TASK=wiki TARGET_TASK=code FREEZE_SHARED=1
export TRAIN_ATTENTION_WITH_NEW_EXPERTS=0 FREEZE_DENSE_ATTENTION_LORA_WITH_NEW_EXPERTS=1
export TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY=1 LOAD_EXPANDED_SOURCE=1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export LOCAL_BASE="${LOCAL_BASE:-$R/.local}"
export LOCAL_DATASET="${LOCAL_DATASET:-$LOCAL_BASE/dataset}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export FLAME_DATA_ROOT="${FLAME_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup}"
export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/flame_quota_smoke_20260810/scratch}"
export DIRECT_LOCAL_SAVE=1

export SOURCE_WEIGHTS_DIR="${SOURCE_WEIGHTS_DIR:-/data2/seonghyeonnoh/LLM-continual-learning-runs/g2_olddata_kd_9run_20260808/01_common_kd_init/code_e8_to_e16_wiki_kd_step600}"
export SOURCE_REQUIRED_ITERS="${SOURCE_REQUIRED_ITERS:-600}"
export SOURCE_NUM_EXPERTS=8 NUM_EXPERTS=16
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export NUM_QUERY_GROUPS=16
export MODEL_CONFIG_SCRIPT=scripts/experiment/a100/flame-moe-bf16-no-shared.sh

export TRAIN_ITERS="${TRAIN_ITERS:-5}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-32}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
export MOE_JOINT_REPLAY_LM=1
export MOE_JOINT_REPLAY_OLD_DATA_KD=0
export MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=0
export MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=0
export JOINT_REPLAY_DATASET="${JOINT_REPLAY_DATASET:-$(dataset_dir_for_task wiki)}"
export JOINT_REPLAY_DATA_WEIGHT_MODE=equal_dataset
export MOE_JOINT_NEW_EXPERT_QUOTA="${MOE_JOINT_NEW_EXPERT_QUOTA:-0.5}"
export MOE_SEPARATE_ROUTER_EXPERT_GRAD_CLIP=1
export MOE_ALLOW_PARTIAL_OPTIMIZER_STATE=1

export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.01}"
export MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.001}"
export MOE_EXPANSION_DISTILL_MODE=none ENABLE_OLD_MODEL_KL=0
export NO_SAVE_OPTIM=0 RECOVERY_SAVE_INTERVAL=0
export RUN_INITIAL_PROBE_EVAL=0 RUN_INITIAL_VALID_EVAL=0
export PROBE_EVAL_INTERVAL=1000 SECONDARY_PROBE_EVAL_INTERVAL=1000
export EVAL_INTERVAL=1000 LOG_INTERVAL=1 SAVE_INTERVAL="$TRAIN_ITERS"
export WANDB_MODE=offline WANDB_PROJECT="${WANDB_PROJECT:-}"

export RUN_ID="${RUN_ID:-g2-ffn-only-code-newgroup-top4-quota-q${MOE_JOINT_NEW_EXPERT_QUOTA}-persistent-opt-smoke-${TRAIN_ITERS}step}"
export STAGE_DIR_NAME=a100/mha/g2-checkpoints/code/quota_persistent_optimizer_smoke
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-/data2/seonghyeonnoh/LLM-continual-learning-runs/flame_quota_smoke_20260810/$RUN_ID}"
export RUN_LOG="${RUN_LOG:-$TRAIN_WEIGHTS/logs/run.log}"

echo "[CONFIG] FFN-only backbone frozen; source=$SOURCE_WEIGHTS_DIR"
echo "[CONFIG] natural-new=router-only quota-new=new-expert-only replay=router-only"
echo "[CONFIG] quota=$MOE_JOINT_NEW_EXPERT_QUOTA one persistent optimizer step/update"
echo "[CONFIG] separate router/expert clipping=1 save_optimizer_state=1"
echo "[CONFIG] gpus=$CUDA_VISIBLE_DEVICES mb=$MICRO_BATCH_SIZE gbs=$GLOBAL_BATCH_SIZE steps=$TRAIN_ITERS"

if [ "${PLAN_ONLY:-0}" = "1" ]; then
    exit 0
fi

exec bash "$D/run_continual_moe_a100_bf16.sh"
