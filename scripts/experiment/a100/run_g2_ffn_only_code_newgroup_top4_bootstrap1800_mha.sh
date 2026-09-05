#!/bin/bash
set -euo pipefail

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/../../.." && pwd)"
source "$D/common.sh"

export SOURCE_TASK=wiki TARGET_TASK=code FREEZE_SHARED=1
export TRAIN_ATTENTION_WITH_NEW_EXPERTS=0 FREEZE_DENSE_ATTENTION_LORA_WITH_NEW_EXPERTS=1
export TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY=1 LOAD_EXPANDED_SOURCE=1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export LOCAL_BASE="${LOCAL_BASE:-$R/.local}"
export LOCAL_DATASET="${LOCAL_DATASET:-$LOCAL_BASE/dataset}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export FLAME_DATA_ROOT="${FLAME_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup}"
export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/flame_code_bootstrap_20260810/scratch}"
export DIRECT_LOCAL_SAVE=1

export SOURCE_WEIGHTS_DIR="${SOURCE_WEIGHTS_DIR:-/data2/seonghyeonnoh/LLM-continual-learning-runs/g2_olddata_kd_9run_20260808/01_common_kd_init/code_e8_to_e16_wiki_kd_step600}"
export SOURCE_REQUIRED_ITERS="${SOURCE_REQUIRED_ITERS:-600}"
export SOURCE_NUM_EXPERTS=8 NUM_EXPERTS=16
export MOE_ROUTER_TOPK=4 MOE_FFN_HIDDEN_SIZE=352 NUM_QUERY_GROUPS=16
export MODEL_CONFIG_SCRIPT=scripts/experiment/a100/flame-moe-bf16-no-shared.sh

export TRAIN_ITERS=1800 MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-48}" GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export MOE_JOINT_REPLAY_LM=1
export MOE_JOINT_REPLAY_OLD_DATA_KD=0
export MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=0
export MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=0
export JOINT_REPLAY_DATASET="${JOINT_REPLAY_DATASET:-$(dataset_dir_for_task wiki)}"
export JOINT_REPLAY_DATA_WEIGHT_MODE=equal_dataset

# Bootstrap only at the beginning.  A quota token is dispatched to the natural
# top-4 within the newly added 8-expert task group; no expert-specific quota is
# used.  At step 600 this branch disappears and ordinary joint training resumes.
export MOE_JOINT_NEW_EXPERT_QUOTA=0
export MOE_JOINT_NEW_EXPERT_QUOTA_SCHEDULE="${MOE_JOINT_NEW_EXPERT_QUOTA_SCHEDULE:-300:0.5,600:0.3}"
export MOE_SEPARATE_ROUTER_EXPERT_GRAD_CLIP=1
export MOE_ALLOW_PARTIAL_OPTIMIZER_STATE=0

export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.01}"
export MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.001}"
export MOE_EXPANSION_DISTILL_MODE=none ENABLE_OLD_MODEL_KL=0
export NO_SAVE_OPTIM=1 SAVE_INTERVAL=1800 RECOVERY_SAVE_INTERVAL=300
export EVAL_INTERVAL=1800 LOG_INTERVAL=20
export RUN_INITIAL_PROBE_EVAL=1 RUN_INITIAL_VALID_EVAL=0
export LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND=0
export PROBE_EVAL_INTERVAL=100 SECONDARY_PROBE_EVAL_INTERVAL=100
export PROBE_DATASET="${PROBE_DATASET:-$(probe_dir_for_task code)}" PROBE_NAME=code_probe
export SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET:-$(probe_dir_for_task wiki)}" SECONDARY_PROBE_NAME=wiki_probe
export WANDB_MODE=offline WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"

export RUN_ID="${RUN_ID:-g2-ffn-only-code1800-newgroup-top4-bootstrap-q50s300-q30s600-persistent-opt}"
export STAGE_DIR_NAME=a100/mha/g2-checkpoints/code/newgroup_top4_bootstrap
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-/data2/seonghyeonnoh/LLM-continual-learning-runs/flame_code_bootstrap_20260810/$RUN_ID}"
export RUN_LOG="${RUN_LOG:-$TRAIN_WEIGHTS/logs/run.log}"
export RESUME_CONTINUAL_FROM_TRAIN_WEIGHTS=1

echo "[CONFIG] source=post-Wiki KD-init $SOURCE_WEIGHTS_DIR"
echo "[CONFIG] Code updates=$TRAIN_ITERS; router natural-Code + Wiki replay on every update"
echo "[CONFIG] new task group=experts 8..15; bootstrap dispatch=group-internal top-4"
echo "[CONFIG] bootstrap schedule=$MOE_JOINT_NEW_EXPERT_QUOTA_SCHEDULE"
echo "[CONFIG] one in-memory persistent optimizer + independent router/expert clipping"
echo "[CONFIG] model-only recovery checkpoints (torch_dist partial optimizer state unsupported)"
echo "[CONFIG] gpus=$CUDA_VISIBLE_DEVICES mb=$MICRO_BATCH_SIZE gbs=$GLOBAL_BATCH_SIZE"

if [ "${PLAN_ONLY:-0}" = "1" ]; then
    exit 0
fi

exec bash "$D/run_continual_moe_a100_bf16.sh"
