#!/usr/bin/env bash
set -euo pipefail

# Matched D control for the fingerprint/router-geometry study.
#
# B (16 experts, post Wiki KD-init, step 600)
#   -> 1800 optimizer steps on Code only
#   -> train experts 8:16 and router rows 8:16 only
#   -> no Wiki replay, no old-model KD/hidden loss, no router-FT phase

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

FLAME_ENV="${FLAME_ENV:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100}"
[[ -x "$FLAME_ENV/bin/python" ]] || { echo "[ERROR] H100 FLAME environment missing: $FLAME_ENV" >&2; exit 1; }
export PATH="$FLAME_ENV/bin:/usr/bin:/bin"
export PYTHON_BIN="$FLAME_ENV/bin/python"
export PYTHONNOUSERSITE=1
export CUDA_HOME="${CUDA_HOME:-$FLAME_ENV}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

STUDY_ROOT="${STUDY_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/fingerprint_router_geometry_20260809}"
B_WEIGHTS="${B_WEIGHTS:-/data2/seonghyeonnoh/LLM-continual-learning-runs/g2_olddata_kd_9run_20260808/01_common_kd_init/code_e8_to_e16_wiki_kd_step600}"
D_WEIGHTS="${D_WEIGHTS:-$STUDY_ROOT/checkpoints/D_B_to_Code_only_no_olddata_mb48_gbs2304_step1800}"
FLAME_DATA_ROOT="${FLAME_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup}"

if [[ ! -f "$B_WEIGHTS/latest_checkpointed_iteration.txt" ]]; then
    echo "[ERROR] B checkpoint tracker missing: $B_WEIGHTS" >&2
    exit 1
fi
if [[ "$(<"$B_WEIGHTS/latest_checkpointed_iteration.txt")" != "600" ]]; then
    echo "[ERROR] B must be the verified step-600 KD-init checkpoint: $B_WEIGHTS" >&2
    exit 1
fi
if [[ ! -d "$FLAME_DATA_ROOT/code/train" ]]; then
    echo "[ERROR] exact Code train directory missing: $FLAME_DATA_ROOT/code/train" >&2
    exit 1
fi

export STUDY_ROOT B_WEIGHTS D_WEIGHTS FLAME_DATA_ROOT
export SOURCE_WEIGHTS_DIR="$B_WEIGHTS"
export SOURCE_REQUIRED_ITERS=600
export TRAIN_WEIGHTS="$D_WEIGHTS"
export RUN_ID="D-B-to-Code-only-no-olddata-mb48-gbs2304-step1800"
export STAGE_DIR_NAME="fingerprint-router-geometry/D"
export WANDB_EXP_NAME="D | B to Code-only | no old data/router FT | 1800"
export WANDB_RUN_ID="fingerprint-D-B-code-only-1800-20260809"
export WANDB_MODE="${WANDB_MODE:-offline}"

# Match C's Code optimizer/data geometry. Eight ranks and MB48 give the same
# global-batch accumulation structure used by the matched C run.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_PORT="${MASTER_PORT:-33901}"
export MICRO_BATCH_SIZE=48
export GLOBAL_BATCH_SIZE=2304
export TRAIN_ITERS=1800
export LR=3e-4
export MIN_LR=3e-5
export LR_DECAY_STYLE=WSD
export LR_DECAY_ITERS=1800
export LR_WSD_DECAY_ITERS=180
export LR_WARMUP_FRACTION=0.01
export SAVE_INTERVAL="${SAVE_INTERVAL:-300}"
export RECOVERY_SAVE_INTERVAL="${RECOVERY_SAVE_INTERVAL:-50}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export EVAL_INTERVAL=1800
export DIRECT_LOCAL_SAVE=1
export RESUME_CONTINUAL_FROM_TRAIN_WEIGHTS=1
export NO_SAVE_OPTIM=0

# Exact D training scope: Code LM only, newly initialized/KD-initialized half
# of the experts and their router rows. Router FT is deliberately a later run.
export LOAD_EXPANDED_SOURCE=1
export SOURCE_NUM_EXPERTS=8
export NUM_EXPERTS=16
export MOE_ROUTER_TOPK=4
export MOE_FFN_HIDDEN_SIZE=352
export FREEZE_SHARED=1
export TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY=1
export TRAIN_ATTENTION_WITH_NEW_EXPERTS=0
export FREEZE_DENSE_ATTENTION_LORA_WITH_NEW_EXPERTS=1
export ENABLE_OLD_MODEL_KL=0
export OLD_MODEL_KL_COEFF=0.0
export MOE_EXPANSION_DISTILL_MODE=none
export MOE_JOINT_REPLAY_LM=0
export MOE_JOINT_REPLAY_OLD_DATA_KD=0
export MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=0
export MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=0
export MOE_INTERLEAVE_CODE_STEPS=0
export MOE_INTERLEAVE_ROUTER_STEPS=0
export TRAIN_DATASET_SECONDARY=""
export JOINT_REPLAY_DATASET=""
export JOINT_REPLAY_SECONDARY_DATASET=""

# Evaluation-only probes do not enter the optimizer. They let us observe Code
# learning and Wiki forgetting while preserving the Code-only training claim.
export RUN_INITIAL_PROBE_EVAL=1
export RUN_INITIAL_VALID_EVAL=1
export PROBE_EVAL_INTERVAL=100
export SECONDARY_PROBE_EVAL_INTERVAL=100
export PROBE_EVAL_ITERS=25
export SECONDARY_PROBE_EVAL_ITERS=25
export TERTIARY_PROBE_DATASET="$FLAME_DATA_ROOT/conversation/test"
export TERTIARY_PROBE_NAME=conversation_probe
export TERTIARY_PROBE_EVAL_INTERVAL=100
export TERTIARY_PROBE_EVAL_ITERS=25

echo "[D CONFIG] source B: $B_WEIGHTS"
echo "[D CONFIG] output:   $D_WEIGHTS"
echo "[D CONFIG] train: Code only, 1800 steps, MB48/GBS2304, 8 GPUs"
echo "[D CONFIG] trainable: experts 8:16 + router rows 8:16 only"
echo "[D CONFIG] disabled: Wiki replay, KD/hidden loss, interleave, router FT"

exec bash "$SCRIPT_DIR/run_g2_ffn_only_code_from_distill_init_mha.sh" logits
