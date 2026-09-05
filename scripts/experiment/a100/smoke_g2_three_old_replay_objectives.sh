#!/bin/bash
set -euo pipefail

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLAME_ENV="${FLAME_ENV:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100}"
LOCAL_BASE="${LOCAL_BASE:-/data3/seonghyeonnoh/LLM-continual-learning-runs/local}"
SOURCE="${SOURCE_WEIGHTS_DIR:-$LOCAL_BASE/weights/a100/mha/g2-checkpoints/code/expansion_distill_init/r1-code-expand-wiki-kd-e8to16-mb36-600}"
SMOKE_TAG="${SMOKE_TAG:-$(date +%Y%m%d_%H%M%S)}"
SMOKE_ROOT="${SMOKE_ROOT:-$LOCAL_BASE/smoke/g2_three_old_replay_objectives/$SMOKE_TAG}"
SMOKE_GPUS="${SMOKE_GPUS:-0,1,2}"
SHORT_STEPS="${SHORT_STEPS:-2}"
MSE_STEPS="${MSE_STEPS:-12}"
IFS=',' read -r -a GPUS <<< "$SMOKE_GPUS"

[ "${#GPUS[@]}" -eq 3 ] || { echo "[ERROR] SMOKE_GPUS must contain exactly three GPU IDs" >&2; exit 1; }
[ -x "$FLAME_ENV/bin/python" ] || { echo "[ERROR] missing environment: $FLAME_ENV" >&2; exit 1; }
[ -f "$SOURCE/latest_checkpointed_iteration.txt" ] || { echo "[ERROR] missing source tracker: $SOURCE" >&2; exit 1; }
[ "$(tr -d '[:space:]' < "$SOURCE/latest_checkpointed_iteration.txt")" = 600 ] || {
    echo "[ERROR] source must be the common R1 step-600 checkpoint: $SOURCE" >&2
    exit 1
}

cat <<PLAN
[SMOKE] three old-data replay objectives
[SOURCE/STUDENT/TEACHER] $SOURCE (16 experts, post expansion-KD)
[GPU] hidden-KL=${GPUS[0]} hidden-MSE=${GPUS[1]} vocab-KL=${GPUS[2]}
[CONTRACT] primary Code LM=new experts + all router rows
[CONTRACT] replay Wiki distillation=all router rows only
[LOSS] fixed coefficient 10; T=1 where applicable; hidden layers=2,3,4,5,6,7,8,9
[STEPS] hidden-KL=$SHORT_STEPS hidden-MSE=$MSE_STEPS vocab-KL=$SHORT_STEPS
[OUTPUT] $SMOKE_ROOT
PLAN

[ "${PLAN_ONLY:-0}" != 1 ] || exit 0
mkdir -p "$SMOKE_ROOT/launcher_logs" "$SMOKE_ROOT/scratch"

COMMON_ENV=(
    PATH="$FLAME_ENV/bin:/usr/bin:/bin"
    PYTHON_BIN="$FLAME_ENV/bin/python"
    PYTHONNOUSERSITE=1
    CUDA_HOME="$FLAME_ENV"
    TORCH_CUDA_ARCH_LIST=9.0
    WANDB_MODE=offline
    TOKENIZER_MODEL=/data3/seonghyeonnoh/LLM-continual-learning-models/pythia-12b-tokenizer
    LOCAL_BASE="$LOCAL_BASE"
    LOCAL_SSD_ROOT="$SMOKE_ROOT/scratch"
    SOURCE_WEIGHTS_DIR="$SOURCE"
    SOURCE_REQUIRED_ITERS=600
    OLD_MODEL_KL_WEIGHTS_DIR="$SOURCE"
    OLD_MODEL_KL_NUM_EXPERTS=16
    MICRO_BATCH_SIZE=1
    GLOBAL_BATCH_SIZE=1
    NPROC_PER_NODE=1
    RUN_INITIAL_PROBE_EVAL=0
    RUN_INITIAL_VALID_EVAL=0
    PROBE_EVAL_ITERS=1
    PROBE_EVAL_INTERVAL=999999
    SECONDARY_PROBE_EVAL_ITERS=1
    SECONDARY_PROBE_EVAL_INTERVAL=999999
    TERTIARY_PROBE_EVAL_ITERS=1
    TERTIARY_PROBE_EVAL_INTERVAL=999999
    EVAL_INTERVAL=999999
    NO_SAVE_OPTIM=1
    LR_WSD_DECAY_ITERS=1
    LOG_INTERVAL=1
    TENSORBOARD_LOG_INTERVAL=1
    MOE_NEW_EXPERT_LR_RAMP_STEPS=0
    MOE_ROUTER_LR_MULTIPLIER=1.0
    LOG_ROUTER_GRAD_NORM_SOURCES=1
)

env "${COMMON_ENV[@]}" \
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" MASTER_PORT=29741 \
    TRAIN_ITERS="$SHORT_STEPS" LR_DECAY_ITERS="$SHORT_STEPS" SAVE_INTERVAL="$SHORT_STEPS" \
    OLD_HIDDEN_KL_COEFF=10 OLD_HIDDEN_KL_COEFF_START= OLD_HIDDEN_KL_COEFF_DECAY_STEPS=0 \
    OLD_HIDDEN_KL_TEMPERATURE=1.0 OLD_HIDDEN_KL_LAYERS=2,3,4,5,6,7,8,9 \
    RUN_ID="smoke-hidden-kl-$SMOKE_TAG" TRAIN_WEIGHTS="$SMOKE_ROOT/hidden_kl" \
    bash "$D/run_g2_ffn_only_code_wiki_joint_old_data_hidden_kl_allrouter_mha.sh" \
    > "$SMOKE_ROOT/launcher_logs/hidden_kl.log" 2>&1 &
PID_HKL=$!

env "${COMMON_ENV[@]}" \
    CUDA_VISIBLE_DEVICES="${GPUS[1]}" MASTER_PORT=29742 \
    TRAIN_ITERS="$MSE_STEPS" LR_DECAY_ITERS="$MSE_STEPS" SAVE_INTERVAL="$MSE_STEPS" \
    OLD_HIDDEN_MSE_COEFF=10 OLD_HIDDEN_MSE_LAYERS=2,3,4,5,6,7,8,9 \
    RUN_ID="smoke-hidden-mse-$SMOKE_TAG" TRAIN_WEIGHTS="$SMOKE_ROOT/hidden_mse" \
    bash "$D/run_g2_ffn_only_code_wiki_joint_old_data_hidden_mse_allrouter_mha.sh" \
    > "$SMOKE_ROOT/launcher_logs/hidden_mse.log" 2>&1 &
PID_HMSE=$!

env "${COMMON_ENV[@]}" \
    CUDA_VISIBLE_DEVICES="${GPUS[2]}" MASTER_PORT=29743 \
    TRAIN_ITERS="$SHORT_STEPS" LR_DECAY_ITERS="$SHORT_STEPS" SAVE_INTERVAL="$SHORT_STEPS" \
    OLD_MODEL_KL_COEFF=10 OLD_MODEL_KL_TEMPERATURE=1.0 \
    RUN_ID="smoke-vocab-kl-$SMOKE_TAG" TRAIN_WEIGHTS="$SMOKE_ROOT/vocab_kl" \
    bash "$D/run_g2_ffn_only_code_wiki_joint_old_data_kd_allrouter_mha.sh" \
    > "$SMOKE_ROOT/launcher_logs/vocab_kl.log" 2>&1 &
PID_VKL=$!

status=0
for spec in "hidden_kl:$PID_HKL" "hidden_mse:$PID_HMSE" "vocab_kl:$PID_VKL"; do
    name="${spec%%:*}"
    pid="${spec##*:}"
    if wait "$pid"; then
        echo "[PASS] $name process"
    else
        echo "[FAIL] $name process; see $SMOKE_ROOT/launcher_logs/$name.log" >&2
        status=1
    fi
done

for spec in "hidden_kl:$SHORT_STEPS" "hidden_mse:$MSE_STEPS" "vocab_kl:$SHORT_STEPS"; do
    name="${spec%%:*}"
    expected="${spec##*:}"
    tracker="$SMOKE_ROOT/$name/latest_checkpointed_iteration.txt"
    if [ ! -f "$tracker" ] || [ "$(tr -d '[:space:]' < "$tracker")" != "$expected" ]; then
        echo "[FAIL] $name missing step-$expected checkpoint" >&2
        status=1
    else
        echo "[PASS] $name step-$expected checkpoint"
    fi
done

exit "$status"
