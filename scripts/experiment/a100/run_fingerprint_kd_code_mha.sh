#!/usr/bin/env bash
set -euo pipefail

# Matched Code training from the expansion-KD-init-complete checkpoint.
# FP_MODE: lm_only | soft_stable | hard_stable | soft_random | soft_random_permuted | soft_full
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FP_MODE="${FP_MODE:?set FP_MODE=lm_only|soft_stable|hard_stable|soft_random|soft_random_permuted|soft_full}"
case "$FP_MODE" in lm_only|soft_stable|hard_stable|soft_random|soft_random_permuted|soft_full) ;; *) echo "[ERROR] invalid FP_MODE=$FP_MODE" >&2; exit 2 ;; esac

STUDY_ROOT="${STUDY_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/layer_output_fingerprint_kd_20260810}"
SOURCE_WEIGHTS_DIR="${SOURCE_WEIGHTS_DIR:-/data2/seonghyeonnoh/LLM-continual-learning-runs/g2_olddata_kd_9run_20260808/01_common_kd_init/code_e8_to_e16_wiki_kd_step600}"
SOURCE_REQUIRED_ITERS="${SOURCE_REQUIRED_ITERS:-600}"
TEACHER_WEIGHTS_DIR="${TEACHER_WEIGHTS_DIR:-/data2/seonghyeonnoh/LLM-continual-learning-runs/g2_olddata_kd_9run_20260808/01_common_kd_init/code_e8_to_e16_wiki_kd_step600}"
FINGERPRINT_KD_BUNDLE="${FINGERPRINT_KD_BUNDLE:-$STUDY_ROOT/fingerprint/score_bundle.npz}"
# The generic launcher stages roughly 13 GiB of Code data + source weights per
# run.  The host root filesystem is intentionally small, so keep this study's
# staging and canonical dataset lookup on /data2 unless the caller overrides
# them explicitly.
FLAME_DATA_ROOT="${FLAME_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup}"
LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-$STUDY_ROOT/staging}"
TRAIN_ITERS="${TRAIN_ITERS:-200}"
RUN_ID="${RUN_ID:-fingerprint-kd-${FP_MODE}-code-${TRAIN_ITERS}step}"
TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$STUDY_ROOT/checkpoints/$RUN_ID}"

[[ -f "$SOURCE_WEIGHTS_DIR/latest_checkpointed_iteration.txt" ]] || { echo "[ERROR] source tracker missing" >&2; exit 1; }
[[ "$(tr -d '[:space:]' < "$SOURCE_WEIGHTS_DIR/latest_checkpointed_iteration.txt")" == "$SOURCE_REQUIRED_ITERS" ]] || {
    echo "[ERROR] source tracker does not match expected step $SOURCE_REQUIRED_ITERS" >&2; exit 1;
}
[[ -f "$TEACHER_WEIGHTS_DIR/latest_checkpointed_iteration.txt" ]] || { echo "[ERROR] teacher tracker missing" >&2; exit 1; }
if [[ -e "$TRAIN_WEIGHTS" && "${ALLOW_EXISTING_OUTPUT:-0}" != 1 ]]; then
    echo "[ERROR] refusing to overwrite existing output: $TRAIN_WEIGHTS" >&2
    exit 1
fi

export STUDY_ROOT SOURCE_WEIGHTS_DIR SOURCE_REQUIRED_ITERS TEACHER_WEIGHTS_DIR
export FINGERPRINT_KD_BUNDLE FLAME_DATA_ROOT LOCAL_SSD_ROOT
export TRAIN_ITERS RUN_ID TRAIN_WEIGHTS
export SOURCE_NUM_EXPERTS=8 NUM_EXPERTS=16 MOE_ROUTER_TOPK=4 MOE_FFN_HIDDEN_SIZE=352
export LOAD_EXPANDED_SOURCE=1 FREEZE_SHARED=1 TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY=1
export TRAIN_ATTENTION_WITH_NEW_EXPERTS=0 FREEZE_DENSE_ATTENTION_LORA_WITH_NEW_EXPERTS=1
export ENABLE_OLD_MODEL_KL=0 OLD_MODEL_KL_COEFF=0.0
export OLD_MODEL_KL_WEIGHTS_DIR="$TEACHER_WEIGHTS_DIR" OLD_MODEL_KL_NUM_EXPERTS=16
export MOE_EXPANSION_DISTILL_MODE=none
export MOE_JOINT_REPLAY_LM=0 MOE_JOINT_REPLAY_OLD_DATA_KD=0
export MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=0 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=0
export MOE_INTERLEAVE_CODE_STEPS=0 MOE_INTERLEAVE_ROUTER_STEPS=0
export TRAIN_ROUTER_USAGE_LOG_INTERVAL="${TRAIN_ROUTER_USAGE_LOG_INTERVAL:-20}"
export TRAIN_ROUTER_USAGE_NUM_EXISTING_EXPERTS="${TRAIN_ROUTER_USAGE_NUM_EXISTING_EXPERTS:-8}"
export DIRECT_LOCAL_SAVE=1 NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-1}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-48}" GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-$TRAIN_ITERS}" RECOVERY_SAVE_INTERVAL=0
export EVAL_INTERVAL="${EVAL_INTERVAL:-$TRAIN_ITERS}" LOG_INTERVAL="${LOG_INTERVAL:-10}"
export RUN_INITIAL_PROBE_EVAL="${RUN_INITIAL_PROBE_EVAL:-1}" RUN_INITIAL_VALID_EVAL="${RUN_INITIAL_VALID_EVAL:-1}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-20}" SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-20}"
export PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-25}" SECONDARY_PROBE_EVAL_ITERS="${SECONDARY_PROBE_EVAL_ITERS:-25}"
export WANDB_MODE="${WANDB_MODE:-offline}"

export FINGERPRINT_KD_ENABLE=0 FINGERPRINT_KD_FORCE_ENABLE_ZERO_COEFF=0
export FINGERPRINT_KD_COEFF="${FINGERPRINT_KD_COEFF:-0.0}"
export FINGERPRINT_KD_RANK=64 FINGERPRINT_KD_LAYERS=2,3,4,5,6,7,8,9
export FINGERPRINT_KD_SCORE_REPRESENTATION=stable
export FINGERPRINT_KD_THRESHOLD=0.1297607421875
export FINGERPRINT_KD_SOFT_TEMPERATURE=0.0069580078125
export FINGERPRINT_KD_WEIGHT_ASSIGNMENT=stable
case "$FP_MODE" in
    lm_only)
        ;;
    soft_stable)
        export FINGERPRINT_KD_ENABLE=1 FINGERPRINT_KD_GATE_MODE=soft FINGERPRINT_KD_LOSS_REPRESENTATION=stable
        ;;
    hard_stable)
        export FINGERPRINT_KD_ENABLE=1 FINGERPRINT_KD_GATE_MODE=hard FINGERPRINT_KD_LOSS_REPRESENTATION=stable
        ;;
    soft_random)
        export FINGERPRINT_KD_ENABLE=1 FINGERPRINT_KD_GATE_MODE=soft FINGERPRINT_KD_LOSS_REPRESENTATION=random
        ;;
    soft_random_permuted)
        export FINGERPRINT_KD_ENABLE=1 FINGERPRINT_KD_GATE_MODE=soft FINGERPRINT_KD_LOSS_REPRESENTATION=random
        export FINGERPRINT_KD_WEIGHT_ASSIGNMENT=permuted
        ;;
    soft_full)
        export FINGERPRINT_KD_ENABLE=1 FINGERPRINT_KD_GATE_MODE=soft FINGERPRINT_KD_LOSS_REPRESENTATION=full
        ;;
esac
if [[ "${FORCE_ZERO_COEFF_DIAGNOSTIC:-0}" == 1 ]]; then
    export FINGERPRINT_KD_ENABLE=1 FINGERPRINT_KD_FORCE_ENABLE_ZERO_COEFF=1 FINGERPRINT_KD_COEFF=0.0
fi

echo "[FINGERPRINT KD] mode=$FP_MODE source=$SOURCE_WEIGHTS_DIR(step=$SOURCE_REQUIRED_ITERS) iters=$TRAIN_ITERS"
echo "[FINGERPRINT KD] frozen teacher=$TEACHER_WEIGHTS_DIR"
echo "[FINGERPRINT KD] old replay=off forced routing=off natural routing=on"
echo "[FINGERPRINT KD] output=$TRAIN_WEIGHTS coeff=$FINGERPRINT_KD_COEFF weight_assignment=$FINGERPRINT_KD_WEIGHT_ASSIGNMENT"

exec bash "$SCRIPT_DIR/run_g2_ffn_only_code_from_distill_init_mha.sh" logits
