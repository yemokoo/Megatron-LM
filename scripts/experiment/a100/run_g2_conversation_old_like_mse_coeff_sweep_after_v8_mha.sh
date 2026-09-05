#!/usr/bin/env bash
set -euo pipefail

# Queue a method-matched hidden-MSE coefficient sweep behind the active v8
# five-run chain.  Every run starts independently from the same completed
# 16E->24E MSE-branch expansion KD-init checkpoint.

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="${PIPELINE_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/conversation_old_like_gt_pipeline_20260812}"
PARENT_STUDY_ROOT="${PARENT_STUDY_ROOT:-$PIPELINE_ROOT/final_5run_20pct_contextual_occurrence_exact_axis_method_matched_v8_all_steps3600}"
SWEEP_ROOT="${MSE_COEFF_SWEEP_ROOT:-$PIPELINE_ROOT/mse_coeff_sweep_20pct_contextual_occurrence_exact_axis_v1_all_steps3600}"
SOURCE="${CONVERSATION_MSE_KD_INIT:-$PIPELINE_ROOT/checkpoints/01_expansion_kd_init_e16_to_e24_step600}"
C10_OUTPUT="$PARENT_STUDY_ROOT/checkpoints/02_oldlike_hidden_mse"
CONVERSATION_DIR="${CONVERSATION_TRAIN_DIR:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/conversation/train}"
FLAME_DATA_ROOT="${FLAME_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup}"
GT_ROOT="${CONVERSATION_OLD_LIKE_GT_ROOT:-$PIPELINE_ROOT/analysis/conversation_old_like_gt_l2_l9_top1}"
FLAME_ENV="${FLAME_ENV:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100}"
PYTHON_BIN="${PYTHON_BIN:-$FLAME_ENV/bin/python}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
WANDB_RUN_MODE="${WANDB_RUN_MODE:-online}"
WANDB_GLOBAL_STEP_OFFSET="${WANDB_GLOBAL_STEP_OFFSET:-3600}"
RETIRED_PARENT_PID="${RETIRED_PARENT_PID:-0}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export FLAME_ENV PYTHON_BIN PYTHONNOUSERSITE=1
export PATH="$FLAME_ENV/bin:/usr/bin:/bin"

TRAIN_ITERS=1800
GBS=2304
PRIMARY_MB="${PRIMARY_MICRO_BATCH_SIZE:-96}"
REPLAY_MB="${REPLAY_MICRO_BATCH_SIZE:-48}"
SEQ=512
FULL_TOKENS=$((TRAIN_ITERS * GBS * SEQ))
REPLAY_SAMPLES=$(((FULL_TOKENS / 5) / SEQ))
MSE_ENTRY="$D/run_g2_ffn_only_conversation_wikicode_joint_old_data_hidden_mse_allrouter_mha.sh"

die() { echo "[ERROR] $*" >&2; exit 1; }
checkpoint_at() {
    local root="$1" step="$2"
    [[ -s "$root/latest_checkpointed_iteration.txt" ]] &&
        [[ "$(tr -d '[:space:]' < "$root/latest_checkpointed_iteration.txt")" == "$step" ]] &&
        [[ -s "$root/iter_$(printf '%07d' "$step")/common.pt" ]] &&
        [[ -s "$root/iter_$(printf '%07d' "$step")/.metadata" ]]
}
fresh_or_complete() {
    local root="$1"
    checkpoint_at "$root" 1800 && return 0
    [[ ! -e "$root" ]] && return 0
    [[ -d "$root" && -z "$(find "$root" -mindepth 1 -maxdepth 1 -print -quit)" ]]
}

[[ -x "$PYTHON_BIN" ]] || die "Python missing: $PYTHON_BIN"
[[ "$CUDA_VISIBLE_DEVICES" == "0,1,2,3,4,5,6,7" ]] || die "requires GPUs 0..7"
[[ "$NPROC_PER_NODE" == 8 ]] || die "requires NPROC_PER_NODE=8"
checkpoint_at "$SOURCE" 600 || die "MSE expansion KD-init is incomplete: $SOURCE"
[[ -s "$GT_ROOT/metadata.json" && -s "$GT_ROOT/replay_occurrence_metadata.json" ]] ||
    die "GT metadata is incomplete: $GT_ROOT"

declare -a COEFFS=(0.3 0.4 1.0)
declare -a TAGS=(c0p3 c0p4 c1p0)
for i in "${!COEFFS[@]}"; do
    fresh_or_complete "$SWEEP_ROOT/checkpoints/$((i + 1))_mse_${TAGS[$i]}" ||
        die "refusing nonempty incomplete output for ${TAGS[$i]}"
done

mkdir -p "$SWEEP_ROOT"/{checkpoints,logs,scratch,local}
cat > "$SWEEP_ROOT/run_manifest.json.inprogress" <<EOF
{
  "schema": "conversation_old_like_hidden_mse_coeff_sweep_v1",
  "coefficients": [0.3, 0.4, 1.0],
  "source_checkpoint": "$SOURCE",
  "replay_gt_root": "$GT_ROOT",
  "replay_fraction": 0.2,
  "hidden_layers": [2, 3, 4, 5, 6, 7, 8, 9],
  "wandb_step_offset": $WANDB_GLOBAL_STEP_OFFSET,
  "independent_runs": true
}
EOF
mv "$SWEEP_ROOT/run_manifest.json.inprogress" "$SWEEP_ROOT/run_manifest.json"

echo "$(date -Is) QUEUED waiting_for_c10=$C10_OUTPUT retired_parent_pid=$RETIRED_PARENT_PID" | tee -a "$SWEEP_ROOT/logs/status.tsv"

# The operator may SIGSTOP only the top-level parent chain while its current
# c=10 child keeps running. Once the completed persistent checkpoint appears,
# retire that stopped parent so its previously scheduled hidden-KL/vocab-KL/LM
# commands can never start. Descendant launchers retain the lock until their
# normal teardown completes, so the flock below still prevents GPU overlap.
if (( RETIRED_PARENT_PID > 0 )); then
    parent_cmd="$(ps -p "$RETIRED_PARENT_PID" -o args= 2>/dev/null || true)"
    [[ "$parent_cmd" == *run_g2_conversation_old_like_method_matched_5run_chain_mha.sh* ]] ||
        die "refusing to retire unexpected PID $RETIRED_PARENT_PID: $parent_cmd"
    while ! checkpoint_at "$C10_OUTPUT" 1800; do
        sleep 20
    done
    echo "$(date -Is) C10_COMPLETE retiring_parent_pid=$RETIRED_PARENT_PID" | tee -a "$SWEEP_ROOT/logs/status.tsv"
    kill -KILL "$RETIRED_PARENT_PID" 2>/dev/null || true
fi

# Wait for the c=10 child launcher's normal teardown and lock release, then
# claim the same lock before allocating any sweep model.
exec 9>"$PARENT_STUDY_ROOT/.chain.lock"
flock 9

checkpoint_at "$C10_OUTPUT" 1800 || die "the preceding MSE c10 run did not complete: $C10_OUTPUT"
for gpu in 0 1 2 3 4 5 6 7; do
    apps="$(nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d;/No running processes found/d')"
    [[ -z "$apps" ]] || die "GPU $gpu is busy after parent chain released lock: $apps"
done

run_one() {
    local coeff="$1" tag="$2" ordinal="$3"
    local label="conversation_oldlike_hidden_mse_${tag}_l2to9_exactaxis_sweep_v1"
    local output="$SWEEP_ROOT/checkpoints/${ordinal}_mse_${tag}"
    local port="$((34610 + ordinal))"
    if checkpoint_at "$output" 1800; then
        echo "$(date -Is) SKIP completed $label" | tee -a "$SWEEP_ROOT/logs/status.tsv"
        return
    fi
    mkdir -p "$output"
    echo "$(date -Is) START $label" | tee -a "$SWEEP_ROOT/logs/status.tsv"
    env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" NPROC_PER_NODE="$NPROC_PER_NODE" MASTER_PORT="$port" \
        FLAME_ENV="$FLAME_ENV" PYTHON_BIN="$PYTHON_BIN" FLAME_DATA_ROOT="$FLAME_DATA_ROOT" TOKENIZER_MODEL="$TOKENIZER_MODEL" WANDB_MODE="$WANDB_RUN_MODE" \
        SOURCE_WEIGHTS_DIR="$SOURCE" SOURCE_REQUIRED_ITERS=600 OLD_MODEL_KL_WEIGHTS_DIR="$SOURCE" OLD_MODEL_KL_NUM_EXPERTS=24 \
        TRAIN_DATASET="$CONVERSATION_DIR" JOINT_REPLAY_DATASET="$CONVERSATION_DIR" JOINT_REPLAY_SECONDARY_DATASET= \
        JOINT_REPLAY_DATA_WEIGHT_MODE=equal_dataset MOE_JOINT_REPLAY_OLD_LIKE_GT_PATH="$GT_ROOT" \
        MOE_JOINT_REPLAY_OLD_LIKE_UNIT=token_occurrence MOE_JOINT_REPLAY_OLD_LIKE_SELECTED_TOKEN_COUNT=1311043 \
        MOE_JOINT_REPLAY_OLD_LIKE_FULL_TRAIN_TOKEN_COUNT="$FULL_TOKENS" \
        OLD_LIKE_REPLAY_SUBSET_COUNT=1311043 OLD_LIKE_REPLAY_SUBSET_SHA256=fbff892045f8f1ee5ca90f7648a8352945de98941726c32cf131e65f3ab083ee \
        MOE_JOINT_REPLAY_TOTAL_SAMPLES="$REPLAY_SAMPLES" MOE_JOINT_REPLAY_MICRO_BATCH_SIZE="$REPLAY_MB" \
        TRAIN_ITERS="$TRAIN_ITERS" MICRO_BATCH_SIZE="$PRIMARY_MB" GLOBAL_BATCH_SIZE="$GBS" SEQ_LENGTH="$SEQ" \
        LR=3e-4 MIN_LR=3e-5 LR_DECAY_STYLE=WSD LR_DECAY_ITERS=1800 LR_WSD_DECAY_ITERS=180 LR_WARMUP_FRACTION=0.01 \
        SAVE_INTERVAL=1800 RECOVERY_SAVE_INTERVAL=300 EVAL_INTERVAL=600 LOG_INTERVAL=20 \
        PROBE_EVAL_INTERVAL=100 SECONDARY_PROBE_EVAL_INTERVAL=100 TERTIARY_PROBE_EVAL_INTERVAL=100 \
        PROBE_MICRO_BATCH_SIZE=24 PROBE_EVAL_ITERS=25 SECONDARY_PROBE_EVAL_ITERS=25 TERTIARY_PROBE_EVAL_ITERS=25 \
        PROBE_STEP_OFFSET="$WANDB_GLOBAL_STEP_OFFSET" SECONDARY_PROBE_STEP_OFFSET="$WANDB_GLOBAL_STEP_OFFSET" TERTIARY_PROBE_STEP_OFFSET="$WANDB_GLOBAL_STEP_OFFSET" \
        RUN_INITIAL_PROBE_EVAL=1 RUN_INITIAL_VALID_EVAL=0 WANDB_STEP_OFFSET="$WANDB_GLOBAL_STEP_OFFSET" \
        MOE_JOINT_NEW_EXPERT_QUOTA=0 MOE_JOINT_NEW_EXPERT_QUOTA_SCHEDULE=200:0.5 MOE_JOINT_NEW_EXPERT_QUOTA_MIN_NEW_SLOTS=1 \
        MOE_JOINT_NEW_EXPERT_QUOTA_LOSS_COEFF=0.1 MOE_JOINT_NEW_EXPERT_QUOTA_PRESERVE_NATURAL_GRADS=1 \
        MOE_NEW_EXPERT_LR_RAMP_STEPS=0 MOE_ROUTER_LR_MULTIPLIER=1 LOG_ROUTER_GRAD_NORM_SOURCES=1 \
        ENABLE_OLD_MODEL_KL=1 OLD_MODEL_KL_COEFF=0 MOE_JOINT_REPLAY_LM=1 MOE_JOINT_REPLAY_OLD_DATA_KD=0 \
        MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=0 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=1 \
        OLD_HIDDEN_MSE_COEFF="$coeff" OLD_HIDDEN_MSE_LAYERS=2,3,4,5,6,7,8,9 \
        NO_SAVE_OPTIM=1 DIRECT_LOCAL_SAVE=1 LOCAL_BASE="$SWEEP_ROOT/local" LOCAL_SSD_ROOT="$SWEEP_ROOT/scratch/$label" \
        RUN_ID="$label" TRAIN_WEIGHTS="$output" WANDB_RUN_ID="$label" WANDB_EXP_NAME="$label" \
        bash "$MSE_ENTRY" >> "$SWEEP_ROOT/logs/$label.log" 2>&1
    checkpoint_at "$output" 1800 || die "$label exited without step 1800"
    echo "$(date -Is) DONE $label" | tee -a "$SWEEP_ROOT/logs/status.tsv"
}

run_one 0.3 c0p3 01
run_one 0.4 c0p4 02
run_one 1.0 c1p0 03
echo "$(date -Is) COMPLETE mse coefficient sweep" | tee -a "$SWEEP_ROOT/logs/status.tsv"
