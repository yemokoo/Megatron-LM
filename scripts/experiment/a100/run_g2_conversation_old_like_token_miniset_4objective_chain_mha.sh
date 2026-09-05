#!/usr/bin/env bash
set -euo pipefail

# Four independent Conversation 1-Phase runs from the same 24E expansion
# KD-init.  Primary is full Conversation LM.  Replay contains only GT-positive
# Conversation token IDs, repeated to exactly 20% of primary token exposure;
# replay gradients update router rows only.

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="${PIPELINE_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/conversation_old_like_gt_pipeline_20260812}"
SOURCE="${CONVERSATION_KD_INIT_CHECKPOINT:-$PIPELINE_ROOT/checkpoints/01_expansion_kd_init_e16_to_e24_step600}"
CONVERSATION_DIR="${CONVERSATION_TRAIN_DIR:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/conversation/train}"
FLAME_DATA_ROOT="${FLAME_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup}"
MINISET_DIR="${CONVERSATION_MINISET_DIR:-$PIPELINE_ROOT/data/conversation_old_like_gt_token_miniset/train}"
STUDY_ROOT="${CONVERSATION_STUDY_ROOT:-$PIPELINE_ROOT/final_4objective_20pct}"
FLAME_ENV="${FLAME_ENV:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100}"
PYTHON_BIN="${PYTHON_BIN:-$FLAME_ENV/bin/python}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TRAIN_ITERS=1800
GBS=2304
PRIMARY_MB="${PRIMARY_MICRO_BATCH_SIZE:-96}"
REPLAY_MB="${REPLAY_MICRO_BATCH_SIZE:-48}"
SEQ=512
FULL_TOKENS=$((TRAIN_ITERS * GBS * SEQ))
REPLAY_TOKENS=$((FULL_TOKENS / 5))
REPLAY_SAMPLES=$((REPLAY_TOKENS / SEQ))
PLAN_ONLY="${PLAN_ONLY:-0}"

LM_ENTRY="$D/run_g2_ffn_only_conversation_wikicode_joint_lm_allrouter_mha.sh"
MSE_ENTRY="$D/run_g2_ffn_only_conversation_wikicode_joint_old_data_hidden_mse_allrouter_mha.sh"
HKL_ENTRY="$D/run_g2_ffn_only_conversation_wikicode_joint_old_data_hidden_kl_allrouter_mha.sh"
VKL_ENTRY="$D/run_g2_ffn_only_conversation_wikicode_joint_old_data_kd_allrouter_mha.sh"

die() { echo "[ERROR] $*" >&2; exit 1; }
checkpoint_at() {
    local root="$1" step="$2"
    [[ -s "$root/latest_checkpointed_iteration.txt" ]] &&
        [[ "$(tr -d '[:space:]' < "$root/latest_checkpointed_iteration.txt")" == "$step" ]] &&
        [[ -s "$root/iter_$(printf '%07d' "$step")/common.pt" ]]
}
fresh_or_complete() {
    local root="$1"
    checkpoint_at "$root" 1800 && return 0
    [[ ! -e "$root" ]] && return 0
    [[ -d "$root" && -z "$(find "$root" -mindepth 1 -maxdepth 1 -print -quit)" ]]
}

checkpoint_at "$SOURCE" 600 || die "24E expansion KD-init is incomplete: $SOURCE"
[[ -s "$MINISET_DIR/miniset_metadata.json" ]] || die "Conversation GT token miniset missing"
[[ -x "$PYTHON_BIN" ]] || die "Python missing: $PYTHON_BIN"
[[ "$CUDA_VISIBLE_DEVICES" == "0,1,2,3,4,5,6,7" && "$NPROC_PER_NODE" == 8 ]] || die "matched runs require GPUs 0..7"
"$PYTHON_BIN" - "$MINISET_DIR/miniset_metadata.json" "$FULL_TOKENS" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); full=int(sys.argv[2])
assert m['complete'] and m['schema']=='old_like_gt_token_packed_indexed_dataset_v1'
assert m['full_train_token_count']==full, (m['full_train_token_count'],full)
assert abs(m['target_train_fraction']-0.2)<1e-12
assert m['selected_token_count']>0
print('[VALID] miniset tokens={:,} effective_epochs={:.3f}'.format(
    m['selected_token_count'],m['effective_token_epochs']))
PY
for output in "$STUDY_ROOT"/checkpoints/{01_hidden_mse,02_hidden_kl,03_vocab_kl,04_lm}; do
    fresh_or_complete "$output" || die "refusing nonempty incomplete output: $output"
done

cat <<EOF
[PLAN] source: $SOURCE (24E Conversation expansion KD-init step 600)
[PLAN] primary: full Conversation LM, 1800 steps, MB=$PRIMARY_MB, GBS=$GBS
[PLAN] replay: GT-positive token-only indexed miniset; no Wiki/Code replay data
[PLAN] replay exposure: $REPLAY_SAMPLES sequences = $REPLAY_TOKENS tokens = 20%
[PLAN] replay MB=$REPLAY_MB; replay gradients router-only; natural routing
[PLAN] objectives: hidden-MSE L2-9 c10, hidden-KL L2-9 c1, vocab-KL c1, LM
[PLAN] every objective independently restarts from the same source
EOF
[[ "$PLAN_ONLY" == 1 ]] && exit 0

mkdir -p "$STUDY_ROOT/checkpoints" "$STUDY_ROOT/logs" "$STUDY_ROOT/scratch"
exec 9>"$STUDY_ROOT/.chain.lock"
flock -n 9 || die "final 4-objective chain is already active"
for gpu in 0 1 2 3 4 5 6 7; do
    apps="$(nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d;/No running processes found/d')"
    [[ -z "$apps" ]] || die "GPU $gpu is busy: $apps"
done

run_stage() {
    local label="$1" output="$2" port="$3" entry="$4"
    shift 4
    if checkpoint_at "$output" 1800; then
        echo "[SKIP] $label complete"
        return
    fi
    mkdir -p "$output"
    echo "$(date -Is) START $label" | tee -a "$STUDY_ROOT/logs/status.tsv"
    env \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" NPROC_PER_NODE="$NPROC_PER_NODE" MASTER_PORT="$port" \
        FLAME_ENV="$FLAME_ENV" PYTHON_BIN="$PYTHON_BIN" FLAME_DATA_ROOT="$FLAME_DATA_ROOT" \
        TOKENIZER_MODEL="$TOKENIZER_MODEL" WANDB_MODE=offline \
        SOURCE_WEIGHTS_DIR="$SOURCE" SOURCE_REQUIRED_ITERS=600 \
        TRAIN_DATASET="$CONVERSATION_DIR" JOINT_REPLAY_DATASET="$MINISET_DIR" JOINT_REPLAY_SECONDARY_DATASET= \
        JOINT_REPLAY_DATA_WEIGHT_MODE=equal_dataset MOE_JOINT_REPLAY_OLD_LIKE_GT_PATH= \
        MOE_JOINT_REPLAY_TOTAL_SAMPLES="$REPLAY_SAMPLES" MOE_JOINT_REPLAY_MICRO_BATCH_SIZE="$REPLAY_MB" \
        TRAIN_ITERS="$TRAIN_ITERS" MICRO_BATCH_SIZE="$PRIMARY_MB" GLOBAL_BATCH_SIZE="$GBS" SEQ_LENGTH="$SEQ" \
        LR=3e-4 MIN_LR=3e-5 LR_DECAY_STYLE=WSD LR_DECAY_ITERS=1800 LR_WSD_DECAY_ITERS=180 LR_WARMUP_FRACTION=0.01 \
        SAVE_INTERVAL=1800 RECOVERY_SAVE_INTERVAL=300 EVAL_INTERVAL=600 LOG_INTERVAL=20 \
        PROBE_EVAL_INTERVAL=100 SECONDARY_PROBE_EVAL_INTERVAL=100 TERTIARY_PROBE_EVAL_INTERVAL=100 \
        PROBE_MICRO_BATCH_SIZE=24 PROBE_EVAL_ITERS=25 SECONDARY_PROBE_EVAL_ITERS=25 TERTIARY_PROBE_EVAL_ITERS=25 \
        RUN_INITIAL_PROBE_EVAL=1 RUN_INITIAL_VALID_EVAL=0 \
        MOE_JOINT_NEW_EXPERT_QUOTA=0 MOE_JOINT_NEW_EXPERT_QUOTA_SCHEDULE=200:0.5 \
        MOE_JOINT_NEW_EXPERT_QUOTA_MIN_NEW_SLOTS=1 MOE_JOINT_NEW_EXPERT_QUOTA_LOSS_COEFF=0.1 \
        MOE_JOINT_NEW_EXPERT_QUOTA_PRESERVE_NATURAL_GRADS=1 \
        MOE_NEW_EXPERT_LR_RAMP_STEPS=0 MOE_ROUTER_LR_MULTIPLIER=1 LOG_ROUTER_GRAD_NORM_SOURCES=0 \
        NO_SAVE_OPTIM=1 DIRECT_LOCAL_SAVE=1 LOCAL_BASE="$STUDY_ROOT/local" LOCAL_SSD_ROOT="$STUDY_ROOT/scratch/$label" \
        RUN_ID="$label" TRAIN_WEIGHTS="$output" WANDB_RUN_ID="$label" WANDB_EXP_NAME="$label" \
        OLD_MODEL_KL_NUM_EXPERTS=24 "$@" \
        bash "$entry" >> "$STUDY_ROOT/logs/$label.log" 2>&1
    checkpoint_at "$output" 1800 || die "$label exited without step 1800"
    echo "$(date -Is) DONE $label" | tee -a "$STUDY_ROOT/logs/status.tsv"
}

run_stage conversation_oldlike_token_hidden_mse_c10_l2to9 "$STUDY_ROOT/checkpoints/01_hidden_mse" 34401 "$MSE_ENTRY" \
    ENABLE_OLD_MODEL_KL=1 OLD_MODEL_KL_WEIGHTS_DIR="$SOURCE" OLD_MODEL_KL_COEFF=0 \
    MOE_JOINT_REPLAY_LM=1 MOE_JOINT_REPLAY_OLD_DATA_KD=0 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=0 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=1 \
    OLD_HIDDEN_MSE_COEFF=10 OLD_HIDDEN_MSE_LAYERS=2,3,4,5,6,7,8,9

run_stage conversation_oldlike_token_hidden_kl_c1_l2to9 "$STUDY_ROOT/checkpoints/02_hidden_kl" 34402 "$HKL_ENTRY" \
    ENABLE_OLD_MODEL_KL=1 OLD_MODEL_KL_WEIGHTS_DIR="$SOURCE" OLD_MODEL_KL_COEFF=0 \
    MOE_JOINT_REPLAY_LM=1 MOE_JOINT_REPLAY_OLD_DATA_KD=0 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=1 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=0 \
    OLD_HIDDEN_KL_COEFF=1 OLD_HIDDEN_KL_TEMPERATURE=1 OLD_HIDDEN_KL_LAYERS=2,3,4,5,6,7,8,9

run_stage conversation_oldlike_token_vocab_kl_c1 "$STUDY_ROOT/checkpoints/03_vocab_kl" 34403 "$VKL_ENTRY" \
    ENABLE_OLD_MODEL_KL=1 OLD_MODEL_KL_WEIGHTS_DIR="$SOURCE" OLD_MODEL_KL_COEFF=1 OLD_MODEL_KL_TEMPERATURE=1 \
    MOE_JOINT_REPLAY_LM=1 MOE_JOINT_REPLAY_OLD_DATA_KD=1 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=0 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=0

run_stage conversation_oldlike_token_lm "$STUDY_ROOT/checkpoints/04_lm" 34404 "$LM_ENTRY" \
    ENABLE_OLD_MODEL_KL=0 OLD_MODEL_KL_COEFF=0 \
    MOE_JOINT_REPLAY_LM=1 MOE_JOINT_REPLAY_OLD_DATA_KD=0 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=0 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=0

echo "$(date -Is) COMPLETE all four objectives" | tee -a "$STUDY_ROOT/logs/status.tsv"
