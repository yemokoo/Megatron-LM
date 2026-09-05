#!/usr/bin/env bash
set -euo pipefail

# Four independent 1-Phase runs from the same post-expansion KD-init model.
# Primary: full Code LM on every one of 1,800 steps.
# Replay: a standalone indexed dataset containing only old-like GT token IDs.
#         It is repeated to exactly the existing router-FT 20% data budget and
#         distributed inside the same 1-Phase Code optimization run.

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/../../.." && pwd)"

SOURCE="${SOURCE_CHECKPOINT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/g2_olddata_kd_9run_20260808/01_common_kd_init/code_e8_to_e16_wiki_kd_step600}"
CODE_DIR="${CODE_TRAIN_DIR:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/code/train}"
FLAME_DATA_ROOT="${FLAME_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup}"
MINISET_DIR="${OLD_LIKE_MINISET_DIR:-/data2/seonghyeonnoh/LLM-continual-learning-data/old_like_gt_token_miniset_all8_top1_20260812/train}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6}"
STUDY_ROOT="${STUDY_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/code_old_like_gt_token_miniset_20pct_1phase_4objective_v4_8gpu_mb96_probe24_kdfirst_20260812}"
LOG_ROOT="$STUDY_ROOT/logs"
LOCAL_ROOT="$STUDY_ROOT/scratch"

TRAIN_ITERS=1800
GBS=2304
MB=96
SEQ=512
EXPECTED_SELECTED=2658787
EXPECTED_FULL_TOKENS=2123366400
EXPECTED_REPLAY_GLOBAL_BATCHES=360
EXPECTED_REPLAY_TOKENS=424673280
REPLAY_MB=48
EXPECTED_REPLAY_GLOBAL_MICROBATCHES=2160
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PLAN_ONLY="${PLAN_ONLY:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

LM_ENTRY="$D/run_g2_ffn_only_code_wiki_joint_lm_allrouter_mha.sh"
MSE_ENTRY="$D/run_g2_ffn_only_code_wiki_joint_old_data_hidden_mse_allrouter_mha.sh"
HKL_ENTRY="$D/run_g2_ffn_only_code_wiki_joint_old_data_hidden_kl_allrouter_mha.sh"
VKL_ENTRY="$D/run_g2_ffn_only_code_wiki_joint_old_data_kd_allrouter_mha.sh"

die() { echo "[ERROR] $*" >&2; exit 1; }
checkpoint_complete() {
    local root="$1"
    [[ -s "$root/latest_checkpointed_iteration.txt" ]] &&
        [[ "$(tr -d '[:space:]' < "$root/latest_checkpointed_iteration.txt")" == "$TRAIN_ITERS" ]] &&
        [[ -s "$root/iter_0001800/common.pt" ]] &&
        [[ -s "$root/iter_0001800/.metadata" ]]
}
fresh_or_complete() {
    local root="$1"
    checkpoint_complete "$root" && return 0
    [[ ! -e "$root" ]] && return 0
    [[ -d "$root" && -z "$(find "$root" -mindepth 1 -maxdepth 1 -print -quit)" ]]
}

preflight() {
    [[ "$CUDA_VISIBLE_DEVICES" == "0,1,2,3,4,5,6,7" ]] || die "expected GPUs 0,1,2,3,4,5,6,7"
    [[ "$NPROC_PER_NODE" == 8 ]] || die "expected NPROC_PER_NODE=8"
    [[ -s "$SOURCE/latest_checkpointed_iteration.txt" ]] || die "source tracker missing"
    [[ "$(tr -d '[:space:]' < "$SOURCE/latest_checkpointed_iteration.txt")" == 600 ]] || die "source is not step 600"
    [[ -s "$CODE_DIR/train_text_document.bin" && -s "$CODE_DIR/train_text_document.idx" ]] || die "Code dataset missing"
    [[ -s "$MINISET_DIR/train_text_document.bin" && -s "$MINISET_DIR/train_text_document.idx" ]] || die "old-like token miniset missing"
    [[ -s "$MINISET_DIR/miniset_metadata.json" ]] || die "miniset metadata missing"
    python3 - "$MINISET_DIR/miniset_metadata.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
p = Path(sys.argv[1]); m = json.loads(p.read_text())
root = p.parent
checks = {
  'schema': m.get('schema') == 'old_like_gt_token_packed_indexed_dataset_v1',
  'complete': m.get('complete') is True,
  'selected': m.get('selected_token_count') == 2658787,
  'full': m.get('full_train_token_count') == 2123366400,
  'fraction': m.get('target_train_fraction') == 0.2,
  'steps': m.get('global_batch_size_2304_steps') == 360,
  'bin_sha': hashlib.sha256((root/m['bin_file']).read_bytes()).hexdigest() == m['bin_sha256'],
  'idx_sha': hashlib.sha256((root/m['idx_file']).read_bytes()).hexdigest() == m['idx_sha256'],
}
bad = [k for k,v in checks.items() if not v]
if bad: raise SystemExit('miniset validation failed: '+','.join(bad))
print('[PREFLIGHT] miniset hashes/counts validated')
PY
    [[ $((EXPECTED_REPLAY_GLOBAL_BATCHES * GBS * SEQ)) == "$EXPECTED_REPLAY_TOKENS" ]] || die "replay-token budget mismatch"
    [[ $((EXPECTED_FULL_TOKENS / 5)) == "$EXPECTED_REPLAY_TOKENS" ]] || die "replay exposure must be exactly 20pct"
    [[ $((EXPECTED_REPLAY_GLOBAL_MICROBATCHES * REPLAY_MB * NPROC_PER_NODE * SEQ)) == "$EXPECTED_REPLAY_TOKENS" ]] || die "replay microbatch schedule mismatch"
    local output gpu apps
    for output in "$STUDY_ROOT"/checkpoints/{01_hidden_mse,02_hidden_kl,03_vocab_kl,04_lm}; do
        fresh_or_complete "$output" || die "refusing nonempty incomplete output: $output"
    done
    for gpu in 0 1 2 3 4 5 6 7; do
        apps="$(nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d;/No running processes found/d')"
        [[ -z "$apps" ]] || die "GPU $gpu is busy: $apps"
    done
    echo "[PREFLIGHT] 1-Phase replay=$EXPECTED_REPLAY_GLOBAL_BATCHES x 2304 x 512=$EXPECTED_REPLAY_TOKENS tokens (20pct); effective miniset token epochs=$(python3 -c "print($EXPECTED_REPLAY_TOKENS/$EXPECTED_SELECTED)")"
}

run_stage() {
    local label="$1" output="$2" port="$3" entry="$4"
    shift 4
    if checkpoint_complete "$output"; then
        echo "[SKIP] $label already complete: $output"
        return
    fi
    mkdir -p "$output" "$LOG_ROOT"
    echo "$(date -Is) START $label $output" | tee -a "$LOG_ROOT/status.tsv"
    env \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" NPROC_PER_NODE="$NPROC_PER_NODE" MASTER_PORT="$port" \
        FLAME_ENV=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100 \
        PYTHON_BIN=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin/python \
        PATH=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin:/usr/bin:/bin \
        CUDA_HOME=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100 \
        FLAME_DATA_ROOT="$FLAME_DATA_ROOT" \
        TOKENIZER_MODEL="$TOKENIZER_MODEL" WANDB_MODE=offline \
        SOURCE_WEIGHTS_DIR="$SOURCE" SOURCE_REQUIRED_ITERS=600 \
        TRAIN_DATASET="$CODE_DIR" JOINT_REPLAY_DATASET="$MINISET_DIR" \
        JOINT_REPLAY_DATA_WEIGHT_MODE=equal_dataset \
        MOE_JOINT_REPLAY_OLD_LIKE_GT_PATH= \
        MOE_JOINT_REPLAY_TOTAL_SAMPLES=$((EXPECTED_REPLAY_TOKENS / SEQ)) \
        MOE_JOINT_REPLAY_MICRO_BATCH_SIZE="$REPLAY_MB" \
        TRAIN_ITERS="$TRAIN_ITERS" MICRO_BATCH_SIZE="$MB" GLOBAL_BATCH_SIZE="$GBS" SEQ_LENGTH="$SEQ" \
        DATASET_SPLIT=100,0,0 LR=3e-4 MIN_LR=3e-5 LR_DECAY_STYLE=WSD LR_DECAY_ITERS="$TRAIN_ITERS" \
        LR_WSD_DECAY_ITERS=180 LR_WARMUP_FRACTION=0.01 SAVE_INTERVAL="$TRAIN_ITERS" RECOVERY_SAVE_INTERVAL=300 \
        EVAL_INTERVAL=600 LOG_INTERVAL=20 PROBE_EVAL_INTERVAL=100 SECONDARY_PROBE_EVAL_INTERVAL=100 \
        PROBE_MICRO_BATCH_SIZE=24 \
        PROBE_EVAL_ITERS=25 SECONDARY_PROBE_EVAL_ITERS=25 TERTIARY_PROBE_EVAL_INTERVAL=0 \
        RUN_INITIAL_PROBE_EVAL=1 RUN_INITIAL_VALID_EVAL=0 \
        MOE_JOINT_NEW_EXPERT_QUOTA=0 MOE_JOINT_NEW_EXPERT_QUOTA_SCHEDULE=200:0.5 \
        MOE_JOINT_NEW_EXPERT_QUOTA_MIN_NEW_SLOTS=1 MOE_JOINT_NEW_EXPERT_QUOTA_LOSS_COEFF=0.1 \
        MOE_JOINT_NEW_EXPERT_QUOTA_PRESERVE_NATURAL_GRADS=1 \
        MOE_SEPARATE_ROUTER_EXPERT_GRAD_CLIP=1 MOE_ALLOW_PARTIAL_OPTIMIZER_STATE=0 \
        MOE_NEW_EXPERT_LR_RAMP_STEPS=0 MOE_ROUTER_LR_MULTIPLIER=1.0 LOG_ROUTER_GRAD_NORM_SOURCES=0 \
        MOE_AUX_LOSS_COEFF=0.01 MOE_Z_LOSS_COEFF=0.001 NO_SAVE_OPTIM=1 DIRECT_LOCAL_SAVE=1 \
        LOCAL_BASE="$STUDY_ROOT/local" LOCAL_SSD_ROOT="$LOCAL_ROOT" \
        RUN_ID="$label" TRAIN_WEIGHTS="$output" WANDB_RUN_ID="$label" WANDB_EXP_NAME="$label" \
        "$@" bash "$entry"
    checkpoint_complete "$output" || die "$label exited without a complete step-1800 checkpoint"
    echo "$(date -Is) DONE $label $output" | tee -a "$LOG_ROOT/status.tsv"
}

cat <<EOF
[PLAN] source: Wiki-trained, expanded, KD-init step 600
[PLAN] primary: full Code LM, 1800 steps
[PLAN] replay dataset: only 2,658,787 GT token IDs in standalone .bin/.idx
[PLAN] replay budget: 360 global batches = 829,440 sequences = 424,673,280 tokens = full Code train 20%
[PLAN] repetition: GT-only miniset is revisited for 159.724446 effective token epochs
[PLAN] GPUs=8, primary MB=96, replay MB=48, GBS=2304
[PLAN] probe MB=24, so 25 probe iters evaluate the same 4,800 samples as the old 4-GPU MB48 runs
[PLAN] 1-Phase accumulation: every update computes Code first, then 1 or 2 GT replay microbatches, then calls optimizer.step once
[PLAN] per-update exposure: Code=2304 sequences; GT=384 or 768 sequences (mean 460.8, exact 5:1 total ratio)
[PLAN] objective weighting: mean(Code loss) + mean(GT replay loss), equal branch coefficient 1 as in the existing 1-Phase implementation
[PLAN] objectives/order: layer-output MSE L2-9, layer-output KL L2-9, vocabulary KL, LM
[PLAN] all runs independently start from the same source; natural routing; replay gradients router-only
[PLAN] output: $STUDY_ROOT
EOF

[[ "$PLAN_ONLY" == 1 ]] && exit 0
mkdir -p "$STUDY_ROOT/checkpoints" "$LOG_ROOT" "$LOCAL_ROOT"
exec 9>"$STUDY_ROOT/.chain.lock"
flock -n 9 || die "chain lock is already held"
preflight
[[ "$PREFLIGHT_ONLY" == 1 ]] && exit 0

run_stage old_like_token_miniset_hidden_mse_c10_l2to9 "$STUDY_ROOT/checkpoints/01_hidden_mse" 34211 "$MSE_ENTRY" \
    ENABLE_OLD_MODEL_KL=1 OLD_MODEL_KL_WEIGHTS_DIR="$SOURCE" OLD_MODEL_KL_NUM_EXPERTS=16 OLD_MODEL_KL_COEFF=0 \
    MOE_JOINT_REPLAY_LM=1 MOE_JOINT_REPLAY_OLD_DATA_KD=0 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=0 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=1 \
    OLD_HIDDEN_MSE_COEFF=10 OLD_HIDDEN_MSE_LAYERS=2,3,4,5,6,7,8,9

run_stage old_like_token_miniset_hidden_kl_c1_l2to9 "$STUDY_ROOT/checkpoints/02_hidden_kl" 34212 "$HKL_ENTRY" \
    ENABLE_OLD_MODEL_KL=1 OLD_MODEL_KL_WEIGHTS_DIR="$SOURCE" OLD_MODEL_KL_NUM_EXPERTS=16 OLD_MODEL_KL_COEFF=0 \
    MOE_JOINT_REPLAY_LM=1 MOE_JOINT_REPLAY_OLD_DATA_KD=0 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=1 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=0 \
    OLD_HIDDEN_KL_COEFF=1 OLD_HIDDEN_KL_LAYERS=2,3,4,5,6,7,8,9

run_stage old_like_token_miniset_vocab_kl_c1 "$STUDY_ROOT/checkpoints/03_vocab_kl" 34213 "$VKL_ENTRY" \
    ENABLE_OLD_MODEL_KL=1 OLD_MODEL_KL_WEIGHTS_DIR="$SOURCE" OLD_MODEL_KL_NUM_EXPERTS=16 OLD_MODEL_KL_COEFF=1 \
    MOE_JOINT_REPLAY_LM=1 MOE_JOINT_REPLAY_OLD_DATA_KD=1 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=0 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=0

run_stage old_like_token_miniset_lm "$STUDY_ROOT/checkpoints/04_lm" 34214 "$LM_ENTRY" \
    ENABLE_OLD_MODEL_KL=0 OLD_MODEL_KL_COEFF=0 MOE_JOINT_REPLAY_LM=1 \
    MOE_JOINT_REPLAY_OLD_DATA_KD=0 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=0 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=0

echo "$(date -Is) COMPLETE all four objectives" | tee -a "$LOG_ROOT/status.tsv"
