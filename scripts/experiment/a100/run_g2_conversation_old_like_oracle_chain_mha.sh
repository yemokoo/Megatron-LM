#!/usr/bin/env bash
set -euo pipefail

# Build the two checkpoints required before Conversation old-like GT analysis:
#   1) Code hidden-MSE result -> expand 16E to 24E and output-KD init (600)
#   2) from that KD-init, Conversation LM + Wiki/Code replay LM (1800)
# The second model is an oracle used only to label Conversation occurrences.

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/../../.." && pwd)"

SOURCE="${SOURCE_CHECKPOINT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/code_old_like_gt_token_miniset_20pct_1phase_4objective_v4_8gpu_mb96_probe24_kdfirst_20260812/checkpoints/01_hidden_mse}"
PIPELINE_ROOT="${PIPELINE_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/conversation_old_like_gt_pipeline_20260812}"
KD_INIT="${CONVERSATION_KD_INIT_CHECKPOINT:-$PIPELINE_ROOT/checkpoints/01_expansion_kd_init_e16_to_e24_step600}"
ORACLE="${CONVERSATION_ORACLE_CHECKPOINT:-$PIPELINE_ROOT/checkpoints/02_conversation_wikicode_replay_lm_oracle_step1800}"
FLAME_DATA_ROOT="${FLAME_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup}"
FLAME_ENV="${FLAME_ENV:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100}"
PYTHON_BIN="${PYTHON_BIN:-$FLAME_ENV/bin/python}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PLAN_ONLY="${PLAN_ONLY:-0}"

die() { echo "[ERROR] $*" >&2; exit 1; }
checkpoint_at() {
    local root="$1" step="$2"
    [[ -s "$root/latest_checkpointed_iteration.txt" ]] &&
        [[ "$(tr -d '[:space:]' < "$root/latest_checkpointed_iteration.txt")" == "$step" ]] &&
        [[ -s "$root/iter_$(printf '%07d' "$step")/common.pt" ]] &&
        [[ -s "$root/iter_$(printf '%07d' "$step")/.metadata" ]]
}
fresh_or_complete() {
    local root="$1" step="$2"
    checkpoint_at "$root" "$step" && return 0
    [[ ! -e "$root" ]] && return 0
    [[ -d "$root" && -z "$(find "$root" -mindepth 1 -maxdepth 1 -print -quit)" ]]
}

[[ -x "$PYTHON_BIN" ]] || die "FLAME Python missing: $PYTHON_BIN"
# The dataset-helper Makefile calls `python3`, independently of PYTHON_BIN.
export FLAME_ENV PYTHON_BIN PYTHONNOUSERSITE=1
export PATH="$FLAME_ENV/bin:/usr/bin:/bin"
[[ "$CUDA_VISIBLE_DEVICES" == "0,1,2,3,4,5,6,7" && "$NPROC_PER_NODE" == 8 ]] || die "this matched chain requires GPUs 0..7"
checkpoint_at "$SOURCE" 1800 || die "Code hidden-MSE source is not a complete step-1800 checkpoint: $SOURCE"
fresh_or_complete "$KD_INIT" 600 || die "refusing nonempty incomplete KD-init output: $KD_INIT"
fresh_or_complete "$ORACLE" 1800 || die "refusing nonempty incomplete oracle output: $ORACLE"
for task in wiki code conversation; do
    compgen -G "$FLAME_DATA_ROOT/$task/train/*.bin" >/dev/null || die "$task train data missing"
done

cat <<EOF
[PLAN] source Code model: $SOURCE (16E, hidden-MSE GT-token result, step 1800)
[PLAN] stage 1: expand 16E->24E, output-logit KD on equal Wiki+Code, 600 steps
[PLAN] stage 1 output: $KD_INIT
[PLAN] stage 2: Conversation LM + equal Wiki/Code replay LM, 1800 steps
[PLAN] stage 2 output: $ORACLE
[PLAN] oracle is for GT generation only; final experiments restart from stage 1
EOF
[[ "$PLAN_ONLY" == 1 ]] && exit 0

mkdir -p "$PIPELINE_ROOT/logs" "$PIPELINE_ROOT/scratch"
exec 9>"$PIPELINE_ROOT/.oracle_chain.lock"
flock -n 9 || die "oracle chain is already active"

if ! checkpoint_at "$KD_INIT" 600; then
    env \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" NPROC_PER_NODE="$NPROC_PER_NODE" MASTER_PORT=34301 \
        FLAME_ENV="$FLAME_ENV" PYTHON_BIN="$FLAME_ENV/bin/python" FLAME_DATA_ROOT="$FLAME_DATA_ROOT" \
        TOKENIZER_MODEL="$TOKENIZER_MODEL" WANDB_MODE=offline \
        SOURCE_WEIGHTS_DIR="$SOURCE" SOURCE_REQUIRED_ITERS=1800 \
        SOURCE_NUM_EXPERTS=16 NUM_EXPERTS=24 OLD_MODEL_KL_NUM_EXPERTS=16 \
        OLD_MODEL_KL_COEFF=1 OLD_MODEL_KL_TEMPERATURE=1 \
        TRAIN_ITERS=600 MICRO_BATCH_SIZE=32 GLOBAL_BATCH_SIZE=2304 \
        SAVE_INTERVAL=600 EVAL_INTERVAL=600 PROBE_EVAL_INTERVAL=100 \
        SECONDARY_PROBE_EVAL_INTERVAL=100 TERTIARY_PROBE_EVAL_INTERVAL=100 \
        RUN_ID=conversation-oldlike-expansion-kd-init-e16to24-step600 \
        TRAIN_WEIGHTS="$KD_INIT" LOCAL_BASE="$PIPELINE_ROOT/local" LOCAL_SSD_ROOT="$PIPELINE_ROOT/scratch/kd_init" \
        bash "$D/run_g2_ffn_only_conversation_expert_distill_init_wikicode_mha.sh" \
        >> "$PIPELINE_ROOT/logs/01_expansion_kd_init.log" 2>&1
    checkpoint_at "$KD_INIT" 600 || die "expansion KD-init exited without step 600"
fi

if ! checkpoint_at "$ORACLE" 1800; then
    env \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" NPROC_PER_NODE="$NPROC_PER_NODE" MASTER_PORT=34302 \
        FLAME_ENV="$FLAME_ENV" PYTHON_BIN="$FLAME_ENV/bin/python" FLAME_DATA_ROOT="$FLAME_DATA_ROOT" \
        TOKENIZER_MODEL="$TOKENIZER_MODEL" WANDB_MODE=offline \
        SOURCE_WEIGHTS_DIR="$KD_INIT" SOURCE_REQUIRED_ITERS=600 \
        TRAIN_ITERS=1800 MICRO_BATCH_SIZE=96 GLOBAL_BATCH_SIZE=2304 \
        SAVE_INTERVAL=1800 RECOVERY_SAVE_INTERVAL=300 EVAL_INTERVAL=600 LOG_INTERVAL=20 \
        PROBE_EVAL_INTERVAL=100 SECONDARY_PROBE_EVAL_INTERVAL=100 TERTIARY_PROBE_EVAL_INTERVAL=100 \
        PROBE_MICRO_BATCH_SIZE=24 PROBE_EVAL_ITERS=25 SECONDARY_PROBE_EVAL_ITERS=25 TERTIARY_PROBE_EVAL_ITERS=25 \
        RUN_INITIAL_PROBE_EVAL=1 RUN_INITIAL_VALID_EVAL=0 \
        MOE_JOINT_NEW_EXPERT_QUOTA=0 MOE_JOINT_NEW_EXPERT_QUOTA_SCHEDULE=200:0.5 \
        MOE_JOINT_NEW_EXPERT_QUOTA_MIN_NEW_SLOTS=1 MOE_JOINT_NEW_EXPERT_QUOTA_LOSS_COEFF=0.1 \
        MOE_JOINT_NEW_EXPERT_QUOTA_PRESERVE_NATURAL_GRADS=1 \
        NO_SAVE_OPTIM=1 DIRECT_LOCAL_SAVE=1 \
        RUN_ID=conversation-wikicode-replay-lm-oracle-step1800 \
        TRAIN_WEIGHTS="$ORACLE" LOCAL_BASE="$PIPELINE_ROOT/local" LOCAL_SSD_ROOT="$PIPELINE_ROOT/scratch/oracle" \
        bash "$D/run_g2_ffn_only_conversation_wikicode_joint_lm_allrouter_mha.sh" \
        >> "$PIPELINE_ROOT/logs/02_oracle_1800.log" 2>&1
    checkpoint_at "$ORACLE" 1800 || die "oracle exited without step 1800"
fi

echo "[DONE] reference=$KD_INIT current=$ORACLE"
