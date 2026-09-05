#!/usr/bin/env bash
set -euo pipefail

# 1-Phase Code training with CKA-selected contextual replay.
#
# Unlike the cosine-era token-only miniset, the replay dataset keeps each GT
# occurrence inside its original 512-token document-bounded window, and the
# packed GT mask restricts the replay loss to the selected positions.  Every
# other position is context only.  Primary Code LM and replay gradients
# accumulate inside one optimizer step; the replay branch keeps router
# gradients only.
#
#   TRAIN_ITERS=600 ./run_g2_cka_gt_contextual_replay_mha.sh

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$D/../../.." && pwd)"
cd "$REPO_ROOT"

MINISET_ROOT="${CKA_MINISET_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/cka_gt_contextual_miniset_bundle95_20260817}"
MINISET_DIR="${REPLAY_MINISET_DIR:-$MINISET_ROOT/train}"
# An empty GT path means every replayed position is supervised, which is how
# the random control arm runs: no selector, so no mask.
if [[ -n "${REPLAY_GT_PATH+x}" ]]; then GT_PATH="$REPLAY_GT_PATH"; else GT_PATH="$MINISET_ROOT/gt"; fi
# TASK selects the new task being learned. Code trains an E8->E16 expansion
# against the Wiki KD-init; Conversation trains an E16->E24 expansion against
# its own KD-init, which is also the teacher whose representations we preserve.
TASK="${TASK:-code}"
CODE_DIR="${CODE_DIR:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/code/train}"
CONVERSATION_DIR="${CONVERSATION_DIR:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/conversation/train}"
case "$TASK" in
    code)
        PRIMARY_DIR="$CODE_DIR"
        SOURCE="${SOURCE:-/data2/seonghyeonnoh/LLM-continual-learning-runs/flamemoe/ffn_experts_only/code/expansion_kd_init/kd_init/full_training/g2_olddata_kd_9run_20260808__code_e8_to_e16_wiki_kd_step600}"
        TEACHER_EXPERTS="${TEACHER_EXPERTS:-16}"
        ENTRY_PREFIX="run_g2_ffn_only_code_wiki_joint"
        ;;
    conversation)
        PRIMARY_DIR="$CONVERSATION_DIR"
        SOURCE="${SOURCE:?set SOURCE to the E16->E24 conversation KD-init checkpoint}"
        TEACHER_EXPERTS="${TEACHER_EXPERTS:-24}"
        ENTRY_PREFIX="run_g2_ffn_only_conversation_wikicode_joint"
        ;;
    *) echo "[ERROR] unknown TASK: $TASK" >&2; exit 1 ;;
esac
TEACHER_DIR="${TEACHER_DIR:-$SOURCE}"
SOURCE_REQUIRED_ITERS="${SOURCE_REQUIRED_ITERS:-600}"
# Probe test splits are resolved relative to this root, so it must point at the
# verified data tree rather than its parent; otherwise probe_data_path silently
# resolves to nothing and the run trains without producing any scores.
FLAME_DATA_ROOT="${FLAME_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6}"

STUDY_ROOT="${STUDY_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_gt_contextual_replay_20260817}"
LOCAL_ROOT="${LOCAL_ROOT:-$STUDY_ROOT/local_ssd}"
LOG_ROOT="${LOG_ROOT:-$STUDY_ROOT/logs}"

TRAIN_ITERS="${TRAIN_ITERS:-600}"
# On checkpoint continuation TRAIN_ITERS is the final global iteration, while
# REPLAY_BUDGET_ITERS is the newly executed interval. Keep them separate so a
# 600->1800 continuation does not inject an extra full 1800-step replay budget.
REPLAY_BUDGET_ITERS="${REPLAY_BUDGET_ITERS:-$TRAIN_ITERS}"
GBS="${GBS:-2304}"
SEQ="${SEQ:-512}"
MB="${MB:-96}"
REPLAY_MB="${REPLAY_MB:-48}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-36610}"
OBJECTIVE="${OBJECTIVE:-hidden_mse}"
HIDDEN_MSE_COEFF="${HIDDEN_MSE_COEFF:-10}"
LABEL="${LABEL:-cka-bundle95-contextual-${OBJECTIVE}-c${HIDDEN_MSE_COEFF}-${TRAIN_ITERS}step}"

# 20% of this run's own primary input budget, matching the 1-Phase convention.
PRIMARY_TOKENS=$((TRAIN_ITERS * GBS * SEQ))
REPLAY_BUDGET_TOKENS=$((REPLAY_BUDGET_ITERS * GBS * SEQ))
REPLAY_TOKENS=$((REPLAY_BUDGET_TOKENS / 5))
REPLAY_SAMPLES="${REPLAY_SAMPLES:-$((REPLAY_TOKENS / SEQ))}"
REPLAY_TOKENS=$((REPLAY_SAMPLES * SEQ))

MSE_ENTRY="$D/${ENTRY_PREFIX}_old_data_hidden_mse_allrouter_mha.sh"
HKL_ENTRY="$D/${ENTRY_PREFIX}_old_data_hidden_kl_allrouter_mha.sh"
VKL_ENTRY="$D/${ENTRY_PREFIX}_old_data_kd_allrouter_mha.sh"
LM_ENTRY="$D/${ENTRY_PREFIX}_lm_allrouter_mha.sh"

die() { echo "[ERROR] $*" >&2; exit 1; }

# The frozen teacher is the same step-600 source: hidden MSE is measured
# against the checkpoint the GT was selected against, so student and teacher
# share the layer indexing the census used (residual-included L2..L9).
TEACHER=(ENABLE_OLD_MODEL_KL=1 OLD_MODEL_KL_WEIGHTS_DIR="$TEACHER_DIR" OLD_MODEL_KL_NUM_EXPERTS="$TEACHER_EXPERTS")
case "$OBJECTIVE" in
    hidden_mse) ENTRY="$MSE_ENTRY"; OBJ_ENV=("${TEACHER[@]}" OLD_MODEL_KL_COEFF=0
                    MOE_JOINT_REPLAY_LM=1 MOE_JOINT_REPLAY_OLD_DATA_KD=0
                    MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=0 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=1
                    OLD_HIDDEN_MSE_COEFF="$HIDDEN_MSE_COEFF" OLD_HIDDEN_MSE_LAYERS=2,3,4,5,6,7,8,9) ;;
    hidden_kl)  ENTRY="$HKL_ENTRY"; OBJ_ENV=("${TEACHER[@]}" OLD_MODEL_KL_COEFF=0
                    MOE_JOINT_REPLAY_LM=1 MOE_JOINT_REPLAY_OLD_DATA_KD=0
                    MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=1 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=0) ;;
    vocab_kl)   ENTRY="$VKL_ENTRY"; OBJ_ENV=("${TEACHER[@]}" OLD_MODEL_KL_COEFF=1
                    MOE_JOINT_REPLAY_LM=1 MOE_JOINT_REPLAY_OLD_DATA_KD=1
                    MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=0 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=0) ;;
    lm)         ENTRY="$LM_ENTRY";  OBJ_ENV=(ENABLE_OLD_MODEL_KL=0 OLD_MODEL_KL_COEFF=0
                    MOE_JOINT_REPLAY_LM=1 MOE_JOINT_REPLAY_OLD_DATA_KD=0
                    MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=0 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=0) ;;
    *) die "unknown OBJECTIVE: $OBJECTIVE" ;;
esac

preflight() {
    local required=("$MINISET_DIR/train_text_document.bin" "$MINISET_DIR/train_text_document.idx"
        "$MINISET_DIR/metadata.json" "$SOURCE/latest_checkpointed_iteration.txt" "$ENTRY")
    if [[ -n "$GT_PATH" ]]; then
        required+=("$GT_PATH/metadata.json" "$GT_PATH/rank_000/old_like_gt_packed.npy")
    fi
    for path in "${required[@]}"; do
        [[ -e "$path" ]] || die "missing required path: $path"
    done
    if [[ -z "$GT_PATH" ]]; then
        RANDOM_META="$MINISET_DIR/metadata.json" SEQ_CHECK="$SEQ" \
            python3 "$REPO_ROOT/scripts/analysis/check_random_replay_miniset.py" \
            || die "random miniset preflight failed"
        return
    fi
    # The miniset must be an exact multiple of the sequence length so that
    # sample i is source window i; a ragged stream would silently shift every
    # mask row against its context.
    python3 - "$MINISET_DIR/metadata.json" "$GT_PATH/metadata.json" "$SEQ" <<'PY'
import json, sys
mini = json.load(open(sys.argv[1]))
gt = json.load(open(sys.argv[2]))
seq = int(sys.argv[3])
ALLOWED = {"cka_gt_contextual_window_miniset_v1", "matched_random_gt_window_miniset_v1", "cka_gt_token_pack_miniset_v1"}
assert mini["schema"] in ALLOWED, f"unexpected miniset schema {mini['schema']!r}"
assert mini["schema"] == gt["schema"], "miniset and mask were built by different selectors"
assert mini["token_identity_mismatches"] == 0, "miniset token identity was not verified"
if mini["schema"] == "cka_gt_token_pack_miniset_v1":
    assert mini["context_preserved"] is False and mini["tokens_concatenated_out_of_context"] is True
else:
    assert mini["context_preserved"] is True and mini["tokens_concatenated_out_of_context"] is False
assert mini["sequence_length"] == gt["sequence_length"] == seq, "sequence length mismatch"
assert mini["window_count"] == gt["total_samples"], "window count disagrees with GT total_samples"
assert mini["miniset_tokens"] == mini["window_count"] * seq, "miniset is not window-aligned"
assert mini["packed_popcount"] == mini["gt_occurrences"], "packed popcount disagrees"
selector = ("CKA-token-pack" if mini["schema"] == "cka_gt_token_pack_miniset_v1" else "CKA") if mini["schema"].startswith("cka") else "random"
print(f"[PREFLIGHT] selector={selector} windows={mini['window_count']} "
      f"occurrences={mini['gt_occurrences']} density={mini['gt_density']:.6f} "
      f"documents={mini.get('document_count', mini.get('documents'))}")
PY
    (( PRIMARY_TOKENS / 5 * 5 == PRIMARY_TOKENS )) || die "primary token budget is not divisible by 5"
    (( REPLAY_SAMPLES * SEQ == REPLAY_TOKENS )) || die "replay sample/token budget mismatch"
}

preflight

# Both arms report a window pool and a supervised-position count; the random
# miniset simply supervises all of them.
WINDOWS=$(python3 -c "import json;m=json.load(open('$MINISET_DIR/metadata.json'));print(m.get('window_count') or m['sampled_windows'])")
OCC=$(python3 -c "import json;m=json.load(open('$MINISET_DIR/metadata.json'));print(m.get('gt_occurrences') or m['miniset_tokens'])")
POOL_EPOCHS=$(python3 -c "print(f'{$REPLAY_SAMPLES/$WINDOWS:.1f}')")
SUPERVISED=$(python3 -c "print(int($REPLAY_SAMPLES/$WINDOWS*$OCC))")
OUTPUT="$STUDY_ROOT/$LABEL"

cat <<PLAN
[PLAN] label            : $LABEL
[PLAN] task             : $TASK  (primary $PRIMARY_DIR)
[PLAN] source           : $SOURCE (iter $SOURCE_REQUIRED_ITERS)
[PLAN] teacher          : $TEACHER_DIR (${TEACHER_EXPERTS}E)
[PLAN] primary          : full $TASK LM, $TRAIN_ITERS steps, GBS=$GBS, SEQ=$SEQ
[PLAN] replay dataset   : $MINISET_DIR
[PLAN]                    $WINDOWS original 512-token windows, context preserved
[PLAN] replay mask      : $GT_PATH  ($OCC GT positions, the rest is context only)
[PLAN] replay budget    : $REPLAY_SAMPLES sequences = $REPLAY_TOKENS tokens = 20% of $REPLAY_BUDGET_ITERS newly executed steps
[PLAN] pool repetition  : $POOL_EPOCHS passes over the window pool
[PLAN] supervised posns : ~$SUPERVISED GT presentations
[PLAN] objective        : $OBJECTIVE (hidden MSE coeff $HIDDEN_MSE_COEFF applies to hidden_mse only)
[PLAN] accumulation     : primary then replay microbatches, one optimizer.step per update
[PLAN] replay gradients : router only; natural routing, no forced routing
[PLAN] GPUs             : $CUDA_VISIBLE_DEVICES (nproc=$NPROC_PER_NODE), primary MB=$MB, replay MB=$REPLAY_MB
[PLAN] output           : $OUTPUT
PLAN

if [[ "${PLAN_ONLY:-0}" == 1 ]]; then
    exit 0
fi

mkdir -p "$OUTPUT" "$LOG_ROOT"
echo "$(date -Is) START $LABEL $OUTPUT" | tee -a "$LOG_ROOT/status.tsv"
env \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" NPROC_PER_NODE="$NPROC_PER_NODE" MASTER_PORT="$MASTER_PORT" \
    FLAME_ENV=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100 \
    PYTHON_BIN=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin/python \
    PATH=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin:/usr/bin:/bin \
    CUDA_HOME=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100 \
    PYTHONNOUSERSITE=1 \
    FLAME_DATA_ROOT="$FLAME_DATA_ROOT" TOKENIZER_MODEL="$TOKENIZER_MODEL" WANDB_MODE=offline \
    SOURCE_WEIGHTS_DIR="$SOURCE" SOURCE_REQUIRED_ITERS="$SOURCE_REQUIRED_ITERS" \
    TRAIN_DATASET="$PRIMARY_DIR" JOINT_REPLAY_DATASET="$MINISET_DIR" \
    JOINT_REPLAY_SECONDARY_DATASET= \
    JOINT_REPLAY_DATA_WEIGHT_MODE=equal_dataset \
    MOE_JOINT_REPLAY_OLD_LIKE_GT_PATH="$GT_PATH" \
    MOE_JOINT_REPLAY_TOTAL_SAMPLES="$REPLAY_SAMPLES" \
    MOE_JOINT_REPLAY_MICRO_BATCH_SIZE="$REPLAY_MB" \
    TRAIN_ITERS="$TRAIN_ITERS" REPLAY_BUDGET_ITERS="$REPLAY_BUDGET_ITERS" MICRO_BATCH_SIZE="$MB" GLOBAL_BATCH_SIZE="$GBS" SEQ_LENGTH="$SEQ" \
    DATASET_SPLIT=100,0,0 LR=3e-4 MIN_LR=3e-5 LR_DECAY_STYLE=WSD LR_DECAY_ITERS="$TRAIN_ITERS" \
    LR_WSD_DECAY_ITERS=60 LR_WARMUP_FRACTION=0.01 SAVE_INTERVAL="$TRAIN_ITERS" RECOVERY_SAVE_INTERVAL=200 \
    EVAL_INTERVAL="$TRAIN_ITERS" LOG_INTERVAL=20 PROBE_EVAL_INTERVAL=100 SECONDARY_PROBE_EVAL_INTERVAL=100 \
    PROBE_MICRO_BATCH_SIZE=24 PROBE_EVAL_ITERS=25 SECONDARY_PROBE_EVAL_ITERS=25 \
    TERTIARY_PROBE_EVAL_INTERVAL=100 TERTIARY_PROBE_EVAL_ITERS=25 \
    RUN_INITIAL_PROBE_EVAL=1 RUN_INITIAL_VALID_EVAL=0 \
    MOE_JOINT_NEW_EXPERT_QUOTA=0 MOE_JOINT_NEW_EXPERT_QUOTA_SCHEDULE="${QUOTA_SCHEDULE-200:0.5}" \
    MOE_JOINT_NEW_EXPERT_QUOTA_MIN_NEW_SLOTS=1 MOE_JOINT_NEW_EXPERT_QUOTA_LOSS_COEFF=0.1 \
    MOE_JOINT_NEW_EXPERT_QUOTA_PRESERVE_NATURAL_GRADS=1 \
    MOE_SEPARATE_ROUTER_EXPERT_GRAD_CLIP=1 MOE_ALLOW_PARTIAL_OPTIMIZER_STATE=0 \
    MOE_NEW_EXPERT_LR_RAMP_STEPS=0 MOE_ROUTER_LR_MULTIPLIER=1.0 LOG_ROUTER_GRAD_NORM_SOURCES=0 \
    MOE_AUX_LOSS_COEFF=0.01 MOE_Z_LOSS_COEFF=0.001 NO_SAVE_OPTIM=1 DIRECT_LOCAL_SAVE=1 \
    LOCAL_BASE="$STUDY_ROOT/local" LOCAL_SSD_ROOT="$LOCAL_ROOT" \
    RUN_ID="$LABEL" TRAIN_WEIGHTS="$OUTPUT" WANDB_RUN_ID="$LABEL" WANDB_EXP_NAME="$LABEL" \
    "${OBJ_ENV[@]}" bash "$ENTRY"
echo "$(date -Is) DONE $LABEL $OUTPUT" | tee -a "$LOG_ROOT/status.tsv"
