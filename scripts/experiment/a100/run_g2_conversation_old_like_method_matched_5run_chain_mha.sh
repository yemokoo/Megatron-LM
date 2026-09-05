#!/usr/bin/env bash
set -euo pipefail

# Five-run Conversation comparison:
#   1) the already-completed full Wiki+Code replay LM oracle;
#   2--5) contextual-occurrence old-like replay MSE / hidden-KL / vocab-KL / LM.
# Each old-like run follows its matching Code objective checkpoint through an
# independent 16E->24E expansion output-KD init. Primary Conversation data and
# the GT-occurrence replay subset are always separate logical datasets and
# iterators. Replay reads the original context but supervises exactly one GT
# hidden position and its original shifted next-token label per item.

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="${PIPELINE_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/conversation_old_like_gt_pipeline_20260812}"
CODE_ROOT="${CODE_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/code_old_like_gt_token_miniset_20pct_1phase_4objective_v4_8gpu_mb96_probe24_kdfirst_20260812}"
ORACLE="${CONVERSATION_ORACLE_CHECKPOINT:-$PIPELINE_ROOT/checkpoints/02_conversation_wikicode_replay_lm_oracle_step1800}"
MSE_KD_INIT="${CONVERSATION_MSE_KD_INIT:-$PIPELINE_ROOT/checkpoints/01_expansion_kd_init_e16_to_e24_step600}"
STUDY_ROOT="${CONVERSATION_STUDY_ROOT:-$PIPELINE_ROOT/final_5run_20pct_contextual_occurrence_exact_axis_method_matched_v8_all_steps3600}"
CONVERSATION_DIR="${CONVERSATION_TRAIN_DIR:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/conversation/train}"
FLAME_DATA_ROOT="${FLAME_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup}"
GT_ROOT="${CONVERSATION_OLD_LIKE_GT_ROOT:-$PIPELINE_ROOT/analysis/conversation_old_like_gt_l2_l9_top1}"
PAIRED_ROOT="${CONVERSATION_PAIRED_ROOT:-$PIPELINE_ROOT/analysis/conversation_token_hidden_pair}"
PAIRED_CACHE="${CONVERSATION_PAIRED_CACHE:-$PAIRED_ROOT/data_cache_omp8_noworkers/rank_000}"
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
WANDB_RUN_MODE="${WANDB_RUN_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
WANDB_GLOBAL_STEP_OFFSET="${WANDB_GLOBAL_STEP_OFFSET:-3600}"
METHODS_CSV="${METHODS_CSV:-mse,hkl,vkl,lm}"
RUN_TAG="${RUN_TAG:-v8}"
IFS=',' read -r -a METHODS <<< "$METHODS_CSV"

# Child launchers source a site runtime that may prepend the base conda Python.
# Pin both explicit Python variables and PATH, matching the verified oracle
# chain, so Megatron uses the existing CPython-3.10 dataset helper.
export FLAME_ENV PYTHON_BIN PYTHONNOUSERSITE=1
export PATH="$FLAME_ENV/bin:/usr/bin:/bin"

declare -A CODE_SOURCE=(
    [mse]="$CODE_ROOT/checkpoints/01_hidden_mse"
    [hkl]="$CODE_ROOT/checkpoints/02_hidden_kl"
    [vkl]="$CODE_ROOT/checkpoints/03_vocab_kl"
    [lm]="$CODE_ROOT/checkpoints/04_lm"
)
declare -A KD_INIT=(
    [mse]="$MSE_KD_INIT"
    [hkl]="$STUDY_ROOT/kd_init/02_hidden_kl_e16_to_e24_step600"
    [vkl]="$STUDY_ROOT/kd_init/03_vocab_kl_e16_to_e24_step600"
    [lm]="$STUDY_ROOT/kd_init/04_lm_e16_to_e24_step600"
)

LM_ENTRY="$D/run_g2_ffn_only_conversation_wikicode_joint_lm_allrouter_mha.sh"
MSE_ENTRY="$D/run_g2_ffn_only_conversation_wikicode_joint_old_data_hidden_mse_allrouter_mha.sh"
HKL_ENTRY="$D/run_g2_ffn_only_conversation_wikicode_joint_old_data_hidden_kl_allrouter_mha.sh"
VKL_ENTRY="$D/run_g2_ffn_only_conversation_wikicode_joint_old_data_kd_allrouter_mha.sh"
EXPAND_ENTRY="$D/run_g2_ffn_only_conversation_expert_distill_init_wikicode_mha.sh"

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

[[ -x "$PYTHON_BIN" ]] || die "Python missing: $PYTHON_BIN"
[[ "$CUDA_VISIBLE_DEVICES" == "0,1,2,3,4,5,6,7" && "$NPROC_PER_NODE" == 8 ]] || die "requires GPUs 0..7"
checkpoint_at "$ORACLE" 1800 || die "completed full-replay oracle missing: $ORACLE"
(( ${#METHODS[@]} > 0 )) || die "METHODS_CSV must select at least one objective"
for method in "${METHODS[@]}"; do
    [[ "$method" =~ ^(mse|hkl|vkl|lm)$ ]] || die "unsupported objective in METHODS_CSV: $method"
    checkpoint_at "${CODE_SOURCE[$method]}" 1800 || die "$method Code checkpoint incomplete: ${CODE_SOURCE[$method]}"
    fresh_or_complete "${KD_INIT[$method]}" 600 || die "refusing nonempty incomplete $method KD-init: ${KD_INIT[$method]}"
done
"$PYTHON_BIN" - "$GT_ROOT/metadata.json" "$GT_ROOT/replay_occurrence_metadata.json" "$REPLAY_SAMPLES" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).parent; gt=json.load(open(sys.argv[1])); occ=json.load(open(sys.argv[2])); replay=int(sys.argv[3])
assert gt['complete'] and gt['schema']=='old_like_gt_l2_l9_all_task_top1_raw_v1'
assert gt['total_samples']==2_740_224 and gt['sequence_length']==512
assert gt['selected_count']==1_311_043
assert occ['complete'] and occ['schema']=='old_like_gt_token_occurrence_subset_v1'
assert occ['source_gt_schema']==gt['schema'] and occ['occurrence_count']==gt['selected_count']
assert occ['virtual_samples_per_1800_step_run']==replay
for key,name in [('sample_sha256',occ['sample_file']),('position_sha256',occ['position_file'])]:
    h=hashlib.sha256((root/name).read_bytes()).hexdigest(); assert h==occ[key],(key,h,occ[key])
print('[VALID] contextual occurrences={:,}; replay samples={:,}; occurrence epochs={:.6f}'.format(
    occ['occurrence_count'],replay,replay/occ['occurrence_count']))
PY

# A physical-shard checksum is insufficient: equal-weight and exhaustive
# blends can contain identical files while assigning different sequences to
# the same outer sample_id.  Compare actual token and shifted-label arrays to
# the paired extraction before allocating any GPU model.
(cd "$D/../../.." && PYTHONPATH=Megatron-LM "$PYTHON_BIN" \
    scripts/analysis/validate_old_like_gt_source_alignment.py \
    --gt-root "$GT_ROOT" \
    --paired-root "$PAIRED_ROOT" \
    --data-cache "$PAIRED_CACHE" \
    --tokenizer-model "$TOKENIZER_MODEL" \
    --checks 32)

declare -A METHOD_OUTPUT=(
    [mse]="$STUDY_ROOT/checkpoints/02_oldlike_hidden_mse"
    [hkl]="$STUDY_ROOT/checkpoints/03_oldlike_hidden_kl"
    [vkl]="$STUDY_ROOT/checkpoints/04_oldlike_vocab_kl"
    [lm]="$STUDY_ROOT/checkpoints/05_oldlike_lm"
)
for method in "${METHODS[@]}"; do
    fresh_or_complete "${METHOD_OUTPUT[$method]}" 1800 || die "refusing nonempty incomplete output: ${METHOD_OUTPUT[$method]}"
done

cat <<EOF
[PLAN] selected method-matched objectives: $METHODS_CSV
  01 oracle (complete): $ORACLE
[PLAN] primary dataset: $CONVERSATION_DIR
[PLAN] replay selector: $GT_ROOT (contextual occurrence, separate iterator)
[PLAN] replay exposure: $REPLAY_SAMPLES original-context sequences / $REPLAY_TOKENS input tokens = 20%
[PLAN] replay loss: exactly one GT position per replay item; original next-token label retained
[PLAN] Weights & Biases: mode=$WANDB_RUN_MODE project=$WANDB_PROJECT local 0..1800 -> global $WANDB_GLOBAL_STEP_OFFSET..$((WANDB_GLOBAL_STEP_OFFSET + TRAIN_ITERS))
EOF
[[ "$PLAN_ONLY" == 1 ]] && exit 0

mkdir -p "$STUDY_ROOT"/{checkpoints,kd_init,logs,scratch,local}
exec 9>"$STUDY_ROOT/.chain.lock"
flock -n 9 || die "method-matched five-run chain is already active"
for gpu in 0 1 2 3 4 5 6 7; do
    apps="$(nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d;/No running processes found/d')"
    [[ -z "$apps" ]] || die "GPU $gpu is busy: $apps"
done

write_manifest() {
    "$PYTHON_BIN" - "$STUDY_ROOT/run_manifest.json" "$ORACLE" "$CONVERSATION_DIR" "$GT_ROOT" "$FULL_TOKENS" "$REPLAY_TOKENS" "$REPLAY_SAMPLES" "$METHODS_CSV" "$RUN_TAG" <<'PY'
import json,sys,datetime,os,tempfile
out,oracle,primary,gt_root,full,replay_tokens,replay_samples,methods,run_tag=sys.argv[1:]
methods=methods.split(',')
payload={'schema':'conversation_old_like_contextual_occurrence_method_matched_5run_v2','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
'run_count':len(methods),'selected_objectives':methods,'run_tag':run_tag,'comparison_oracle':oracle,'primary_dataset':primary,'replay_gt_root':gt_root,
'primary_replay_iterators_are_separate':True,'replay_unit':'token_occurrence',
'replay_sample_axis':'paired_extraction_exhaustive_blend',
'primary_sample_axis':'ordinary_equal_weight_task_blend',
'non_gt_replay_positions_are_context_only':True,'original_next_token_label_retained':True,
'full_train_tokens':int(full),'replay_input_tokens':int(replay_tokens),'replay_samples':int(replay_samples)}
fd,tmp=tempfile.mkstemp(dir=os.path.dirname(out),prefix='.run_manifest.',suffix='.tmp')
with os.fdopen(fd,'w') as f: json.dump(payload,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
PY
}
write_manifest

expand_if_needed() {
    local method="$1" port="$2" source="${CODE_SOURCE[$1]}" output="${KD_INIT[$1]}"
    checkpoint_at "$output" 600 && { echo "[SKIP] $method KD-init complete"; return; }
    echo "$(date -Is) START ${method}_expansion_kd_init" | tee -a "$STUDY_ROOT/logs/status.tsv"
    env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" NPROC_PER_NODE="$NPROC_PER_NODE" MASTER_PORT="$port" \
        FLAME_ENV="$FLAME_ENV" PYTHON_BIN="$PYTHON_BIN" FLAME_DATA_ROOT="$FLAME_DATA_ROOT" TOKENIZER_MODEL="$TOKENIZER_MODEL" WANDB_MODE="$WANDB_RUN_MODE" \
        SOURCE_WEIGHTS_DIR="$source" SOURCE_REQUIRED_ITERS=1800 SOURCE_NUM_EXPERTS=16 NUM_EXPERTS=24 OLD_MODEL_KL_NUM_EXPERTS=16 \
        OLD_MODEL_KL_COEFF=1 OLD_MODEL_KL_TEMPERATURE=1 TRAIN_ITERS=600 MICRO_BATCH_SIZE=32 GLOBAL_BATCH_SIZE=2304 \
        SAVE_INTERVAL=600 EVAL_INTERVAL=600 PROBE_EVAL_INTERVAL=100 SECONDARY_PROBE_EVAL_INTERVAL=100 TERTIARY_PROBE_EVAL_INTERVAL=100 \
        RUN_ID="conversation-oldlike-${method}-expansion-kd-init-e16to24" TRAIN_WEIGHTS="$output" \
        LOCAL_BASE="$STUDY_ROOT/local" LOCAL_SSD_ROOT="$STUDY_ROOT/scratch/${method}_kd_init" \
        bash "$EXPAND_ENTRY" >> "$STUDY_ROOT/logs/${method}_expansion_kd_init.log" 2>&1
    checkpoint_at "$output" 600 || die "$method expansion KD-init exited without step 600"
    echo "$(date -Is) DONE ${method}_expansion_kd_init" | tee -a "$STUDY_ROOT/logs/status.tsv"
}

train_method() {
    local method="$1" label="$2" output="$3" port="$4" entry="$5" source="${KD_INIT[$1]}"
    shift 5
    checkpoint_at "$output" 1800 && { echo "[SKIP] $label complete"; return; }
    mkdir -p "$output"
    echo "$(date -Is) START $label" | tee -a "$STUDY_ROOT/logs/status.tsv"
    env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" NPROC_PER_NODE="$NPROC_PER_NODE" MASTER_PORT="$port" \
        FLAME_ENV="$FLAME_ENV" PYTHON_BIN="$PYTHON_BIN" FLAME_DATA_ROOT="$FLAME_DATA_ROOT" TOKENIZER_MODEL="$TOKENIZER_MODEL" WANDB_MODE="$WANDB_RUN_MODE" \
        SOURCE_WEIGHTS_DIR="$source" SOURCE_REQUIRED_ITERS=600 OLD_MODEL_KL_WEIGHTS_DIR="$source" OLD_MODEL_KL_NUM_EXPERTS=24 \
        TRAIN_DATASET="$CONVERSATION_DIR" JOINT_REPLAY_DATASET="$CONVERSATION_DIR" JOINT_REPLAY_SECONDARY_DATASET= \
        JOINT_REPLAY_DATA_WEIGHT_MODE=equal_dataset MOE_JOINT_REPLAY_OLD_LIKE_GT_PATH="$GT_ROOT" \
        MOE_JOINT_REPLAY_OLD_LIKE_UNIT=token_occurrence \
        MOE_JOINT_REPLAY_OLD_LIKE_SELECTED_TOKEN_COUNT=1311043 \
        MOE_JOINT_REPLAY_OLD_LIKE_FULL_TRAIN_TOKEN_COUNT="$FULL_TOKENS" \
        OLD_LIKE_REPLAY_SUBSET_COUNT=1311043 OLD_LIKE_REPLAY_SUBSET_SHA256=fbff892045f8f1ee5ca90f7648a8352945de98941726c32cf131e65f3ab083ee \
        MOE_JOINT_REPLAY_TOTAL_SAMPLES="$REPLAY_SAMPLES" MOE_JOINT_REPLAY_MICRO_BATCH_SIZE="$REPLAY_MB" \
        TRAIN_ITERS="$TRAIN_ITERS" MICRO_BATCH_SIZE="$PRIMARY_MB" GLOBAL_BATCH_SIZE="$GBS" SEQ_LENGTH="$SEQ" \
        LR=3e-4 MIN_LR=3e-5 LR_DECAY_STYLE=WSD LR_DECAY_ITERS=1800 LR_WSD_DECAY_ITERS=180 LR_WARMUP_FRACTION=0.01 \
        SAVE_INTERVAL=1800 RECOVERY_SAVE_INTERVAL=300 EVAL_INTERVAL=600 LOG_INTERVAL=20 \
        PROBE_EVAL_INTERVAL=100 SECONDARY_PROBE_EVAL_INTERVAL=100 TERTIARY_PROBE_EVAL_INTERVAL=100 \
        PROBE_MICRO_BATCH_SIZE=24 PROBE_EVAL_ITERS=25 SECONDARY_PROBE_EVAL_ITERS=25 TERTIARY_PROBE_EVAL_ITERS=25 \
        PROBE_STEP_OFFSET="$WANDB_GLOBAL_STEP_OFFSET" SECONDARY_PROBE_STEP_OFFSET="$WANDB_GLOBAL_STEP_OFFSET" TERTIARY_PROBE_STEP_OFFSET="$WANDB_GLOBAL_STEP_OFFSET" \
        RUN_INITIAL_PROBE_EVAL=1 RUN_INITIAL_VALID_EVAL=0 \
        WANDB_STEP_OFFSET="$WANDB_GLOBAL_STEP_OFFSET" \
        MOE_JOINT_NEW_EXPERT_QUOTA=0 MOE_JOINT_NEW_EXPERT_QUOTA_SCHEDULE=200:0.5 MOE_JOINT_NEW_EXPERT_QUOTA_MIN_NEW_SLOTS=1 \
        MOE_JOINT_NEW_EXPERT_QUOTA_LOSS_COEFF=0.1 MOE_JOINT_NEW_EXPERT_QUOTA_PRESERVE_NATURAL_GRADS=1 \
        MOE_NEW_EXPERT_LR_RAMP_STEPS=0 MOE_ROUTER_LR_MULTIPLIER=1 LOG_ROUTER_GRAD_NORM_SOURCES=0 \
        NO_SAVE_OPTIM=1 DIRECT_LOCAL_SAVE=1 LOCAL_BASE="$STUDY_ROOT/local" LOCAL_SSD_ROOT="$STUDY_ROOT/scratch/$label" \
        RUN_ID="$label" TRAIN_WEIGHTS="$output" WANDB_RUN_ID="$label" WANDB_EXP_NAME="$label" "$@" \
        bash "$entry" >> "$STUDY_ROOT/logs/$label.log" 2>&1
    checkpoint_at "$output" 1800 || die "$label exited without step 1800"
    echo "$(date -Is) DONE $label" | tee -a "$STUDY_ROOT/logs/status.tsv"
}

for method in "${METHODS[@]}"; do
    case "$method" in
        mse)
            train_method mse "conversation_oldlike_methodmatched_hidden_mse_c10_l2to9_exactaxis_${RUN_TAG}" "${METHOD_OUTPUT[mse]}" 34502 "$MSE_ENTRY" \
                ENABLE_OLD_MODEL_KL=1 OLD_MODEL_KL_COEFF=0 MOE_JOINT_REPLAY_LM=1 MOE_JOINT_REPLAY_OLD_DATA_KD=0 \
                MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=0 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=1 OLD_HIDDEN_MSE_COEFF=10 OLD_HIDDEN_MSE_LAYERS=2,3,4,5,6,7,8,9
            ;;
        hkl)
            expand_if_needed hkl 34513
            train_method hkl "conversation_oldlike_methodmatched_hidden_kl_c1_l2to9_exactaxis_${RUN_TAG}" "${METHOD_OUTPUT[hkl]}" 34503 "$HKL_ENTRY" \
                ENABLE_OLD_MODEL_KL=1 OLD_MODEL_KL_COEFF=0 MOE_JOINT_REPLAY_LM=1 MOE_JOINT_REPLAY_OLD_DATA_KD=0 \
                MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=1 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=0 OLD_HIDDEN_KL_COEFF=1 OLD_HIDDEN_KL_TEMPERATURE=1 OLD_HIDDEN_KL_LAYERS=2,3,4,5,6,7,8,9
            ;;
        vkl)
            expand_if_needed vkl 34514
            train_method vkl "conversation_oldlike_methodmatched_vocab_kl_c1_exactaxis_${RUN_TAG}" "${METHOD_OUTPUT[vkl]}" 34504 "$VKL_ENTRY" \
                ENABLE_OLD_MODEL_KL=1 OLD_MODEL_KL_COEFF=1 OLD_MODEL_KL_TEMPERATURE=1 MOE_JOINT_REPLAY_LM=1 \
                MOE_JOINT_REPLAY_OLD_DATA_KD=1 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=0 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=0
            ;;
        lm)
            expand_if_needed lm 34515
            train_method lm "conversation_oldlike_methodmatched_lm_exactaxis_${RUN_TAG}" "${METHOD_OUTPUT[lm]}" 34505 "$LM_ENTRY" \
                ENABLE_OLD_MODEL_KL=0 OLD_MODEL_KL_COEFF=0 MOE_JOINT_REPLAY_LM=1 MOE_JOINT_REPLAY_OLD_DATA_KD=0 \
                MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=0 MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=0
            ;;
    esac
done

echo "$(date -Is) COMPLETE selected method-matched objectives: $METHODS_CSV" | tee -a "$STUDY_ROOT/logs/status.tsv"
