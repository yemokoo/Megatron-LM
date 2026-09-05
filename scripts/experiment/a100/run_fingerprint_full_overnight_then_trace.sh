#!/usr/bin/env bash
set -euo pipefail

# Resumable overnight chain:
# B -> Code-only 1800 -> Code+Wiki router-only LM 1800
#   -> Conversation expansion KD-init 600 -> Conversation-only 1800
#   -> Wiki+Code+Conversation router-only LM 1800 -> interrupted TRACE eval.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TRACE_ROOT="$REPO_ROOT/trace"

FLAME_ENV="${FLAME_ENV:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100}"
[[ -x "$FLAME_ENV/bin/python" ]] || { echo "[ERROR] H100 FLAME environment missing: $FLAME_ENV" >&2; exit 1; }
export PATH="$FLAME_ENV/bin:/usr/bin:/bin"
export PYTHON_BIN="$FLAME_ENV/bin/python"
export PYTHONNOUSERSITE=1
export CUDA_HOME="${CUDA_HOME:-$FLAME_ENV}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

STUDY_ROOT="${STUDY_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/fingerprint_router_geometry_20260809}"
FLAME_DATA_ROOT="${FLAME_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup}"
B_WEIGHTS="${B_WEIGHTS:-/data2/seonghyeonnoh/LLM-continual-learning-runs/g2_olddata_kd_9run_20260808/01_common_kd_init/code_e8_to_e16_wiki_kd_step600}"
D_WEIGHTS="${D_WEIGHTS:-$STUDY_ROOT/checkpoints/D_B_to_Code_only_no_olddata_mb48_gbs2304_step1800}"
CODE_ROUTER_WEIGHTS="${CODE_ROUTER_WEIGHTS:-$STUDY_ROOT/checkpoints/CodeWiki_router_only_LM_1800_from_D}"
CONV_KD_WEIGHTS="${CONV_KD_WEIGHTS:-$STUDY_ROOT/checkpoints/Conversation_expand_WikiCode_logits_KD_mb36_step600}"
CONV_WEIGHTS="${CONV_WEIGHTS:-$STUDY_ROOT/checkpoints/Conversation_only_from_KDinit_mb48_gbs2304_step1800}"
FINAL_ROUTER_WEIGHTS="${FINAL_ROUTER_WEIGHTS:-$STUDY_ROOT/checkpoints/WikiCodeConversation_router_only_LM_1800}"
LOG_ROOT="${LOG_ROOT:-$STUDY_ROOT/logs/overnight_full_chain}"
LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-$STUDY_ROOT/scratch}"

# This is the TRACE sparse-15 run that was explicitly interrupted earlier.
TRACE_RUN="${TRACE_RUN:-/data2/seonghyeonnoh/LLM-continual-learning-runs/v2_new_top4_kd2x_20260808}"
TRACE_METHOD="${TRACE_METHOD:-ours_lora_moe_v2_new_top4}"

export STUDY_ROOT FLAME_DATA_ROOT B_WEIGHTS D_WEIGHTS CONV_KD_WEIGHTS CONV_WEIGHTS LOCAL_SSD_ROOT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

mkdir -p "$LOG_ROOT" "$STUDY_ROOT/checkpoints"
STATUS_FILE="$LOG_ROOT/status.tsv"
PID_FILE="$LOG_ROOT/chain.pid"
echo "$$" > "$PID_FILE"

record() {
    printf '%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" >> "$STATUS_FILE"
    echo "[$1] $2"
}

hold_gpus_after_error() {
    local rc=$?
    trap - EXIT
    if [[ "$rc" -ne 0 ]]; then
        record GPU_HOLD "chain exited with rc=$rc; starting gpu_hold.py on GPUs 0-7"
        exec "$FLAME_ENV/bin/python" /home/seonghyeonnoh/yemokoo/gpu_hold.py --gpus 0,1,2,3,4,5,6,7
    fi
}
trap hold_gpus_after_error EXIT

checkpoint_at() {
    local root="$1" expected="$2" tracker="$1/latest_checkpointed_iteration.txt"
    [[ -f "$tracker" ]] && [[ "$(tr -d '[:space:]' < "$tracker")" == "$expected" ]]
}

run_stage() {
    local label="$1" output="$2" expected="$3" log="$4"
    shift 4
    if checkpoint_at "$output" "$expected"; then
        record SKIP "$label already complete at step $expected: $output"
        return
    fi
    record START "$label -> expected step $expected"
    set +e
    "$@" 2>&1 | tee -a "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    if [[ "$rc" -ne 0 ]] || ! checkpoint_at "$output" "$expected"; then
        record ERROR "$label failed rc=$rc output=$output log=$log"
        exit "$([[ "$rc" -eq 0 ]] && echo 1 || echo "$rc")"
    fi
    record DONE "$label reached step $expected: $output"
}

prepare_router_copy() {
    local source="$1" target="$2" source_step="$3"
    if [[ ! -e "$target" ]]; then
        mkdir -p "$(dirname "$target")" "$target"
        record COPY "router source $source -> $target"
        rsync -rlptD "$source/" "$target/"
    fi
    if [[ ! -f "$target/latest_checkpointed_iteration.txt" ]]; then
        echo "[ERROR] router checkpoint copy has no tracker: $target" >&2
        exit 1
    fi
    if [[ ! -f "$target/PHASE3_SOURCE.txt" ]]; then
        {
            printf 'source_checkpoint=%s\n' "$source"
            printf 'source_step=%s\n' "$source_step"
        } > "$target/PHASE3_SOURCE.txt"
    fi
}

run_router_stage() {
    local label="$1" source="$2" target="$3" source_step="$4" experts="$5" include_conv="$6" port="$7"
    local target_step=$((source_step + 1800))
    local expansion_boundary=$((experts - 8))
    if checkpoint_at "$target" "$target_step"; then
        record SKIP "$label already complete at step $target_step: $target"
        return
    fi
    checkpoint_at "$source" "$source_step" || {
        echo "[ERROR] $label source must be at step $source_step: $source" >&2
        exit 1
    }
    prepare_router_copy "$source" "$target" "$source_step"

    local conv_data="" tertiary_data="" tertiary_name="" tertiary_interval=0 mixture=equal_dataset
    if [[ "$include_conv" == 1 ]]; then
        conv_data="$FLAME_DATA_ROOT/conversation/train"
        tertiary_data="$FLAME_DATA_ROOT/conversation/test"
        tertiary_name=conversation_probe
        tertiary_interval=100
        mixture=equal_dataset
    fi

    run_stage "$label" "$target" "$target_step" "$LOG_ROOT/${label}.log" \
        env \
            RUN_ID="$label" TRAIN_WEIGHTS="$target" LOG_DIR="$target/logs" \
            CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" NPROC_PER_NODE="$NPROC_PER_NODE" MASTER_PORT="$port" \
            SOURCE_STEP="$source_step" RETUNE_ITERS=1800 TRAIN_ITERS="$target_step" \
            SOURCE_NUM_EXPERTS="$expansion_boundary" RESUME_FROM_NUM_EXPERTS="$expansion_boundary" NUM_EXPERTS="$experts" \
            MICRO_BATCH_SIZE=48 GLOBAL_BATCH_SIZE=2304 LR=3e-4 MIN_LR=3e-5 \
            SAVE_INTERVAL=300 EVAL_INTERVAL="$target_step" LOG_INTERVAL=20 \
            TRAIN_DATASET_WIKI="$FLAME_DATA_ROOT/wiki/train" \
            TRAIN_DATASET_CODE="$FLAME_DATA_ROOT/code/train" \
            TRAIN_DATASET_CONVERSATION="$conv_data" \
            MIXED_DATA_WEIGHT_MODE="$mixture" \
            MOE_AUX_LOSS_COEFF=0.0 MOE_Z_LOSS_COEFF=0.0 MOE_LPR_LOSS_COEFF=0.0 \
            PROBE_DATASET="$FLAME_DATA_ROOT/code/test" PROBE_NAME=code_probe PROBE_EVAL_INTERVAL=100 \
            SECONDARY_PROBE_DATASET="$FLAME_DATA_ROOT/wiki/test" SECONDARY_PROBE_NAME=wiki_probe SECONDARY_PROBE_EVAL_INTERVAL=100 \
            TERTIARY_PROBE_DATASET="$tertiary_data" TERTIARY_PROBE_NAME="$tertiary_name" TERTIARY_PROBE_EVAL_INTERVAL="$tertiary_interval" \
            WANDB_MODE="$WANDB_MODE" WANDB_PROJECT="" DIRECT_LOCAL_SAVE=1 \
            bash "$SCRIPT_DIR/phase3_router_only_retune_moe_mixed_local_bf16.sh"
}

record CHAIN "B -> Code -> router FT -> KD init -> Conversation -> router FT -> TRACE"

run_stage "01_code_only" "$D_WEIGHTS" 1800 "$LOG_ROOT/01_code_only.log" \
    env MASTER_PORT=33901 bash "$SCRIPT_DIR/run_fingerprint_D_B_to_code_only_1800_mha.sh"

run_router_stage "02_code_wiki_router_only_lm" "$D_WEIGHTS" "$CODE_ROUTER_WEIGHTS" 1800 16 0 33902

run_stage "03_conversation_expand_kd_init" "$CONV_KD_WEIGHTS" 600 "$LOG_ROOT/03_conversation_expand_kd_init.log" \
    env \
        SOURCE_WEIGHTS_DIR="$CODE_ROUTER_WEIGHTS" SOURCE_REQUIRED_ITERS=3600 \
        TRAIN_WEIGHTS="$CONV_KD_WEIGHTS" RUN_ID=Conversation-expand-WikiCode-logits-KD-step600 \
        TRAIN_ITERS=600 MICRO_BATCH_SIZE=36 GLOBAL_BATCH_SIZE=2304 \
        SAVE_INTERVAL=300 EVAL_INTERVAL=600 RESUME_CONTINUAL_FROM_TRAIN_WEIGHTS=1 \
        DIRECT_LOCAL_SAVE=1 MASTER_PORT=33903 WANDB_MODE="$WANDB_MODE" \
        bash "$SCRIPT_DIR/run_g2_ffn_only_conversation_expert_distill_init_wikicode_mha.sh"

run_stage "04_conversation_only" "$CONV_WEIGHTS" 1800 "$LOG_ROOT/04_conversation_only.log" \
    env MASTER_PORT=33904 bash "$SCRIPT_DIR/run_fingerprint_conversation_only_from_kd_init_1800_mha.sh"

run_router_stage "05_wiki_code_conversation_router_only_lm" "$CONV_WEIGHTS" "$FINAL_ROUTER_WEIGHTS" 1800 24 1 33905

record TRACE "resuming interrupted sparse-15 queue: $TRACE_RUN"
export OURS_LORAMOE_OUTPUT_ROOT="$TRACE_RUN"
export SPARSE15_METHODS="$TRACE_METHOD"
export SPARSE15_GPUS=0,1,2,3,4,5,6,7
export SPARSE15_LOG_ROOT="$TRACE_RUN/eval_queue_logs"
export SPARSE15_EVAL_BATCH=32
export SPARSE15_SCIENCEQA_BATCH=128
export SPARSE15_20MINUTEN_BATCH=32
export SPARSE15_MEETINGBANK_BATCH=1
export SPARSE15_PY150_BATCH=8
export SPARSE15_CPU_THREADS=4
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
"$TRACE_ROOT/.venv-runtime/bin/python" -u "$TRACE_ROOT/scripts/run_ours_sparse15_efficient.py" \
    2>&1 | tee -a "$LOG_ROOT/06_trace_eval.log"
[[ -s "$TRACE_RUN/sparse15_summary.json" ]] || {
    record ERROR "TRACE evaluator returned without sparse15_summary.json"
    exit 1
}

record COMPLETE "all training stages and TRACE evaluation completed"
record GPU_HOLD "starting /home/seonghyeonnoh/yemokoo/gpu_hold.py on GPUs 0-7"
exec "$FLAME_ENV/bin/python" /home/seonghyeonnoh/yemokoo/gpu_hold.py --gpus 0,1,2,3,4,5,6,7
