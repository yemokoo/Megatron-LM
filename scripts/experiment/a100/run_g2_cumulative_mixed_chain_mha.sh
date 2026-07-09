#!/bin/bash
set -euo pipefail

# Cumulative-mixed continual-learning chain (CL-internal upper bound), model-agnostic
# engine shared by exp 6 (dense-active) and exp 7 (fixed24 MoE).
#
# Three stages, each re-mixing every dataset seen so far (full replay), each stage
# continuing from the previous stage's checkpoint and running its own fresh WSD
# schedule. "1800 steps per dataset" budget => stage iters grow 1800 / 3600 / 5400:
#
#   stage 1 (wiki)                    : from scratch,   1800 iters, x-offset 0
#   stage 2 (wiki+code)               : from stage-1,   3600 iters, x-offset 1800
#   stage 3 (wiki+code+conversation)  : from stage-2,   5400 iters, x-offset 5400
#                                                        (total 10800 iters)
#
# All three probes (wiki/code/conversation) are logged at every stage; probe/wandb
# step offsets are cumulative so the whole chain plots on one continuous x-axis
# (0 .. 10800). No KD (replay is the mitigation; the nway trainer has no teacher).
#
# Architecture (dense-active vs fixed24) and micro-batch come from env, set by the
# exp6/exp7 wrappers. This file only owns the 3-stage chaining + step bookkeeping.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

NWAY_TRAINER="$SCRIPT_DIR/pretrain_mixed_nway_local_bf16.sh"

run_and_pause() {
    local name="$1"; shift
    echo "[START] $name $(date)"
    "$@"
    echo "[END] $name $(date)"
    sleep "${PAUSE_SECONDS:-180}"
}

# A stage is only "done" when its checkpoint tracker reports the FULL expected
# iteration count. A tracker can exist after any periodic save (e.g. an OOM
# crash mid-stage at iteration 1200/3600), so a plain file-exists check would
# wrongly skip a stage that only partially finished. Re-running an incomplete
# stage re-enters pretrain_mixed_nway_local_bf16.sh, which resumes from that
# stage's own latest checkpoint instead of restarting from iteration 0.
is_stage_complete() {
    local weights_dir="$1"
    local expected_iters="$2"
    local tracker="$weights_dir/latest_checkpointed_iteration.txt"
    [ -f "$tracker" ] && [ "$(tr -d '\n\r[:space:]' < "$tracker")" = "$expected_iters" ]
}

export SEED="${SEED:-1234}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
export NUM_LAYERS="${NUM_LAYERS:-9}"
export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-5472}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"

# Per-dataset step budget; stage iters are 1x / 2x / 3x this.
export STEPS_PER_DATASET="${STEPS_PER_DATASET:-1800}"
STAGE1_ITERS=$((STEPS_PER_DATASET * 1))
STAGE2_ITERS=$((STEPS_PER_DATASET * 2))
STAGE3_ITERS=$((STEPS_PER_DATASET * 3))
# Cumulative x-axis offsets: stage_k starts where the previous stage ended.
STAGE1_OFFSET=0
STAGE2_OFFSET=$STAGE1_ITERS
STAGE3_OFFSET=$((STAGE1_ITERS + STAGE2_ITERS))

export SAVE_INTERVAL="${SAVE_INTERVAL:-600}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}"
export TERTIARY_PROBE_EVAL_INTERVAL="${TERTIARY_PROBE_EVAL_INTERVAL:-50}"

# --- architecture (defaults = fixed24 MoE; dense-active wrapper overrides) ---
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-configs/model/flame-moe-ffn-only-no-shared.sh}"
export NUM_EXPERTS="${NUM_EXPERTS:-24}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_LAYER_FREQ="${MOE_LAYER_FREQ:-[0,1,1,1,1,1,1,1,1]}"
export MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}"
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.01}"
export MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.001}"
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-72}"

# GPU fan-out (default all 4 A100s).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

# Use `-` (not `:-`) so an explicitly-empty WANDB_PROJECT="" disables W&B entirely.
export WANDB_PROJECT="${WANDB_PROJECT-flame-continual-top2-qv-lora}"
export WANDB_MODE="${WANDB_MODE:-offline}"

export VARIANT_TAG="${VARIANT_TAG:-fixed24}"
export VARIANT_LABEL="${VARIANT_LABEL:-fixed24 ffn-only e${NUM_EXPERTS} moe${MOE_FFN_HIDDEN_SIZE}}"

export WIKI_TRAIN_DATASET="${WIKI_TRAIN_DATASET:-$PROJECT_ROOT/data/wiki/train}"
export WIKI_TEST_DATASET="${WIKI_TEST_DATASET:-$PROJECT_ROOT/data/wiki/test}"
export CODE_TRAIN_DATASET="${CODE_TRAIN_DATASET:-$PROJECT_ROOT/data/code/train}"
export CODE_TEST_DATASET="${CODE_TEST_DATASET:-$PROJECT_ROOT/data/code/test}"
export CONVERSATION_TRAIN_DATASET="${CONVERSATION_TRAIN_DATASET:-$PROJECT_ROOT/data/conversation/train}"
export CONVERSATION_TEST_DATASET="${CONVERSATION_TEST_DATASET:-$PROJECT_ROOT/data/conversation/test}"

export CUMULATIVE_ROOT="${CUMULATIVE_ROOT:-$PROJECT_ROOT/.local/weights/a100/mha/cumulative-mixed/${VARIANT_TAG}}"

export STAGE1_RUN_ID="${STAGE1_RUN_ID:-g2-${VARIANT_TAG}-cumulmix-s1-wiki-mha-a100-bf16-mb${MICRO_BATCH_SIZE}-${STAGE1_ITERS}}"
export STAGE2_RUN_ID="${STAGE2_RUN_ID:-g2-${VARIANT_TAG}-cumulmix-s2-wiki+code-mha-a100-bf16-mb${MICRO_BATCH_SIZE}-${STAGE2_ITERS}}"
export STAGE3_RUN_ID="${STAGE3_RUN_ID:-g2-${VARIANT_TAG}-cumulmix-s3-wiki+code+conv-mha-a100-bf16-mb${MICRO_BATCH_SIZE}-${STAGE3_ITERS}}"

export STAGE1_WEIGHTS="${STAGE1_WEIGHTS:-$CUMULATIVE_ROOT/stage1_wiki/$STAGE1_RUN_ID}"
export STAGE2_WEIGHTS="${STAGE2_WEIGHTS:-$CUMULATIVE_ROOT/stage2_wiki_code/$STAGE2_RUN_ID}"
export STAGE3_WEIGHTS="${STAGE3_WEIGHTS:-$CUMULATIVE_ROOT/stage3_wiki_code_conv/$STAGE3_RUN_ID}"

echo "[CONFIG] G2 ${VARIANT_LABEL} cumulative-mixed CL upper bound (full replay)"
echo "[CONFIG] arch: experts=$NUM_EXPERTS topk=$MOE_ROUTER_TOPK moe_ffn=$MOE_FFN_HIDDEN_SIZE layer1_ffn=$FFN_HIDDEN_SIZE grouped_gemm=$MOE_GROUPED_GEMM"
echo "[CONFIG] mb=$MICRO_BATCH_SIZE gbs=$GLOBAL_BATCH_SIZE gpus=$CUDA_VISIBLE_DEVICES (nproc=$NPROC_PER_NODE)"
echo "[CONFIG] stage iters: s1=$STAGE1_ITERS s2=$STAGE2_ITERS s3=$STAGE3_ITERS (total $((STAGE1_ITERS+STAGE2_ITERS+STAGE3_ITERS)))"
echo "[CONFIG] x-offsets: s1=$STAGE1_OFFSET s2=$STAGE2_OFFSET s3=$STAGE3_OFFSET"
echo "[CONFIG] no KD (pure replay); probes = wiki/code/conversation every stage"

# Shared architecture/probe env passed to every stage.
common_stage_env() {
    echo \
        WANDB_MODE="$WANDB_MODE" \
        WANDB_PROJECT="$WANDB_PROJECT" \
        MODEL_CONFIG_SCRIPT="$MODEL_CONFIG_SCRIPT" \
        FFN_HIDDEN_SIZE="$FFN_HIDDEN_SIZE" \
        HIDDEN_SIZE="$HIDDEN_SIZE" \
        NUM_LAYERS="$NUM_LAYERS" \
        NUM_QUERY_GROUPS="$NUM_QUERY_GROUPS" \
        NUM_EXPERTS="$NUM_EXPERTS" \
        MOE_FFN_HIDDEN_SIZE="$MOE_FFN_HIDDEN_SIZE" \
        MOE_ROUTER_TOPK="$MOE_ROUTER_TOPK" \
        MOE_LAYER_FREQ="$MOE_LAYER_FREQ" \
        MOE_ROUTER_DTYPE="$MOE_ROUTER_DTYPE" \
        MOE_AUX_LOSS_COEFF="$MOE_AUX_LOSS_COEFF" \
        MOE_Z_LOSS_COEFF="$MOE_Z_LOSS_COEFF" \
        MOE_GROUPED_GEMM="$MOE_GROUPED_GEMM" \
        SEED="$SEED" \
        MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" \
        GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        NPROC_PER_NODE="$NPROC_PER_NODE" \
        SAVE_INTERVAL="$SAVE_INTERVAL" \
        EVAL_INTERVAL="$EVAL_INTERVAL" \
        LOG_INTERVAL="$LOG_INTERVAL" \
        PROBE_EVAL_INTERVAL="$PROBE_EVAL_INTERVAL" \
        SECONDARY_PROBE_EVAL_INTERVAL="$SECONDARY_PROBE_EVAL_INTERVAL" \
        TERTIARY_PROBE_EVAL_INTERVAL="$TERTIARY_PROBE_EVAL_INTERVAL" \
        PROBE_NAME="wiki_probe" \
        PROBE_DATASET="$WIKI_TEST_DATASET" \
        SECONDARY_PROBE_NAME="code_probe" \
        SECONDARY_PROBE_DATASET="$CODE_TEST_DATASET" \
        TERTIARY_PROBE_NAME="conversation_probe" \
        TERTIARY_PROBE_DATASET="$CONVERSATION_TEST_DATASET"
}

# ---------------------------------------------------------------------------
# Stage 1: wiki only, from scratch (1800 iters)
# ---------------------------------------------------------------------------
if ! is_stage_complete "$STAGE1_WEIGHTS" "$STAGE1_ITERS"; then
    run_and_pause "${VARIANT_TAG}_cumulmix_s1_wiki" env \
        $(common_stage_env) \
        RUN_ID="$STAGE1_RUN_ID" \
        TRAIN_WEIGHTS="$STAGE1_WEIGHTS" \
        WANDB_EXP_NAME="${STAGE1_WANDB_EXP_NAME:-G2 - ${VARIANT_LABEL} cumulmix S1 wiki ⭐}" \
        MIXED_TRAIN_DATASETS="$WIKI_TRAIN_DATASET" \
        STAGE_SOURCE_WEIGHTS="" \
        TRAIN_ITERS="$STAGE1_ITERS" \
        LR_DECAY_ITERS="$STAGE1_ITERS" \
        LR_WSD_DECAY_ITERS="$((STAGE1_ITERS / 10))" \
        DATASET_NAME="cumulmix_s1_wiki" \
        DATASET_SOURCE="Wikipedia (cumulative stage 1)" \
        PROBE_STEP_OFFSET="$STAGE1_OFFSET" \
        SECONDARY_PROBE_STEP_OFFSET="$STAGE1_OFFSET" \
        TERTIARY_PROBE_STEP_OFFSET="$STAGE1_OFFSET" \
        WANDB_STEP_OFFSET="$STAGE1_OFFSET" \
        MASTER_PORT="${STAGE1_MASTER_PORT:-29991}" \
        bash "$NWAY_TRAINER"
else
    echo "[SKIP] ${VARIANT_TAG}_cumulmix_s1_wiki already completed: $STAGE1_WEIGHTS"
fi

# ---------------------------------------------------------------------------
# Stage 2: wiki + code (1:1), continue from stage 1 (3600 iters)
# ---------------------------------------------------------------------------
if ! is_stage_complete "$STAGE2_WEIGHTS" "$STAGE2_ITERS"; then
    run_and_pause "${VARIANT_TAG}_cumulmix_s2_wiki_code" env \
        $(common_stage_env) \
        RUN_ID="$STAGE2_RUN_ID" \
        TRAIN_WEIGHTS="$STAGE2_WEIGHTS" \
        WANDB_EXP_NAME="${STAGE2_WANDB_EXP_NAME:-G2 - ${VARIANT_LABEL} cumulmix S2 wiki+code ⭐}" \
        MIXED_TRAIN_DATASETS="$WIKI_TRAIN_DATASET $CODE_TRAIN_DATASET" \
        STAGE_SOURCE_WEIGHTS="$STAGE1_WEIGHTS" \
        TRAIN_ITERS="$STAGE2_ITERS" \
        LR_DECAY_ITERS="$STAGE2_ITERS" \
        LR_WSD_DECAY_ITERS="$((STAGE2_ITERS / 10))" \
        DATASET_NAME="cumulmix_s2_wiki_code" \
        DATASET_SOURCE="Wikipedia + Code 1:1 (cumulative stage 2)" \
        PROBE_STEP_OFFSET="$STAGE2_OFFSET" \
        SECONDARY_PROBE_STEP_OFFSET="$STAGE2_OFFSET" \
        TERTIARY_PROBE_STEP_OFFSET="$STAGE2_OFFSET" \
        WANDB_STEP_OFFSET="$STAGE2_OFFSET" \
        MASTER_PORT="${STAGE2_MASTER_PORT:-29992}" \
        bash "$NWAY_TRAINER"
else
    echo "[SKIP] ${VARIANT_TAG}_cumulmix_s2_wiki_code already completed: $STAGE2_WEIGHTS"
fi

# ---------------------------------------------------------------------------
# Stage 3: wiki + code + conversation (1:1:1), continue from stage 2 (5400 iters)
# ---------------------------------------------------------------------------
if ! is_stage_complete "$STAGE3_WEIGHTS" "$STAGE3_ITERS"; then
    run_and_pause "${VARIANT_TAG}_cumulmix_s3_wiki_code_conv" env \
        $(common_stage_env) \
        RUN_ID="$STAGE3_RUN_ID" \
        TRAIN_WEIGHTS="$STAGE3_WEIGHTS" \
        WANDB_EXP_NAME="${STAGE3_WANDB_EXP_NAME:-G2 - ${VARIANT_LABEL} cumulmix S3 wiki+code+conv ⭐}" \
        MIXED_TRAIN_DATASETS="$WIKI_TRAIN_DATASET $CODE_TRAIN_DATASET $CONVERSATION_TRAIN_DATASET" \
        STAGE_SOURCE_WEIGHTS="$STAGE2_WEIGHTS" \
        TRAIN_ITERS="$STAGE3_ITERS" \
        LR_DECAY_ITERS="$STAGE3_ITERS" \
        LR_WSD_DECAY_ITERS="$((STAGE3_ITERS / 10))" \
        DATASET_NAME="cumulmix_s3_wiki_code_conv" \
        DATASET_SOURCE="Wikipedia + Code + Conversation 1:1:1 (cumulative stage 3)" \
        PROBE_STEP_OFFSET="$STAGE3_OFFSET" \
        SECONDARY_PROBE_STEP_OFFSET="$STAGE3_OFFSET" \
        TERTIARY_PROBE_STEP_OFFSET="$STAGE3_OFFSET" \
        WANDB_STEP_OFFSET="$STAGE3_OFFSET" \
        MASTER_PORT="${STAGE3_MASTER_PORT:-29993}" \
        bash "$NWAY_TRAINER"
else
    echo "[SKIP] ${VARIANT_TAG}_cumulmix_s3_wiki_code_conv already completed: $STAGE3_WEIGHTS"
fi

echo "[ALL DONE] ${VARIANT_LABEL} cumulative-mixed wiki -> wiki+code -> wiki+code+conv $(date)"
