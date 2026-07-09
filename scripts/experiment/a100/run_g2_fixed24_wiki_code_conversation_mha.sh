#!/bin/bash
set -euo pipefail

# G2 fixed-capacity FFN-only MoE baseline: wiki -> code -> conversation.
#
# "fixed24" = 24 FFN experts available from the Wiki stage (no 8->16->24
# expansion). Every stage full-finetunes the whole model (all 24 experts, the
# shared router, and every shared/dense parameter). No expert/router freezing.
#
# - FFN-only MoE (no always-on shared expert, no attention experts) so the
#   active FFN capacity is top-4 * 352 = 1408, matching the dense4_active
#   baseline on activated parameters while total capacity (24 * 352 = 8448)
#   matches the fully-expanded model at the end of the 3-task chain.
# - code/conversation stages use old-model logits KD; coefficient is adjustable
#   via OLD_MODEL_KL_COEFF (default 1.0 = kl1; set 0.0 for kl0 / no KD).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

run_and_pause() {
    local name="$1"
    shift
    echo "[START] $name $(date)"
    "$@"
    echo "[END] $name $(date)"
    sleep "${PAUSE_SECONDS:-180}"
}

# A stage is only "done" when its checkpoint tracker reports the FULL expected
# iteration count. A tracker can exist after any periodic save (e.g. an OOM
# crash mid-stage at iteration 1200/1800), so a plain file-exists check would
# wrongly skip a stage that only partially finished. Re-running an incomplete
# stage re-enters the engine, which resumes from that stage's own latest
# checkpoint (see continual_code_from_wiki_dense_local_bf16.sh /
# pretrain_wiki_dense_local_bf16.sh) instead of restarting from iteration 0.
is_stage_complete() {
    local weights_dir="$1"
    local expected_iters="$2"
    local tracker="$weights_dir/latest_checkpointed_iteration.txt"
    [ -f "$tracker" ] && [ "$(tr -d '\n\r[:space:]' < "$tracker")" = "$expected_iters" ]
}

export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
# Use `-` (not `:-`) so an explicitly-empty WANDB_PROJECT="" disables W&B entirely
# (Megatron skips importing/initing wandb when wandb_project is empty).
export WANDB_PROJECT="${WANDB_PROJECT-flame-continual-top2-qv-lora}"
export WANDB_MODE="${WANDB_MODE:-offline}"

export SEED="${SEED:-1234}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
export NUM_LAYERS="${NUM_LAYERS:-9}"
export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
# Dense FFN size of the single non-MoE layer (layer 1), G2 FFN-only convention.
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-5472}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-48}"
# Wiki stage has no KD teacher (unlike code/conv), so it fits a larger micro batch
# on the same GPU. Defaults to MICRO_BATCH_SIZE; override independently to speed up
# the wiki stage without touching the KD-constrained code/conv micro batch (path
# naming below still keys off MICRO_BATCH_SIZE, so downstream stage lookups match).
export WIKI_MICRO_BATCH_SIZE="${WIKI_MICRO_BATCH_SIZE:-$MICRO_BATCH_SIZE}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-600}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}"
export TERTIARY_PROBE_EVAL_INTERVAL="${TERTIARY_PROBE_EVAL_INTERVAL:-50}"

# --- MoE architecture (FFN-only, fixed 24 experts) ---
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-configs/model/flame-moe-ffn-only-no-shared.sh}"
export NUM_EXPERTS="${NUM_EXPERTS:-24}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_LAYER_FREQ="${MOE_LAYER_FREQ:-[0,1,1,1,1,1,1,1,1]}"
export MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}"
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.01}"
export MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.001}"
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"

# --- KD (old-model logits distillation) for code/conversation stages ---
# kl1 (default): OLD_MODEL_KL_COEFF=1.0 ; kl0: OLD_MODEL_KL_COEFF=0.0
export OLD_MODEL_KL_COEFF="${OLD_MODEL_KL_COEFF:-1.0}"
export OLD_MODEL_KL_TEMPERATURE="${OLD_MODEL_KL_TEMPERATURE:-1.0}"

# KD tag used only for run-id readability.
if [ "$OLD_MODEL_KL_COEFF" = "0.0" ] || [ "$OLD_MODEL_KL_COEFF" = "0" ]; then
    KD_TAG="${KD_TAG:-kl0}"
else
    KD_TAG="${KD_TAG:-kl1}"
fi

export VARIANT_TAG="${VARIANT_TAG:-fixed24}"
export VARIANT_LABEL="${VARIANT_LABEL:-fixed24 ffn-only e${NUM_EXPERTS} moe${MOE_FFN_HIDDEN_SIZE}}"

export WIKI_TRAIN_DATASET="${WIKI_TRAIN_DATASET:-$PROJECT_ROOT/data/wiki/train}"
export WIKI_TEST_DATASET="${WIKI_TEST_DATASET:-$PROJECT_ROOT/data/wiki/test}"
export CODE_TRAIN_DATASET="${CODE_TRAIN_DATASET:-$PROJECT_ROOT/data/code/train}"
export CODE_TEST_DATASET="${CODE_TEST_DATASET:-$PROJECT_ROOT/data/code/test}"
export CONVERSATION_TRAIN_DATASET="${CONVERSATION_TRAIN_DATASET:-$PROJECT_ROOT/data/conversation/train}"
export CONVERSATION_TEST_DATASET="${CONVERSATION_TEST_DATASET:-$PROJECT_ROOT/data/conversation/test}"

export WIKI_RUN_ID="${WIKI_RUN_ID:-g2-${VARIANT_TAG}-e${NUM_EXPERTS}-ffn${MOE_FFN_HIDDEN_SIZE}-top${MOE_ROUTER_TOPK}-wiki-ffn-moe-mha-a100-bf16-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"
export CODE_RUN_ID="${CODE_RUN_ID:-g2-${VARIANT_TAG}-e${NUM_EXPERTS}-ffn${MOE_FFN_HIDDEN_SIZE}-top${MOE_ROUTER_TOPK}-wiki-to-code-fullfinetune-${KD_TAG}-mha-a100-bf16-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"
export CONV_RUN_ID="${CONV_RUN_ID:-g2-${VARIANT_TAG}-e${NUM_EXPERTS}-ffn${MOE_FFN_HIDDEN_SIZE}-top${MOE_ROUTER_TOPK}-wikicode-to-conversation-fullfinetune-${KD_TAG}-mha-a100-bf16-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"

export FIXED24_ROOT="${FIXED24_ROOT:-$PROJECT_ROOT/.local/weights/a100/mha/${VARIANT_TAG}}"
export WIKI_WEIGHTS="${WIKI_WEIGHTS:-$FIXED24_ROOT/wiki/$WIKI_RUN_ID}"
export CODE_WEIGHTS="${CODE_WEIGHTS:-$FIXED24_ROOT/code/$CODE_RUN_ID}"
export CONV_WEIGHTS="${CONV_WEIGHTS:-$FIXED24_ROOT/conversation/$CONV_RUN_ID}"

echo "[CONFIG] G2 ${VARIANT_LABEL} wiki -> code -> conversation continual baseline"
echo "[CONFIG] fixed-capacity FFN-only MoE: experts=$NUM_EXPERTS topk=$MOE_ROUTER_TOPK moe_ffn=$MOE_FFN_HIDDEN_SIZE (no shared expert, no attention experts)"
echo "[CONFIG] active FFN = topk*moe_ffn = $((MOE_ROUTER_TOPK * MOE_FFN_HIDDEN_SIZE)); total FFN capacity = experts*moe_ffn = $((NUM_EXPERTS * MOE_FFN_HIDDEN_SIZE))"
echo "[CONFIG] dense(layer1) ffn=$FFN_HIDDEN_SIZE seed=$SEED mb=$MICRO_BATCH_SIZE gbs=$GLOBAL_BATCH_SIZE grouped_gemm=$MOE_GROUPED_GEMM"
echo "[CONFIG] no freeze: all experts/router/shared params trainable in every stage"
echo "[CONFIG] aux_loss=$MOE_AUX_LOSS_COEFF z_loss=$MOE_Z_LOSS_COEFF"
echo "[CONFIG] code/conv old-model logits KD coeff=$OLD_MODEL_KL_COEFF temp=$OLD_MODEL_KL_TEMPERATURE tag=$KD_TAG"
echo "[CONFIG] wiki=$WIKI_WEIGHTS"
echo "[CONFIG] code=$CODE_WEIGHTS"
echo "[CONFIG] conversation=$CONV_WEIGHTS"

# ---------------------------------------------------------------------------
# Stage 1: Wiki pretrain (fixed 24 experts from scratch)
# ---------------------------------------------------------------------------
if ! is_stage_complete "$WIKI_WEIGHTS" "$TRAIN_ITERS"; then
    run_and_pause "${VARIANT_TAG}_wiki" env \
        WANDB_MODE="$WANDB_MODE" \
        RUN_ID="$WIKI_RUN_ID" \
        TRAIN_WEIGHTS="$WIKI_WEIGHTS" \
        WANDB_PROJECT="$WANDB_PROJECT" \
        WANDB_EXP_NAME="${WIKI_WANDB_EXP_NAME:-G2 - ${VARIANT_LABEL} phase1 wiki ⭐}" \
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
        MICRO_BATCH_SIZE="$WIKI_MICRO_BATCH_SIZE" \
        GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
        TRAIN_ITERS="$TRAIN_ITERS" \
        SAVE_INTERVAL="$SAVE_INTERVAL" \
        EVAL_INTERVAL="$EVAL_INTERVAL" \
        LOG_INTERVAL="$LOG_INTERVAL" \
        PROBE_EVAL_INTERVAL="$PROBE_EVAL_INTERVAL" \
        SECONDARY_PROBE_EVAL_INTERVAL="$SECONDARY_PROBE_EVAL_INTERVAL" \
        TERTIARY_PROBE_EVAL_INTERVAL="$TERTIARY_PROBE_EVAL_INTERVAL" \
        TRAIN_DATASET="$WIKI_TRAIN_DATASET" \
        DATASET_NAME="wiki_train" \
        DATASET_SOURCE="Wikipedia train" \
        PROBE_NAME="wiki_probe" \
        PROBE_DATASET="$WIKI_TEST_DATASET" \
        SECONDARY_PROBE_NAME="code_probe" \
        SECONDARY_PROBE_DATASET="$CODE_TEST_DATASET" \
        TERTIARY_PROBE_NAME="conversation_probe" \
        TERTIARY_PROBE_DATASET="$CONVERSATION_TEST_DATASET" \
        MASTER_PORT="${WIKI_MASTER_PORT:-29981}" \
        bash "$SCRIPT_DIR/pretrain_wiki_dense_local_bf16.sh"
else
    echo "[SKIP] ${VARIANT_TAG}_wiki already completed: $WIKI_WEIGHTS"
fi

# ---------------------------------------------------------------------------
# Stage 2: Code (full-finetune all 24 experts + router + shared params)
# ---------------------------------------------------------------------------
if ! is_stage_complete "$CODE_WEIGHTS" "$TRAIN_ITERS"; then
    run_and_pause "${VARIANT_TAG}_code" env \
        WANDB_MODE="$WANDB_MODE" \
        RUN_ID="$CODE_RUN_ID" \
        STAGE1_WEIGHTS_DIR="$WIKI_WEIGHTS" \
        TRAIN_WEIGHTS="$CODE_WEIGHTS" \
        WANDB_PROJECT="$WANDB_PROJECT" \
        WANDB_EXP_NAME="${CODE_WANDB_EXP_NAME:-G2 - ${VARIANT_LABEL} phase2 code ${KD_TAG} ⭐}" \
        MODEL_CONFIG_SCRIPT="$MODEL_CONFIG_SCRIPT" \
        METADATA_STAGE="code_from_wiki_${VARIANT_TAG}_ffn_moe_full_finetune" \
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
        TRAIN_ITERS="$TRAIN_ITERS" \
        SAVE_INTERVAL="$SAVE_INTERVAL" \
        EVAL_INTERVAL="$EVAL_INTERVAL" \
        LOG_INTERVAL="$LOG_INTERVAL" \
        PROBE_EVAL_INTERVAL="$PROBE_EVAL_INTERVAL" \
        SECONDARY_PROBE_EVAL_INTERVAL="$SECONDARY_PROBE_EVAL_INTERVAL" \
        TERTIARY_PROBE_EVAL_INTERVAL="$TERTIARY_PROBE_EVAL_INTERVAL" \
        OLD_MODEL_KL_COEFF="$OLD_MODEL_KL_COEFF" \
        OLD_MODEL_KL_TEMPERATURE="$OLD_MODEL_KL_TEMPERATURE" \
        TRAIN_DATASET="$CODE_TRAIN_DATASET" \
        DATASET_NAME="code_train" \
        DATASET_SOURCE="Code train" \
        PROBE_NAME="code_probe" \
        PROBE_DATASET="$CODE_TEST_DATASET" \
        SECONDARY_PROBE_NAME="wiki_probe" \
        SECONDARY_PROBE_DATASET="$WIKI_TEST_DATASET" \
        TERTIARY_PROBE_NAME="conversation_probe" \
        TERTIARY_PROBE_DATASET="$CONVERSATION_TEST_DATASET" \
        PROBE_STEP_OFFSET="$TRAIN_ITERS" \
        SECONDARY_PROBE_STEP_OFFSET="$TRAIN_ITERS" \
        TERTIARY_PROBE_STEP_OFFSET="$TRAIN_ITERS" \
        WANDB_STEP_OFFSET="$TRAIN_ITERS" \
        MASTER_PORT="${CODE_MASTER_PORT:-29982}" \
        bash "$PROJECT_ROOT/scripts/experiment/continual_code_from_wiki_dense_local_bf16.sh"
else
    echo "[SKIP] ${VARIANT_TAG}_code already completed: $CODE_WEIGHTS"
fi

# ---------------------------------------------------------------------------
# Stage 3: Conversation (full-finetune, continue from code checkpoint)
# ---------------------------------------------------------------------------
if ! is_stage_complete "$CONV_WEIGHTS" "$TRAIN_ITERS"; then
    CUMULATIVE_CODE_OFFSET=$((TRAIN_ITERS * 2))
    run_and_pause "${VARIANT_TAG}_conversation" env \
        WANDB_MODE="$WANDB_MODE" \
        RUN_ID="$CONV_RUN_ID" \
        STAGE1_WEIGHTS_DIR="$CODE_WEIGHTS" \
        TRAIN_WEIGHTS="$CONV_WEIGHTS" \
        WANDB_PROJECT="$WANDB_PROJECT" \
        WANDB_EXP_NAME="${CONV_WANDB_EXP_NAME:-G2 - ${VARIANT_LABEL} phase3 conversation ${KD_TAG} ⭐}" \
        MODEL_CONFIG_SCRIPT="$MODEL_CONFIG_SCRIPT" \
        METADATA_STAGE="conversation_from_code_${VARIANT_TAG}_ffn_moe_full_finetune" \
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
        TRAIN_ITERS="$TRAIN_ITERS" \
        SAVE_INTERVAL="$SAVE_INTERVAL" \
        EVAL_INTERVAL="$EVAL_INTERVAL" \
        LOG_INTERVAL="$LOG_INTERVAL" \
        PROBE_EVAL_INTERVAL="$PROBE_EVAL_INTERVAL" \
        SECONDARY_PROBE_EVAL_INTERVAL="$SECONDARY_PROBE_EVAL_INTERVAL" \
        TERTIARY_PROBE_EVAL_INTERVAL="$TERTIARY_PROBE_EVAL_INTERVAL" \
        OLD_MODEL_KL_COEFF="$OLD_MODEL_KL_COEFF" \
        OLD_MODEL_KL_TEMPERATURE="$OLD_MODEL_KL_TEMPERATURE" \
        TRAIN_DATASET="$CONVERSATION_TRAIN_DATASET" \
        DATASET_NAME="conversation_train" \
        DATASET_SOURCE="Conversation train" \
        PROBE_NAME="conversation_probe" \
        PROBE_DATASET="$CONVERSATION_TEST_DATASET" \
        SECONDARY_PROBE_NAME="wiki_probe" \
        SECONDARY_PROBE_DATASET="$WIKI_TEST_DATASET" \
        TERTIARY_PROBE_NAME="code_probe" \
        TERTIARY_PROBE_DATASET="$CODE_TEST_DATASET" \
        PROBE_STEP_OFFSET="$CUMULATIVE_CODE_OFFSET" \
        SECONDARY_PROBE_STEP_OFFSET="$CUMULATIVE_CODE_OFFSET" \
        TERTIARY_PROBE_STEP_OFFSET="$CUMULATIVE_CODE_OFFSET" \
        WANDB_STEP_OFFSET="$CUMULATIVE_CODE_OFFSET" \
        MASTER_PORT="${CONV_MASTER_PORT:-29983}" \
        bash "$PROJECT_ROOT/scripts/experiment/continual_code_from_wiki_dense_local_bf16.sh"
else
    echo "[SKIP] ${VARIANT_TAG}_conversation already completed: $CONV_WEIGHTS"
fi

echo "[ALL DONE] ${VARIANT_LABEL} wiki -> code -> conversation $(date)"
