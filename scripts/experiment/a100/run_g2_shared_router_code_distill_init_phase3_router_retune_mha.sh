#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

usage() {
    cat >&2 <<'EOF'
usage: run_g2_shared_router_code_distill_init_phase3_router_retune_mha.sh <mode|all>

Stage C of the G2 shared-router (FFN + QKVO attention experts) pre-Code
expert-init experiment: router-only retuning on wiki+code mixed data
(RETUNE_ITERS steps) for each Stage B checkpoint produced by
run_g2_shared_router_code_from_distill_init_mha.sh.

  mode: logits | logits_hidden | logits_hidden_router | all

Each run copies the completed Stage B checkpoint into a phase3 working directory
before retuning, so the Stage B checkpoints are not modified. Only the shared MoE
router (all rows) trains; experts and everything else stay frozen. Data = wiki +
code. This mirrors run_g2_phase3_router_only_retune_mha.sh but sources the
shared-router distill-init lineage instead of the random-init baseline.
EOF
}

if [ "$#" -ne 1 ]; then
    usage
    exit 1
fi

MODE_ARG="$1"
case "$MODE_ARG" in
    logits|logits_hidden|logits_hidden_router|all)
        ;;
    *)
        usage
        exit 1
        ;;
esac

export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_LOG_CHECKPOINTS="${WANDB_LOG_CHECKPOINTS:-0}"
export RETUNE_ITERS="${RETUNE_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-600}"
export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-1}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-72}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export G2_ROOT="${G2_ROOT:-$PROJECT_ROOT/.local/weights/a100/mha/g2-checkpoints}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-/home/work/.cache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6}"

# Shared-router hybrid (FFN + QKVO attention experts), router-only retune.
export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-8}"
export NUM_EXPERTS="${NUM_EXPERTS:-16}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.0}"
export MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.0}"
export ATTN_LORA_RANK="${ATTN_LORA_RANK:-256}"
export ATTN_LORA_ALPHA="${ATTN_LORA_ALPHA:-$ATTN_LORA_RANK}"
export ATTN_FULL_RANK_LORA_RANK="${ATTN_FULL_RANK_LORA_RANK:-256}"
export ATTN_FULL_RANK_LORA_ALPHA="${ATTN_FULL_RANK_LORA_ALPHA:-$ATTN_FULL_RANK_LORA_RANK}"
export ATTN_FULL_RANK_LORA_TARGETS="${ATTN_FULL_RANK_LORA_TARGETS:-qkvo}"
export ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="${ATTN_FULL_RANK_LORA_ACTIVE_TARGETS:-}"
export ROUTER_MEMORY_KL_COEFF=0.0
export ROUTER_MEMORY_INTERVAL=0
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"
export ATTN_LORA_GROUPED_GEMM="${ATTN_LORA_GROUPED_GEMM:-1}"
export MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}"

# Stage B (source) and Stage C (output) layout.
export STAGE_B_MB="${STAGE_B_MB:-72}"
export STAGE_B_ITERS="${STAGE_B_ITERS:-1800}"
export STAGE_B_ROOT="${STAGE_B_ROOT:-$G2_ROOT/code/shared_router_from_distill_init}"
export PHASE3_ROOT="${PHASE3_ROOT:-$G2_ROOT/code/shared_router_from_distill_init_phase3}"
mkdir -p "$PHASE3_ROOT"

tracker_value() {
    tr -d '\n\r[:space:]' < "$1/latest_checkpointed_iteration.txt"
}

ensure_copy() {
    local src="$1"
    local dst="$2"
    local label="$3"

    if [ -e "$dst" ]; then
        return
    fi

    local src_step
    src_step="$(tracker_value "$src")"
    if [ "$src_step" != "$STAGE_B_ITERS" ]; then
        echo "[ERROR] Stage B source is not complete. expected $STAGE_B_ITERS, got $src_step: $src" >&2
        exit 1
    fi

    echo "[COPY] source:      $src"
    echo "[COPY] destination: $dst"
    mkdir -p "$dst"
    rsync -aH --info=progress2 \
        --exclude 'wandb/' \
        "$src/" "$dst/"

    {
        echo "phase=3"
        echo "purpose=shared-router distill-init Stage C router-only retuning copy"
        echo "source_label=$label"
        echo "source=$src"
        echo "source_step=$src_step"
        echo "router_init=no-reinit"
    } > "$dst/PHASE3_SOURCE.txt"
    echo "[DONE] $dst"
}

run_one() {
    local mode="$1"
    local safe_mode="${mode//_/-}"

    local source_run_id="g2-shared-router-e8to16-code-from-distill-init-${safe_mode}-qkvo-mha-a100-bf16-mb${STAGE_B_MB}-${STAGE_B_ITERS}"
    local src="${STAGE_B_ROOT}/${source_run_id}"
    if [ ! -f "$src/latest_checkpointed_iteration.txt" ]; then
        echo "[ERROR] Stage B checkpoint not found: $src" >&2
        echo "        Run Stage B first: run_g2_shared_router_code_from_distill_init_mha.sh $mode" >&2
        exit 1
    fi

    local run_id="g2-shared-router-code-from-distill-init-${safe_mode}-phase3-router-retune-wikicode-mb${MICRO_BATCH_SIZE}-${RETUNE_ITERS}"
    local exp_name="G2 shared-router - ${mode} distill-init Stage C router-only retune wiki+code"
    local weights="$PHASE3_ROOT/$run_id"
    local port

    case "$mode" in
        logits) port="${MASTER_PORT_LOGITS:-29791}" ;;
        logits_hidden) port="${MASTER_PORT_LOGITS_HIDDEN:-29792}" ;;
        logits_hidden_router) port="${MASTER_PORT_LOGITS_HIDDEN_ROUTER:-29793}" ;;
    esac

    ensure_copy "$src" "$weights" "$exp_name source"

    local base_step
    if [ -f "$weights/PHASE3_SOURCE.txt" ]; then
        base_step="$(grep -E '^source_step=' "$weights/PHASE3_SOURCE.txt" | tail -1 | cut -d= -f2- || true)"
    fi
    base_step="${base_step:-$(tracker_value "$weights")}"

    local latest_step
    latest_step="$(tracker_value "$weights")"
    local target_step=$((base_step + RETUNE_ITERS))
    if [ "$latest_step" -ge "$target_step" ]; then
        echo "[SKIP] already at or past target: $weights latest=$latest_step target=$target_step"
        return
    fi

    echo "[CONFIG] G2 shared-router distill-init Stage C router-only retune (mode=$mode)"
    echo "[CONFIG] source(Stage B)=$src"
    echo "[CONFIG] base_step=${base_step}, latest_step=${latest_step}, retune_iters=${RETUNE_ITERS}, target_step=${target_step}"
    echo "[CONFIG] data=wiki train + code train, trainable=shared MoE router only (all rows)"
    echo "[CONFIG] wandb_mode=${WANDB_MODE}, weights=${weights}"

    env \
        RUN_ID="$run_id" \
        TRAIN_WEIGHTS="$weights" \
        SOURCE_STEP="$base_step" \
        TRAIN_ITERS="$target_step" \
        WANDB_EXP_NAME="$exp_name" \
        WANDB_RUN_ID="$run_id" \
        MASTER_PORT="$port" \
        SHARED_ROUTER_HYBRID_REINIT_ROUTER=0 \
        bash "$SCRIPT_DIR/run_guarded_training.sh" \
            bash "$SCRIPT_DIR/phase3_router_only_retune_shared_router_hybrid_mixed_local_bf16.sh"
}

if [ "$MODE_ARG" = "all" ]; then
    run_one logits
    run_one logits_hidden
    run_one logits_hidden_router
else
    run_one "$MODE_ARG"
fi
