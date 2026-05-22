#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

usage() {
    cat >&2 <<'EOF'
usage: run_g2_phase3_router_only_retune_mha.sh <exp1|exp2|all>

Runs Phase 3 router-only retuning on a completed Phase 3 checkpoint copy.
Use prepare_g2_phase3_copy.sh first. For exp2, run after Phase 1 reaches step 1800.
EOF
}

if [ "$#" -ne 1 ]; then
    usage
    exit 1
fi

TARGET="$1"
case "$TARGET" in
    exp1|exp2|all)
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
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-72}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export G2_ROOT="${G2_ROOT:-$PROJECT_ROOT/.local/weights/a100/mha/g2-checkpoints}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-/home/work/.cache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6}"

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

run_one() {
    local exp="$1"
    local run_id
    local weights
    local exp_name
    local port

    case "$exp" in
        exp1)
            run_id="g2-exp1-phase3-router-only-retune-wikicode-from-all-experts-router-no-reinit-mb72-1800"
            exp_name="G2 - exp1 Phase 3 router-only retune wiki+code"
            port="${MASTER_PORT_EXP1:-29761}"
            ;;
        exp2)
            run_id="g2-exp2-phase3-router-only-retune-wikicode-from-new-experts-all-router-no-reinit-mb72-1800"
            exp_name="G2 - exp2 Phase 3 router-only retune wiki+code"
            port="${MASTER_PORT_EXP2:-29762}"
            ;;
        *)
            echo "[ERROR] unknown exp: $exp" >&2
            exit 1
            ;;
    esac

    weights="$G2_ROOT/code/phase3/$run_id"
    if [ ! -f "$weights/latest_checkpointed_iteration.txt" ]; then
        echo "[ERROR] missing Phase 3 copy: $weights" >&2
        echo "Run: scripts/experiment/a100/prepare_g2_phase3_copy.sh $exp no-reinit" >&2
        exit 1
    fi

    local base_step
    if [ -f "$weights/PHASE3_SOURCE.txt" ]; then
        base_step="$(grep -E '^source_step=' "$weights/PHASE3_SOURCE.txt" | tail -1 | cut -d= -f2- || true)"
    fi
    base_step="${base_step:-$(tr -d '\n\r[:space:]' < "$weights/latest_checkpointed_iteration.txt")}"

    local latest_step
    latest_step="$(tr -d '\n\r[:space:]' < "$weights/latest_checkpointed_iteration.txt")"
    local target_step=$((base_step + RETUNE_ITERS))
    if [ "$latest_step" -ge "$target_step" ]; then
        echo "[SKIP] already at or past target: $weights latest=$latest_step target=$target_step"
        return
    fi

    echo "[CONFIG] G2 $exp Phase 3 router-only retune"
    echo "[CONFIG] base_step=${base_step}, latest_step=${latest_step}, retune_iters=${RETUNE_ITERS}, target_step=${target_step}"
    echo "[CONFIG] data=wiki train + code train, loss=LM loss, trainable=shared router only"
    echo "[CONFIG] weights=${weights}"

    env \
        RUN_ID="$run_id" \
        TRAIN_WEIGHTS="$weights" \
        SOURCE_STEP="$base_step" \
        TRAIN_ITERS="$target_step" \
        WANDB_EXP_NAME="$exp_name" \
        WANDB_RUN_ID="$run_id" \
        MASTER_PORT="$port" \
        bash "$SCRIPT_DIR/run_guarded_training.sh" \
            bash "$SCRIPT_DIR/phase3_router_only_retune_shared_router_hybrid_mixed_local_bf16.sh"
}

if [ "$TARGET" = "all" ]; then
    run_one exp1
    run_one exp2
else
    run_one "$TARGET"
fi
