#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

usage() {
    cat >&2 <<'EOF'
usage: run_g2matched_phase3_router_only_retune_mha.sh <ffn-only|fullrank|all>

Runs offline Phase 3 router-only retuning for the two completed G2matched baselines:
  ffn-only : FFN-MoE experts only, attention frozen
  fullrank : FFN-MoE experts + attention full-rank LoRA qkvo r1024

Each run copies the completed baseline checkpoint into g2-checkpoints/code/phase3
before retuning, so the original baseline checkpoints are not modified.
EOF
}

if [ "$#" -ne 1 ]; then
    usage
    exit 1
fi

TARGET="$1"
case "$TARGET" in
    ffn-only|attn-freeze|fullrank|all)
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
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-96}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export G2_ROOT="${G2_ROOT:-$PROJECT_ROOT/.local/weights/a100/mha/g2-checkpoints}"
export BASE_WEIGHTS_DIR="${BASE_WEIGHTS_DIR:-$PROJECT_ROOT/.local/weights/a100/mha}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-/home/work/.cache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6}"

export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-8}"
export NUM_EXPERTS="${NUM_EXPERTS:-16}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.0}"
export MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.0}"
export MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}"

PHASE3_ROOT="${PHASE3_ROOT:-$G2_ROOT/code/phase3}"
mkdir -p "$PHASE3_ROOT"

tracker_value() {
    tr -d '\n\r[:space:]' < "$1/latest_checkpointed_iteration.txt"
}

find_completed_source() {
    local run_id="$1"
    local stage_dir="$2"
    local registry_candidate="$G2_ROOT/code/g2matched/$run_id"
    local original_candidate="$BASE_WEIGHTS_DIR/$stage_dir/$run_id"

    if [ -f "$registry_candidate/latest_checkpointed_iteration.txt" ]; then
        echo "$registry_candidate"
        return
    fi
    if [ -f "$original_candidate/latest_checkpointed_iteration.txt" ]; then
        echo "$original_candidate"
        return
    fi

    echo "[ERROR] completed source not found for $run_id" >&2
    echo "Checked:" >&2
    echo "  $registry_candidate" >&2
    echo "  $original_candidate" >&2
    exit 1
}

ensure_copy() {
    local src="$1"
    local dst="$2"
    local label="$3"
    local baseline="$4"

    if [ -e "$dst" ]; then
        return
    fi

    local src_step
    src_step="$(tracker_value "$src")"
    if [ "$src_step" != "1800" ]; then
        echo "[ERROR] source is not complete. expected 1800, got $src_step: $src" >&2
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
        echo "purpose=g2matched router-only retuning copy"
        echo "baseline=$baseline"
        echo "source_label=$label"
        echo "source=$src"
        echo "source_step=$src_step"
        echo "router_init=no-reinit"
        echo "copied_at=$(date -Is)"
    } > "$dst/PHASE3_SOURCE.txt"
    echo "[DONE] $dst"
}

run_one() {
    local baseline="$1"
    local source_run_id
    local source_stage_dir
    local run_id
    local exp_name
    local model_config
    local attn_rank
    local port

    case "$baseline" in
        ffn-only|attn-freeze)
            baseline="ffn-only"
            source_run_id="g2matched-top4-e8to16-ffn352-wiki-to-code-ffn-moe-attn-freeze-mha-a100-bf16-mb96-1800"
            source_stage_dir="g2matched-ffn-moe-attn-freeze-bf16"
            run_id="g2matched-attn-freeze-phase3-router-only-retune-wikicode-no-reinit-mb96-1800"
            exp_name="G2matched - FFN-only baseline Phase 3 router-only retune wiki+code"
            model_config="scripts/experiment/a100/flame-moe-bf16-no-shared.sh"
            attn_rank=0
            port="${MASTER_PORT_FFN_ONLY:-29771}"
            ;;
        fullrank)
            source_run_id="g2matched-top4-e8to16-ffn352-wiki-to-code-ffn-moe-attn-fullrank-qkvo-r1024-mha-a100-bf16-mb96-1800"
            source_stage_dir="g2matched-ffn-moe-attn-full-rank-lora-bf16"
            run_id="g2matched-attn-fullrank-qkvo-r1024-phase3-router-only-retune-wikicode-no-reinit-mb96-1800"
            exp_name="G2matched - FFN + attention full-rank LoRA baseline Phase 3 router-only retune wiki+code"
            model_config="configs/model/flame-attn-full-rank-lora.sh"
            attn_rank=1024
            port="${MASTER_PORT_FULLRANK:-29772}"
            ;;
        *)
            echo "[ERROR] unknown baseline: $baseline" >&2
            exit 1
            ;;
    esac

    local source
    source="$(find_completed_source "$source_run_id" "$source_stage_dir")"
    local weights="$PHASE3_ROOT/$run_id"
    ensure_copy "$source" "$weights" "$exp_name source" "$baseline"

    if [ ! -f "$weights/latest_checkpointed_iteration.txt" ]; then
        echo "[ERROR] missing checkpoint copy tracker: $weights/latest_checkpointed_iteration.txt" >&2
        exit 1
    fi

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

    echo "[CONFIG] G2matched $baseline Phase 3 router-only retune"
    echo "[CONFIG] base_step=${base_step}, latest_step=${latest_step}, retune_iters=${RETUNE_ITERS}, target_step=${target_step}"
    echo "[CONFIG] data=wiki train + code train, loss=LM loss, trainable=MoE router only"
    echo "[CONFIG] wandb_mode=${WANDB_MODE}, weights=${weights}"

    env \
        RUN_ID="$run_id" \
        TRAIN_WEIGHTS="$weights" \
        SOURCE_STEP="$base_step" \
        TRAIN_ITERS="$target_step" \
        WANDB_EXP_NAME="$exp_name" \
        WANDB_RUN_ID="$run_id" \
        MASTER_PORT="$port" \
        MODEL_CONFIG_SCRIPT="$model_config" \
        ATTN_FULL_RANK_LORA_RANK="$attn_rank" \
        ATTN_FULL_RANK_LORA_ALPHA="$attn_rank" \
        ATTN_FULL_RANK_LORA_TARGETS=qkvo \
        ATTN_FULL_RANK_LORA_ACTIVE_TARGETS= \
        bash "$SCRIPT_DIR/run_guarded_training.sh" \
            bash "$SCRIPT_DIR/phase3_router_only_retune_moe_mixed_local_bf16.sh"
}

if [ "$TARGET" = "all" ]; then
    run_one ffn-only
    run_one fullrank
else
    run_one "$TARGET"
fi
