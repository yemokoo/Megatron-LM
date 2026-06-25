#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

usage() {
    cat >&2 <<'EOF'
usage: run_g2_router_finetune_miniset_repeats_mha.sh <ffn-only|ffn-attn|attn-only|all> <0p01pctx2000|0p1pctx200|1pctx20|5pctx4|10pctx2|all>

Runs router-only retuning from the code-trained checkpoint, using fixed small
wiki+code router-finetune subsets:
  1pctx20: 1% fixed subset repeated for 20 epochs, 720 retune steps total
  5pctx4 : 5% fixed subset repeated for 4 epochs, 720 retune steps total
  10pctx2: 10% fixed subset repeated for 2 epochs, 720 retune steps total
  0p1pctx200: 0.1% fixed subset repeated for 200 epochs, 720 retune steps total
  0p01pctx2000: 0.01% fixed subset repeated for 2000 epochs, 720 retune steps total

The subset conditions are independent: each copies the code-trained source
checkpoint before retuning, so one miniset condition never continues from another.
EOF
}

if [ "$#" -ne 2 ]; then
    usage
    exit 1
fi

TARGET="$1"
SPEC_TARGET="$2"
case "$TARGET" in
    ffn-only|ffn-attn|attn-only|all) ;;
    *) usage; exit 1 ;;
esac
case "$SPEC_TARGET" in
    0p01pctx2000|0p1pctx200|1pctx20|5pctx4|10pctx2|all) ;;
    *) usage; exit 1 ;;
esac

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
export G2_ROOT="${G2_ROOT:-$LOCAL_WEIGHTS/a100/mha/g2-checkpoints}"
export BASE_WEIGHTS_DIR="${BASE_WEIGHTS_DIR:-$PROJECT_ROOT/.local/weights/a100/mha}"

export FIXED_DATA_SEED="${FIXED_DATA_SEED:-1234}"
export MINISET_ROOT_BASE="${MINISET_ROOT_BASE:-$LOCAL_BASE/datasets/router_finetune_miniset_repeats/seed${FIXED_DATA_SEED}}"
export PHASE3_REPEAT_ROOT="${PHASE3_REPEAT_ROOT:-$G2_ROOT/code/phase3_miniset_repeats}"

export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_LOG_CHECKPOINTS="${WANDB_LOG_CHECKPOINTS:-0}"
export RETUNE_ITERS="${RETUNE_ITERS:-720}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-36}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-36}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-36}"
export PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-25}"
export SECONDARY_PROBE_EVAL_ITERS="${SECONDARY_PROBE_EVAL_ITERS:-25}"
export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-1}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export TRAIN_ROUTER_USAGE_LOG_INTERVAL="${TRAIN_ROUTER_USAGE_LOG_INTERVAL:-0}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"

export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-8}"
export NUM_EXPERTS="${NUM_EXPERTS:-16}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.0}"
export MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.0}"
export MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}"

export FFN_ONLY_SOURCE_RUN_ID="${FFN_ONLY_SOURCE_RUN_ID:-g2matched-top4-e8to16-ffn352-wiki-to-code-ffn-moe-attn-freeze-mha-a100-bf16-mb96-1800}"
export FFN_ONLY_SOURCE_STAGE_DIR="${FFN_ONLY_SOURCE_STAGE_DIR:-g2matched-ffn-moe-attn-freeze-bf16}"
export FFN_ATTN_SOURCE_WEIGHTS="${FFN_ATTN_SOURCE_WEIGHTS:-$G2_ROOT/code/phase1/g2-exp2-top4-e8to16-ffn352-r256-wiki-to-code-new-experts-all-router-mha-a100-bf16-mb72-1800}"
export ATTN_ONLY_SOURCE_WEIGHTS="${ATTN_ONLY_SOURCE_WEIGHTS:-$G2_ROOT/code/phase1/g2-attn-only-dense5632-top4-e8to16-r256-wiki-to-code-qkvo-mha-a100-bf16-mb72-1800}"

mkdir -p "$PHASE3_REPEAT_ROOT"

if ps -ef | grep -E 'pretrain_gpt.py|torchrun' | grep -v grep >/dev/null; then
    echo "[ERROR] another training process is running"
    ps -ef | grep -E 'pretrain_gpt.py|torchrun' | grep -v grep || true
    exit 1
fi

tracker_value() {
    tr -d '\n\r[:space:]' < "$1/latest_checkpointed_iteration.txt"
}

find_ffn_only_source() {
    local registry_candidate="$G2_ROOT/code/g2matched/$FFN_ONLY_SOURCE_RUN_ID"
    local original_candidate="$BASE_WEIGHTS_DIR/$FFN_ONLY_SOURCE_STAGE_DIR/$FFN_ONLY_SOURCE_RUN_ID"

    if [ -f "$registry_candidate/latest_checkpointed_iteration.txt" ]; then
        echo "$registry_candidate"
        return
    fi
    if [ -f "$original_candidate/latest_checkpointed_iteration.txt" ]; then
        echo "$original_candidate"
        return
    fi

    echo "[ERROR] completed FFN-only source not found." >&2
    echo "Checked:" >&2
    echo "  $registry_candidate" >&2
    echo "  $original_candidate" >&2
    exit 1
}

ensure_source_copy() {
    local src="$1"
    local dst="$2"
    local label="$3"
    local spec_label="$4"

    if [ -f "$dst/latest_checkpointed_iteration.txt" ]; then
        return 0
    fi
    if [ -e "$dst" ]; then
        echo "[ERROR] destination exists but has no tracker: $dst" >&2
        exit 1
    fi
    if [ ! -f "$src/latest_checkpointed_iteration.txt" ]; then
        echo "[ERROR] source checkpoint tracker missing: $src/latest_checkpointed_iteration.txt" >&2
        exit 1
    fi

    local src_step
    src_step="$(tracker_value "$src")"
    if [ "$src_step" != "1800" ]; then
        echo "[ERROR] expected code-trained source at 1800, got $src_step: $src" >&2
        exit 1
    fi

    echo "[COPY] source:      $src"
    echo "[COPY] destination: $dst"
    mkdir -p "$dst"
    rsync -aH --info=progress2 \
        --exclude 'wandb/' \
        --exclude 'events.out.tfevents*' \
        "$src/" "$dst/"

    {
        echo "phase=3_miniset_repeat_router_finetune"
        echo "purpose=router-only retune from code-trained checkpoint using fixed small repeated wiki+code subset"
        echo "variant=$label"
        echo "miniset=$spec_label"
        echo "source=$src"
        echo "source_step=$src_step"
        echo "copied_at=$(date -Iseconds)"
    } > "$dst/PHASE3_SOURCE.txt"
}

spec_fraction() {
    case "$1" in
        0p01pctx2000) echo "0.0001" ;;
        0p1pctx200) echo "0.001" ;;
        1pctx20) echo "0.01" ;;
        5pctx4) echo "0.05" ;;
        10pctx2) echo "0.10" ;;
        *) echo "[ERROR] bad spec: $1" >&2; exit 1 ;;
    esac
}

spec_epochs() {
    case "$1" in
        0p01pctx2000) echo "2000" ;;
        0p1pctx200) echo "200" ;;
        1pctx20) echo "20" ;;
        5pctx4) echo "4" ;;
        10pctx2) echo "2" ;;
        *) echo "[ERROR] bad spec: $1" >&2; exit 1 ;;
    esac
}

run_one() {
    local variant="$1"
    local spec_label="$2"
    local dataset_root="$MINISET_ROOT_BASE/$spec_label"
    local env_file="$dataset_root/env.sh"

    if [ ! -f "$env_file" ]; then
        echo "[ERROR] missing fixed miniset env: $env_file" >&2
        echo "[HINT] create minisets first:" >&2
        echo "  bash scripts/experiment/a100/prepare_g2_router_finetune_miniset_repeats_mha.sh" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$env_file"

    local source_weights
    local run_id
    local train_weights
    local retune_script
    local exp_name
    local model_config
    local attn_rank
    local micro_batch_size
    local port
    local variant_label

    case "$variant" in
        ffn-only)
            variant_label="ffn_only"
            source_weights="$(find_ffn_only_source)"
            run_id="g2matched-ffn-only-phase3-router-only-retune-wikicode-miniset-${spec_label}-from-code-mb96-720"
            exp_name="G2matched FFN-only router finetune ${spec_label} repeated"
            retune_script="$SCRIPT_DIR/phase3_router_only_retune_moe_mixed_local_bf16.sh"
            model_config="scripts/experiment/a100/flame-moe-bf16-no-shared.sh"
            attn_rank=0
            micro_batch_size="${FFN_ONLY_MICRO_BATCH_SIZE:-96}"
            port="${MASTER_PORT_FFN_ONLY:-29851}"
            ;;
        ffn-attn)
            variant_label="ffn_attn"
            source_weights="$FFN_ATTN_SOURCE_WEIGHTS"
            run_id="g2-ffn-attn-phase3-router-only-retune-wikicode-miniset-${spec_label}-from-code-mb72-720"
            exp_name="G2 FFN+attention expert router finetune ${spec_label} repeated"
            retune_script="$SCRIPT_DIR/phase3_router_only_retune_shared_router_hybrid_mixed_local_bf16.sh"
            model_config=""
            attn_rank=256
            micro_batch_size="${FFN_ATTN_MICRO_BATCH_SIZE:-72}"
            port="${MASTER_PORT_FFN_ATTN:-29852}"
            ;;
        attn-only)
            variant_label="attn_only"
            source_weights="$ATTN_ONLY_SOURCE_WEIGHTS"
            run_id="g2-attn-only-dense5632-phase3-router-only-retune-wikicode-miniset-${spec_label}-from-code-mb72-720"
            exp_name="G2 attention-only expert router finetune ${spec_label} repeated"
            retune_script="$SCRIPT_DIR/phase3_router_only_retune_shared_router_hybrid_mixed_local_bf16.sh"
            model_config="configs/model/flame-attn-only-shared-router-qkvo-experts.sh"
            attn_rank=256
            micro_batch_size="${ATTN_ONLY_MICRO_BATCH_SIZE:-72}"
            port="${MASTER_PORT_ATTN_ONLY:-29853}"
            ;;
        *)
            echo "[ERROR] bad variant: $variant" >&2
            exit 1
            ;;
    esac

    train_weights="$PHASE3_REPEAT_ROOT/$run_id"
    ensure_source_copy "$source_weights" "$train_weights" "$variant_label" "$spec_label"

    local source_step
    source_step="$(grep -E '^source_step=' "$train_weights/PHASE3_SOURCE.txt" | tail -1 | cut -d= -f2-)"
    local target_step=$((source_step + RETUNE_ITERS))
    local latest_step
    latest_step="$(tracker_value "$train_weights")"

    if [ "$latest_step" -ge "$target_step" ]; then
        echo "[SKIP] already at or past target: variant=$variant spec=$spec_label latest=$latest_step target=$target_step"
        return
    fi

    local fraction
    local repeat_epochs
    fraction="$(spec_fraction "$spec_label")"
    repeat_epochs="$(spec_epochs "$spec_label")"

    echo "[CONFIG] router finetune miniset repeat"
    echo "[CONFIG] variant=$variant spec=$spec_label fraction=$fraction repeat_epochs=$repeat_epochs"
    echo "[CONFIG] source=$source_weights"
    echo "[CONFIG] source_step=$source_step latest_step=$latest_step target_step=$target_step retune_iters=$RETUNE_ITERS"
    echo "[CONFIG] train_weights=$train_weights"
    echo "[CONFIG] dataset_root=$dataset_root"
    echo "[CONFIG] trainable=router only | aux/z=${MOE_AUX_LOSS_COEFF}/${MOE_Z_LOSS_COEFF}"

    env_args=(
        WANDB_MODE="$WANDB_MODE"
        WANDB_PROJECT="$WANDB_PROJECT"
        WANDB_LOG_CHECKPOINTS="$WANDB_LOG_CHECKPOINTS"
        RUN_ID="$run_id"
        TRAIN_WEIGHTS="$train_weights"
        SOURCE_STEP="$source_step"
        TRAIN_ITERS="$target_step"
        RETUNE_ITERS="$RETUNE_ITERS"
        SAVE_INTERVAL="$SAVE_INTERVAL"
        PROBE_EVAL_INTERVAL="$PROBE_EVAL_INTERVAL"
        SECONDARY_PROBE_EVAL_INTERVAL="$SECONDARY_PROBE_EVAL_INTERVAL"
        PROBE_EVAL_ITERS="$PROBE_EVAL_ITERS"
        SECONDARY_PROBE_EVAL_ITERS="$SECONDARY_PROBE_EVAL_ITERS"
        TRAIN_ROUTER_USAGE_LOG_INTERVAL="$TRAIN_ROUTER_USAGE_LOG_INTERVAL"
        MICRO_BATCH_SIZE="$micro_batch_size"
        GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE"
        ROUTER_FINETUNE_DATASET_ROOT="$dataset_root"
        TRAIN_DATASET_WIKI="$dataset_root/wiki/train"
        TRAIN_DATASET_CODE="$dataset_root/code/train"
        DATASET_NAME="wiki_code_fixed_${spec_label}_seed${FIXED_DATA_SEED}"
        DATASET_SOURCE="Fixed ${spec_label} wiki/code train subsets repeated to 20pct budget; seed=${FIXED_DATA_SEED}; fraction=${fraction}; repeat_epochs=${repeat_epochs}"
        MOE_AUX_LOSS_COEFF="$MOE_AUX_LOSS_COEFF"
        MOE_Z_LOSS_COEFF="$MOE_Z_LOSS_COEFF"
        SOURCE_NUM_EXPERTS="$SOURCE_NUM_EXPERTS"
        NUM_EXPERTS="$NUM_EXPERTS"
        MOE_ROUTER_TOPK="$MOE_ROUTER_TOPK"
        MOE_FFN_HIDDEN_SIZE="$MOE_FFN_HIDDEN_SIZE"
        MOE_ROUTER_DTYPE="$MOE_ROUTER_DTYPE"
        WANDB_EXP_NAME="$exp_name"
        WANDB_RUN_ID="$run_id"
        MASTER_PORT="$port"
    )

    if [ "$variant" = "ffn-only" ]; then
        env "${env_args[@]}" \
            MODEL_CONFIG_SCRIPT="$model_config" \
            ATTN_FULL_RANK_LORA_RANK="$attn_rank" \
            ATTN_FULL_RANK_LORA_ALPHA="$attn_rank" \
            ATTN_FULL_RANK_LORA_TARGETS=qkvo \
            ATTN_FULL_RANK_LORA_ACTIVE_TARGETS= \
            bash "$SCRIPT_DIR/run_guarded_training.sh" bash "$retune_script"
    elif [ "$variant" = "attn-only" ]; then
        env "${env_args[@]}" \
            MODEL_CONFIG_SCRIPT="$model_config" \
            FFN_HIDDEN_SIZE="${ATTN_ONLY_FFN_HIDDEN_SIZE:-5632}" \
            ATTN_LORA_RANK="${ATTN_LORA_RANK:-256}" \
            ATTN_LORA_ALPHA="${ATTN_LORA_ALPHA:-256}" \
            ATTN_FULL_RANK_LORA_RANK="${ATTN_FULL_RANK_LORA_RANK:-256}" \
            ATTN_FULL_RANK_LORA_ALPHA="${ATTN_FULL_RANK_LORA_ALPHA:-256}" \
            ATTN_FULL_RANK_LORA_TARGETS="${ATTN_FULL_RANK_LORA_TARGETS:-qkvo}" \
            RESUME_FROM_NUM_EXPERTS="$SOURCE_NUM_EXPERTS" \
            bash "$SCRIPT_DIR/run_guarded_training.sh" bash "$retune_script"
    else
        env "${env_args[@]}" \
            ATTN_LORA_RANK="${ATTN_LORA_RANK:-256}" \
            ATTN_FULL_RANK_LORA_RANK="${ATTN_FULL_RANK_LORA_RANK:-256}" \
            ATTN_FULL_RANK_LORA_TARGETS="${ATTN_FULL_RANK_LORA_TARGETS:-qkvo}" \
            bash "$SCRIPT_DIR/run_guarded_training.sh" bash "$retune_script"
    fi
}

variants=()
specs=()
if [ "$TARGET" = "all" ]; then
    # Preserve historical behavior. Run `attn-only` explicitly after its
    # code-trained checkpoint has been produced.
    variants=(ffn-only ffn-attn)
else
    variants=("$TARGET")
fi
if [ "$SPEC_TARGET" = "all" ]; then
    specs=(0p01pctx2000 0p1pctx200 1pctx20 5pctx4 10pctx2)
else
    specs=("$SPEC_TARGET")
fi

for variant in "${variants[@]}"; do
    for spec in "${specs[@]}"; do
        run_one "$variant" "$spec"
    done
done
