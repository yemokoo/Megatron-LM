#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_LOG_CHECKPOINTS="${WANDB_LOG_CHECKPOINTS:-0}"
export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
export SEED="${SEED:-1234}"
export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1800}"
export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-1}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-$TRAIN_ITERS}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}"
export TERTIARY_PROBE_EVAL_INTERVAL="${TERTIARY_PROBE_EVAL_INTERVAL:-50}"
export OLD_MODEL_KL_TEMPERATURE="${OLD_MODEL_KL_TEMPERATURE:-1.0}"
export PAUSE_SECONDS="${PAUSE_SECONDS:-180}"
export DUPLICATE_WIKI_PER_KL="${DUPLICATE_WIKI_PER_KL:-0}"
export RUN_ONLY_VARIANT="${RUN_ONLY_VARIANT:-all}"
export KD_MODES="${KD_MODES:-kl1 kl0}"

export ROOT_WEIGHTS="${ROOT_WEIGHTS:-$PROJECT_ROOT/.local/weights/a100/mha/g2-wikicode-baseline-kl-sweep}"

export DENSE16_MICRO_BATCH_SIZE="${DENSE16_MICRO_BATCH_SIZE:-72}"
export FIXED16_MICRO_BATCH_SIZE="${FIXED16_MICRO_BATCH_SIZE:-72}"
export EXPANDED_SHARED_MICRO_BATCH_SIZE="${EXPANDED_SHARED_MICRO_BATCH_SIZE:-48}"

export NUM_LAYERS="${NUM_LAYERS:-9}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-5472}"
export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-8}"
export NUM_EXPERTS="${NUM_EXPERTS:-16}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_LAYER_FREQ="${MOE_LAYER_FREQ:-[0,1,1,1,1,1,1,1,1]}"
export ATTN_LORA_RANK="${ATTN_LORA_RANK:-256}"
export ATTN_LORA_ALPHA="${ATTN_LORA_ALPHA:-$ATTN_LORA_RANK}"
export ATTN_FULL_RANK_LORA_RANK="${ATTN_FULL_RANK_LORA_RANK:-256}"
export ATTN_FULL_RANK_LORA_ALPHA="${ATTN_FULL_RANK_LORA_ALPHA:-$ATTN_FULL_RANK_LORA_RANK}"
export ATTN_FULL_RANK_LORA_TARGETS="${ATTN_FULL_RANK_LORA_TARGETS:-qkvo}"
export ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="${ATTN_FULL_RANK_LORA_ACTIVE_TARGETS:-}"
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"
export ATTN_LORA_GROUPED_GEMM="${ATTN_LORA_GROUPED_GEMM:-1}"
export MOE_ROUTER_DTYPE="${MOE_ROUTER_DTYPE:-fp32}"

dataset_path_for_task() {
    local task="$1"
    bash -lc "source scripts/experiment/a100/common.sh; dataset_dir_for_task $task"
}

probe_path_for_task() {
    local task="$1"
    bash -lc "source scripts/experiment/a100/common.sh; probe_dir_for_task $task"
}

export WIKI_TRAIN_DATASET="${WIKI_TRAIN_DATASET:-$(dataset_path_for_task wiki)}"
export CODE_TRAIN_DATASET="${CODE_TRAIN_DATASET:-$(dataset_path_for_task code)}"
export WIKI_PROBE_DATASET="${WIKI_PROBE_DATASET:-$(probe_path_for_task wiki)}"
export CODE_PROBE_DATASET="${CODE_PROBE_DATASET:-$(probe_path_for_task code)}"
export CONVERSATION_PROBE_DATASET="${CONVERSATION_PROBE_DATASET:-$(probe_path_for_task conversation)}"

is_completed() {
    local run_dir="$1"
    local expected="${2:-$TRAIN_ITERS}"
    local latest_file="$run_dir/latest_checkpointed_iteration.txt"
    [ -f "$latest_file" ] && [ "$(tr -d '\n\r[:space:]' < "$latest_file")" = "$expected" ]
}

pause_between_stages() {
    if [ "$PAUSE_SECONDS" -gt 0 ]; then
        echo "[PAUSE] ${PAUSE_SECONDS}s"
        sleep "$PAUSE_SECONDS"
    fi
}

should_run_variant() {
    local variant="$1"
    [ "$RUN_ONLY_VARIANT" = "all" ] || [ "$RUN_ONLY_VARIANT" = "$variant" ]
}

kl_coeff_for_tag() {
    case "$1" in
        kl1) echo "1.0" ;;
        kl0) echo "0.0" ;;
        *)
            echo "ERROR: unknown KD mode '$1' (expected kl1 or kl0)" >&2
            return 1
            ;;
    esac
}

wiki_suffix_for_kl() {
    local kl_tag="$1"
    if [ "$DUPLICATE_WIKI_PER_KL" = "1" ]; then
        echo "wiki_${kl_tag}"
    else
        echo "wiki"
    fi
}

run_dense16() {
    local kl_tag="$1"
    local kl_coeff="$2"
    local wiki_slot
    wiki_slot="$(wiki_suffix_for_kl "$kl_tag")"
    local wiki_id="g2-dense16-ffn5632-wiki-mha-a100-bf16-mb${DENSE16_MICRO_BATCH_SIZE}-${TRAIN_ITERS}"
    local code_id="g2-dense16-ffn5632-wiki-to-code-${kl_tag}-mha-a100-bf16-mb${DENSE16_MICRO_BATCH_SIZE}-${TRAIN_ITERS}"
    local wiki_dir="$ROOT_WEIGHTS/dense16/$wiki_slot/$wiki_id"
    local code_dir="$ROOT_WEIGHTS/dense16/code/$code_id"

    echo "================================================================================"
    echo "[DENSE16] kl=${kl_coeff} wiki=$wiki_dir"
    echo "================================================================================"
    if is_completed "$wiki_dir"; then
        echo "[SKIP] dense16 wiki already complete"
    else
        env \
            WANDB_MODE="$WANDB_MODE" \
            RUN_ID="$wiki_id" \
            TRAIN_WEIGHTS="$wiki_dir" \
            TRAIN_DATASET="$WIKI_TRAIN_DATASET" \
            WANDB_PROJECT="$WANDB_PROJECT" \
            WANDB_EXP_NAME="G2 dense16 ffn5632 wiki" \
            FFN_HIDDEN_SIZE=5632 \
            MICRO_BATCH_SIZE="$DENSE16_MICRO_BATCH_SIZE" \
            GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
            TRAIN_ITERS="$TRAIN_ITERS" \
            SAVE_INTERVAL="$SAVE_INTERVAL" \
            EVAL_INTERVAL="$EVAL_INTERVAL" \
            LOG_INTERVAL="$LOG_INTERVAL" \
            SEED="$SEED" \
            PROBE_DATASET="$WIKI_PROBE_DATASET" \
            PROBE_EVAL_INTERVAL="$PROBE_EVAL_INTERVAL" \
            SECONDARY_PROBE_DATASET="$CODE_PROBE_DATASET" \
            SECONDARY_PROBE_NAME=code_probe \
            SECONDARY_PROBE_EVAL_INTERVAL="$SECONDARY_PROBE_EVAL_INTERVAL" \
            TERTIARY_PROBE_DATASET="$CONVERSATION_PROBE_DATASET" \
            TERTIARY_PROBE_NAME=conversation_probe \
            TERTIARY_PROBE_EVAL_INTERVAL="$TERTIARY_PROBE_EVAL_INTERVAL" \
            MASTER_PORT="${DENSE16_WIKI_MASTER_PORT:-29871}" \
            bash scripts/experiment/a100/pretrain_wiki_dense_local_bf16.sh
        pause_between_stages
    fi

    if is_completed "$code_dir"; then
        echo "[SKIP] dense16 code ${kl_tag} already complete"
    else
        env \
            WANDB_MODE="$WANDB_MODE" \
            RUN_ID="$code_id" \
            TRAIN_WEIGHTS="$code_dir" \
            STAGE1_WEIGHTS_DIR="$wiki_dir" \
            TRAIN_DATASET="$CODE_TRAIN_DATASET" \
            WANDB_PROJECT="$WANDB_PROJECT" \
            WANDB_EXP_NAME="G2 dense16 ffn5632 wiki-to-code ${kl_tag}" \
            METADATA_STAGE="code_from_wiki_dense16_full_finetune" \
            FFN_HIDDEN_SIZE=5632 \
            MICRO_BATCH_SIZE="$DENSE16_MICRO_BATCH_SIZE" \
            GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
            TRAIN_ITERS="$TRAIN_ITERS" \
            SAVE_INTERVAL="$SAVE_INTERVAL" \
            EVAL_INTERVAL="$EVAL_INTERVAL" \
            LOG_INTERVAL="$LOG_INTERVAL" \
            SEED="$SEED" \
            OLD_MODEL_KL_COEFF="$kl_coeff" \
            OLD_MODEL_KL_TEMPERATURE="$OLD_MODEL_KL_TEMPERATURE" \
            PROBE_DATASET="$CODE_PROBE_DATASET" \
            PROBE_EVAL_INTERVAL="$PROBE_EVAL_INTERVAL" \
            SECONDARY_PROBE_DATASET="$WIKI_PROBE_DATASET" \
            SECONDARY_PROBE_EVAL_INTERVAL="$SECONDARY_PROBE_EVAL_INTERVAL" \
            TERTIARY_PROBE_DATASET="$CONVERSATION_PROBE_DATASET" \
            TERTIARY_PROBE_NAME=conversation_probe \
            TERTIARY_PROBE_EVAL_INTERVAL="$TERTIARY_PROBE_EVAL_INTERVAL" \
            MASTER_PORT="${DENSE16_CODE_MASTER_PORT:-29872}" \
            bash scripts/experiment/continual_code_from_wiki_dense_local_bf16.sh
        pause_between_stages
    fi
}

run_fixed16() {
    local kl_tag="$1"
    local kl_coeff="$2"
    local wiki_slot
    wiki_slot="$(wiki_suffix_for_kl "$kl_tag")"
    local wiki_id="g2-fixed16-top4-e16-ffn352-r256-wiki-shared-router-qkvo-mha-a100-bf16-mb${FIXED16_MICRO_BATCH_SIZE}-${TRAIN_ITERS}"
    local code_id="g2-fixed16-top4-e16-ffn352-r256-wiki-to-code-fullfinetune-${kl_tag}-mha-a100-bf16-mb${FIXED16_MICRO_BATCH_SIZE}-${TRAIN_ITERS}"
    local wiki_dir="$ROOT_WEIGHTS/fixed16/$wiki_slot/$wiki_id"
    local code_dir="$ROOT_WEIGHTS/fixed16/code/$code_id"

    echo "================================================================================"
    echo "[FIXED16] kl=${kl_coeff} wiki=$wiki_dir"
    echo "================================================================================"
    if is_completed "$wiki_dir"; then
        echo "[SKIP] fixed16 wiki already complete"
    else
        env \
            WANDB_MODE="$WANDB_MODE" \
            RUN_ID="$wiki_id" \
            TRAIN_WEIGHTS="$wiki_dir" \
            WANDB_PROJECT="$WANDB_PROJECT" \
            WANDB_EXP_NAME="G2 fixed16 shared-router wiki" \
            NUM_EXPERTS=16 \
            MOE_FFN_HIDDEN_SIZE="$MOE_FFN_HIDDEN_SIZE" \
            MOE_ROUTER_TOPK="$MOE_ROUTER_TOPK" \
            ATTN_LORA_RANK="$ATTN_LORA_RANK" \
            ATTN_LORA_ALPHA="$ATTN_LORA_ALPHA" \
            ATTN_FULL_RANK_LORA_RANK="$ATTN_FULL_RANK_LORA_RANK" \
            ATTN_FULL_RANK_LORA_ALPHA="$ATTN_FULL_RANK_LORA_ALPHA" \
            ATTN_FULL_RANK_LORA_TARGETS="$ATTN_FULL_RANK_LORA_TARGETS" \
            ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="$ATTN_FULL_RANK_LORA_ACTIVE_TARGETS" \
            MOE_GROUPED_GEMM="$MOE_GROUPED_GEMM" \
            ATTN_LORA_GROUPED_GEMM="$ATTN_LORA_GROUPED_GEMM" \
            MOE_ROUTER_DTYPE="$MOE_ROUTER_DTYPE" \
            MICRO_BATCH_SIZE="$FIXED16_MICRO_BATCH_SIZE" \
            GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
            TRAIN_ITERS="$TRAIN_ITERS" \
            SAVE_INTERVAL="$SAVE_INTERVAL" \
            EVAL_INTERVAL="$EVAL_INTERVAL" \
            LOG_INTERVAL="$LOG_INTERVAL" \
            SEED="$SEED" \
            PROBE_EVAL_INTERVAL="$PROBE_EVAL_INTERVAL" \
            SECONDARY_PROBE_EVAL_INTERVAL="$SECONDARY_PROBE_EVAL_INTERVAL" \
            MASTER_PORT="${FIXED16_WIKI_MASTER_PORT:-29873}" \
            bash scripts/experiment/a100/pretrain_wiki_shared_router_hybrid_local_bf16.sh
        pause_between_stages
    fi

    if is_completed "$code_dir"; then
        echo "[SKIP] fixed16 code ${kl_tag} already complete"
    else
        env \
            WANDB_MODE="$WANDB_MODE" \
            RUN_ID="$code_id" \
            TRAIN_WEIGHTS="$code_dir" \
            STAGE1_WEIGHTS_DIR="$wiki_dir" \
            TRAIN_DATASET="$CODE_TRAIN_DATASET" \
            RUN_LOG="$code_dir/logs/code_from_wiki_shared_router_hybrid_full_finetune.log" \
            WANDB_PROJECT="$WANDB_PROJECT" \
            WANDB_EXP_NAME="G2 fixed16 shared-router wiki-to-code full finetune ${kl_tag}" \
            MODEL_CONFIG_SCRIPT=configs/model/flame-shared-router-hybrid-experts.sh \
            METADATA_STAGE="code_from_wiki_fixed16_shared_router_full_finetune" \
            NUM_EXPERTS=16 \
            MOE_FFN_HIDDEN_SIZE="$MOE_FFN_HIDDEN_SIZE" \
            MOE_ROUTER_TOPK="$MOE_ROUTER_TOPK" \
            ATTN_LORA_RANK="$ATTN_LORA_RANK" \
            ATTN_LORA_ALPHA="$ATTN_LORA_ALPHA" \
            ATTN_FULL_RANK_LORA_RANK="$ATTN_FULL_RANK_LORA_RANK" \
            ATTN_FULL_RANK_LORA_ALPHA="$ATTN_FULL_RANK_LORA_ALPHA" \
            ATTN_FULL_RANK_LORA_TARGETS="$ATTN_FULL_RANK_LORA_TARGETS" \
            ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="$ATTN_FULL_RANK_LORA_ACTIVE_TARGETS" \
            MOE_GROUPED_GEMM="$MOE_GROUPED_GEMM" \
            ATTN_LORA_GROUPED_GEMM="$ATTN_LORA_GROUPED_GEMM" \
            MOE_ROUTER_DTYPE="$MOE_ROUTER_DTYPE" \
            MICRO_BATCH_SIZE="$FIXED16_MICRO_BATCH_SIZE" \
            GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
            TRAIN_ITERS="$TRAIN_ITERS" \
            SAVE_INTERVAL="$SAVE_INTERVAL" \
            EVAL_INTERVAL="$EVAL_INTERVAL" \
            LOG_INTERVAL="$LOG_INTERVAL" \
            SEED="$SEED" \
            OLD_MODEL_KL_COEFF="$kl_coeff" \
            OLD_MODEL_KL_TEMPERATURE="$OLD_MODEL_KL_TEMPERATURE" \
            PROBE_DATASET="$CODE_PROBE_DATASET" \
            PROBE_EVAL_INTERVAL="$PROBE_EVAL_INTERVAL" \
            SECONDARY_PROBE_DATASET="$WIKI_PROBE_DATASET" \
            SECONDARY_PROBE_EVAL_INTERVAL="$SECONDARY_PROBE_EVAL_INTERVAL" \
            MASTER_PORT="${FIXED16_CODE_MASTER_PORT:-29874}" \
            bash scripts/experiment/continual_code_from_wiki_dense_local_bf16.sh
        pause_between_stages
    fi
}

run_expanded_oldfreeze_sharedunfreeze() {
    local kl_tag="$1"
    local kl_coeff="$2"
    local wiki_slot
    wiki_slot="$(wiki_suffix_for_kl "$kl_tag")"
    local wiki_id="g2-expanded-oldfreeze-sharedunfreeze-source-e8-ffn352-r256-wiki-shared-router-qkvo-mha-a100-bf16-mb${EXPANDED_SHARED_MICRO_BATCH_SIZE}-${TRAIN_ITERS}"
    local code_id="g2-expanded-oldfreeze-sharedunfreeze-e8to16-ffn352-r256-wiki-to-code-${kl_tag}-mha-a100-bf16-mb${EXPANDED_SHARED_MICRO_BATCH_SIZE}-${TRAIN_ITERS}"
    local wiki_dir="$ROOT_WEIGHTS/expanded_oldfreeze_sharedunfreeze/$wiki_slot/$wiki_id"
    local code_dir="$ROOT_WEIGHTS/expanded_oldfreeze_sharedunfreeze/code/$code_id"

    echo "================================================================================"
    echo "[EXPANDED_OLDFREEZE_SHAREDUNFREEZE] kl=${kl_coeff} wiki=$wiki_dir"
    echo "================================================================================"
    if is_completed "$wiki_dir"; then
        echo "[SKIP] expanded source wiki already complete"
    else
        env \
            WANDB_MODE="$WANDB_MODE" \
            RUN_ID="$wiki_id" \
            TRAIN_WEIGHTS="$wiki_dir" \
            WANDB_PROJECT="$WANDB_PROJECT" \
            WANDB_EXP_NAME="G2 expanded old-freeze source e8 wiki" \
            NUM_EXPERTS=8 \
            MOE_FFN_HIDDEN_SIZE="$MOE_FFN_HIDDEN_SIZE" \
            MOE_ROUTER_TOPK="$MOE_ROUTER_TOPK" \
            ATTN_LORA_RANK="$ATTN_LORA_RANK" \
            ATTN_LORA_ALPHA="$ATTN_LORA_ALPHA" \
            ATTN_FULL_RANK_LORA_RANK="$ATTN_FULL_RANK_LORA_RANK" \
            ATTN_FULL_RANK_LORA_ALPHA="$ATTN_FULL_RANK_LORA_ALPHA" \
            ATTN_FULL_RANK_LORA_TARGETS="$ATTN_FULL_RANK_LORA_TARGETS" \
            ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="$ATTN_FULL_RANK_LORA_ACTIVE_TARGETS" \
            MOE_GROUPED_GEMM="$MOE_GROUPED_GEMM" \
            ATTN_LORA_GROUPED_GEMM="$ATTN_LORA_GROUPED_GEMM" \
            MOE_ROUTER_DTYPE="$MOE_ROUTER_DTYPE" \
            MICRO_BATCH_SIZE="$EXPANDED_SHARED_MICRO_BATCH_SIZE" \
            GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
            TRAIN_ITERS="$TRAIN_ITERS" \
            SAVE_INTERVAL="$SAVE_INTERVAL" \
            EVAL_INTERVAL="$EVAL_INTERVAL" \
            LOG_INTERVAL="$LOG_INTERVAL" \
            SEED="$SEED" \
            PROBE_EVAL_INTERVAL="$PROBE_EVAL_INTERVAL" \
            SECONDARY_PROBE_EVAL_INTERVAL="$SECONDARY_PROBE_EVAL_INTERVAL" \
            MASTER_PORT="${EXPANDED_WIKI_MASTER_PORT:-29875}" \
            bash scripts/experiment/a100/pretrain_wiki_shared_router_hybrid_local_bf16.sh
        pause_between_stages
    fi

    if is_completed "$code_dir"; then
        echo "[SKIP] expanded old-freeze shared-unfreeze code ${kl_tag} already complete"
    else
        env \
            WANDB_MODE="$WANDB_MODE" \
            RUN_ID="$code_id" \
            TRAIN_WEIGHTS="$code_dir" \
            STAGE1_WEIGHTS_DIR="$wiki_dir" \
            WANDB_PROJECT="$WANDB_PROJECT" \
            WANDB_EXP_NAME="G2 expanded old-freeze shared-unfreeze wiki-to-code ${kl_tag}" \
            SOURCE_NUM_EXPERTS=8 \
            NUM_EXPERTS=16 \
            MOE_FFN_HIDDEN_SIZE="$MOE_FFN_HIDDEN_SIZE" \
            MOE_ROUTER_TOPK="$MOE_ROUTER_TOPK" \
            ATTN_LORA_RANK="$ATTN_LORA_RANK" \
            ATTN_LORA_ALPHA="$ATTN_LORA_ALPHA" \
            ATTN_FULL_RANK_LORA_RANK="$ATTN_FULL_RANK_LORA_RANK" \
            ATTN_FULL_RANK_LORA_ALPHA="$ATTN_FULL_RANK_LORA_ALPHA" \
            ATTN_FULL_RANK_LORA_TARGETS="$ATTN_FULL_RANK_LORA_TARGETS" \
            ATTN_FULL_RANK_LORA_ACTIVE_TARGETS="$ATTN_FULL_RANK_LORA_ACTIVE_TARGETS" \
            SHARED_ROUTER_HYBRID_FREEZE_PREEXISTING_ONLY=1 \
            SHARED_ROUTER_HYBRID_TRAIN_ALL_EXPERTS_AND_ROUTER=0 \
            SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS=0 \
            MOE_GROUPED_GEMM="$MOE_GROUPED_GEMM" \
            ATTN_LORA_GROUPED_GEMM="$ATTN_LORA_GROUPED_GEMM" \
            MOE_ROUTER_DTYPE="$MOE_ROUTER_DTYPE" \
            MICRO_BATCH_SIZE="$EXPANDED_SHARED_MICRO_BATCH_SIZE" \
            GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
            TRAIN_ITERS="$TRAIN_ITERS" \
            SAVE_INTERVAL="$SAVE_INTERVAL" \
            SAVE_CHECKPOINTS="$SAVE_CHECKPOINTS" \
            EVAL_INTERVAL="$EVAL_INTERVAL" \
            LOG_INTERVAL="$LOG_INTERVAL" \
            SEED="$SEED" \
            OLD_MODEL_KL_COEFF="$kl_coeff" \
            OLD_MODEL_KL_TEMPERATURE="$OLD_MODEL_KL_TEMPERATURE" \
            PROBE_EVAL_INTERVAL="$PROBE_EVAL_INTERVAL" \
            SECONDARY_PROBE_EVAL_INTERVAL="$SECONDARY_PROBE_EVAL_INTERVAL" \
            MASTER_PORT="${EXPANDED_CODE_MASTER_PORT:-29876}" \
            bash scripts/experiment/continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh
        pause_between_stages
    fi
}

echo "[CONFIG] G2 wiki->code baseline KL sweep"
echo "[CONFIG] variants: dense16, fixed16, expanded_oldfreeze_sharedunfreeze"
echo "[CONFIG] KD modes: $KD_MODES"
echo "[CONFIG] seed=$SEED train_iters=$TRAIN_ITERS gbs=$GLOBAL_BATCH_SIZE"
echo "[CONFIG] dense16 ffn_hidden=5632 mb=$DENSE16_MICRO_BATCH_SIZE"
echo "[CONFIG] fixed16 experts=16 ffn_expert_hidden=$MOE_FFN_HIDDEN_SIZE mb=$FIXED16_MICRO_BATCH_SIZE"
echo "[CONFIG] expanded 8->16 old experts/router frozen; shared trunk trainable; mb=$EXPANDED_SHARED_MICRO_BATCH_SIZE"
echo "[CONFIG] grouped_gemm moe=$MOE_GROUPED_GEMM attention_lora=$ATTN_LORA_GROUPED_GEMM"
echo "[CONFIG] duplicate_wiki_per_kl=$DUPLICATE_WIKI_PER_KL"

for kd_tag in $KD_MODES; do
    kl_coeff="$(kl_coeff_for_tag "$kd_tag")"
    if should_run_variant dense16; then
        run_dense16 "$kd_tag" "$kl_coeff"
    fi
    if should_run_variant fixed16; then
        run_fixed16 "$kd_tag" "$kl_coeff"
    fi
    if should_run_variant expanded_oldfreeze_sharedunfreeze; then
        run_expanded_oldfreeze_sharedunfreeze "$kd_tag" "$kl_coeff"
    fi
done

echo "[ALL DONE] G2 wiki->code baseline KL sweep $(date)"
