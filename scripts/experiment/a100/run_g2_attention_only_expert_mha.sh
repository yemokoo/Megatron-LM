#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

usage() {
    cat >&2 <<'EOF'
usage: run_g2_attention_only_expert_mha.sh <wiki|code|phase3|all>

Runs the G2 attention-only expert baseline:
  wiki   : dense FFN + shared-router QKVO attention experts, 8 experts
  code   : expand QKVO attention experts 8 -> 16, train new experts/router only
  phase3 : router-only retune on mixed wiki+code data
  all    : wiki -> code -> phase3

FFN is dense, not expertized. Its hidden size defaults to 5632, matching the
stored parameter size of the expanded 16 FFN experts (16 * 352).
EOF
}

if [ "$#" -ne 1 ]; then
    usage
    exit 1
fi

STAGE="$1"
case "$STAGE" in
    wiki|code|phase3|all) ;;
    *) usage; exit 1 ;;
esac

export PYTHONPATH="$PROJECT_ROOT/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export G2_ROOT="${G2_ROOT:-$LOCAL_WEIGHTS/a100/mha/g2-checkpoints}"

export SEED="${SEED:-1234}"
export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-600}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-$TRAIN_ITERS}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-72}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-5632}"
export SOURCE_NUM_EXPERTS="${SOURCE_NUM_EXPERTS:-8}"
export NUM_EXPERTS="${NUM_EXPERTS:-16}"
export WIKI_NUM_EXPERTS="${WIKI_NUM_EXPERTS:-8}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MOE_LAYER_FREQ="${MOE_LAYER_FREQ:-[0,1,1,1,1,1,1,1,1]}"
export ATTN_LORA_RANK="${ATTN_LORA_RANK:-256}"
export ATTN_LORA_ALPHA="${ATTN_LORA_ALPHA:-$ATTN_LORA_RANK}"
export ATTN_FULL_RANK_LORA_RANK="${ATTN_FULL_RANK_LORA_RANK:-256}"
export ATTN_FULL_RANK_LORA_ALPHA="${ATTN_FULL_RANK_LORA_ALPHA:-$ATTN_FULL_RANK_LORA_RANK}"
export ATTN_FULL_RANK_LORA_TARGETS="${ATTN_FULL_RANK_LORA_TARGETS:-qkvo}"
export RETUNE_ITERS="${RETUNE_ITERS:-1800}"
export MASTER_PORT_WIKI="${MASTER_PORT_WIKI:-29881}"
export MASTER_PORT_CODE="${MASTER_PORT_CODE:-29882}"
export MASTER_PORT_PHASE3="${MASTER_PORT_PHASE3:-29883}"
export PAUSE_SECONDS="${PAUSE_SECONDS:-180}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-configs/model/flame-attn-only-shared-router-qkvo-experts.sh}"

export WIKI_RUN_ID="${WIKI_RUN_ID:-g2-attn-only-dense5632-top4-e8-r256-wiki-qkvo-mha-a100-bf16-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"
export CODE_RUN_ID="${CODE_RUN_ID:-g2-attn-only-dense5632-top4-e8to16-r256-wiki-to-code-qkvo-mha-a100-bf16-mb${MICRO_BATCH_SIZE}-${TRAIN_ITERS}}"
export PHASE3_RUN_ID="${PHASE3_RUN_ID:-g2-attn-only-dense5632-top4-e16-r256-phase3-router-only-retune-wikicode-qkvo-mha-a100-bf16-mb${MICRO_BATCH_SIZE}-${RETUNE_ITERS}}"

export WIKI_WEIGHTS="${WIKI_WEIGHTS:-$G2_ROOT/wiki_attn_only/$WIKI_RUN_ID}"
export CODE_WEIGHTS="${CODE_WEIGHTS:-$G2_ROOT/code/phase1/$CODE_RUN_ID}"
export PHASE3_WEIGHTS="${PHASE3_WEIGHTS:-$G2_ROOT/code/phase3/$PHASE3_RUN_ID}"

is_completed() {
    local dir="$1"
    local expected="$2"
    [ -f "$dir/latest_checkpointed_iteration.txt" ] || return 1
    local step
    step="$(tr -d '\n\r[:space:]' < "$dir/latest_checkpointed_iteration.txt")"
    [ "$step" -ge "$expected" ]
}

pause_between() {
    if [ "$PAUSE_SECONDS" -gt 0 ]; then
        echo "[PAUSE] ${PAUSE_SECONDS}s"
        sleep "$PAUSE_SECONDS"
    fi
}

run_wiki() {
    if is_completed "$WIKI_WEIGHTS" "$TRAIN_ITERS"; then
        echo "[SKIP] wiki already completed: $WIKI_WEIGHTS"
        return
    fi

    echo "[START] attention-only wiki"
    echo "[CONFIG] weights=$WIKI_WEIGHTS"
    echo "[CONFIG] dense_ffn_hidden=$FFN_HIDDEN_SIZE qkvo_attn_experts=$WIKI_NUM_EXPERTS topk=$MOE_ROUTER_TOPK rank=$ATTN_FULL_RANK_LORA_RANK seed=$SEED"
    env \
        WANDB_MODE="$WANDB_MODE" \
        RUN_ID="$WIKI_RUN_ID" \
        TRAIN_WEIGHTS="$WIKI_WEIGHTS" \
        TRAIN_DATASET="$(dataset_dir_for_task wiki)" \
        PROBE_DATASET="$(probe_dir_for_task wiki)" \
        PROBE_NAME=wiki_probe \
        SECONDARY_PROBE_DATASET="$(probe_dir_for_task code)" \
        SECONDARY_PROBE_NAME=code_probe \
        TRAIN_ITERS="$TRAIN_ITERS" \
        SAVE_INTERVAL="$SAVE_INTERVAL" \
        EVAL_INTERVAL="$EVAL_INTERVAL" \
        LOG_INTERVAL="$LOG_INTERVAL" \
        PROBE_EVAL_INTERVAL="$PROBE_EVAL_INTERVAL" \
        SECONDARY_PROBE_EVAL_INTERVAL="$SECONDARY_PROBE_EVAL_INTERVAL" \
        MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" \
        GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
        FFN_HIDDEN_SIZE="$FFN_HIDDEN_SIZE" \
        MOE_FFN_HIDDEN_SIZE="$MOE_FFN_HIDDEN_SIZE" \
        NUM_EXPERTS="$WIKI_NUM_EXPERTS" \
        MOE_ROUTER_TOPK="$MOE_ROUTER_TOPK" \
        MOE_LAYER_FREQ="$MOE_LAYER_FREQ" \
        ATTN_LORA_RANK="$ATTN_LORA_RANK" \
        ATTN_LORA_ALPHA="$ATTN_LORA_ALPHA" \
        ATTN_FULL_RANK_LORA_RANK="$ATTN_FULL_RANK_LORA_RANK" \
        ATTN_FULL_RANK_LORA_ALPHA="$ATTN_FULL_RANK_LORA_ALPHA" \
        ATTN_FULL_RANK_LORA_TARGETS="$ATTN_FULL_RANK_LORA_TARGETS" \
        MODEL_CONFIG_SCRIPT="$MODEL_CONFIG_SCRIPT" \
        SEED="$SEED" \
        WANDB_PROJECT="$WANDB_PROJECT" \
        WANDB_EXP_NAME="G2 - attention-only expert dense5632 phase1 wiki ⭐" \
        MASTER_PORT="$MASTER_PORT_WIKI" \
        bash scripts/experiment/a100/pretrain_wiki_shared_router_hybrid_local_bf16.sh
    echo "[END] attention-only wiki"
}

run_code() {
    if ! is_completed "$WIKI_WEIGHTS" "$TRAIN_ITERS"; then
        echo "[ERROR] wiki checkpoint is not complete: $WIKI_WEIGHTS" >&2
        exit 1
    fi
    if is_completed "$CODE_WEIGHTS" "$TRAIN_ITERS"; then
        echo "[SKIP] code already completed: $CODE_WEIGHTS"
        return
    fi

    echo "[START] attention-only code"
    echo "[CONFIG] source=$WIKI_WEIGHTS"
    echo "[CONFIG] weights=$CODE_WEIGHTS"
    echo "[CONFIG] dense_ffn_hidden=$FFN_HIDDEN_SIZE qkvo_attn_experts=${SOURCE_NUM_EXPERTS}->${NUM_EXPERTS} topk=$MOE_ROUTER_TOPK rank=$ATTN_FULL_RANK_LORA_RANK seed=$SEED"
    env \
        WANDB_MODE="$WANDB_MODE" \
        RUN_ID="$CODE_RUN_ID" \
        TRAIN_WEIGHTS="$CODE_WEIGHTS" \
        STAGE1_WEIGHTS_DIR="$WIKI_WEIGHTS" \
        TRAIN_DATASET="$(dataset_dir_for_task code)" \
        PROBE_DATASET="$(probe_dir_for_task code)" \
        PROBE_NAME=code_probe \
        SECONDARY_PROBE_DATASET="$(probe_dir_for_task wiki)" \
        SECONDARY_PROBE_NAME=wiki_probe \
        TRAIN_ITERS="$TRAIN_ITERS" \
        SAVE_INTERVAL="$SAVE_INTERVAL" \
        EVAL_INTERVAL="$EVAL_INTERVAL" \
        LOG_INTERVAL="$LOG_INTERVAL" \
        PROBE_EVAL_INTERVAL="$PROBE_EVAL_INTERVAL" \
        SECONDARY_PROBE_EVAL_INTERVAL="$SECONDARY_PROBE_EVAL_INTERVAL" \
        MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" \
        GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
        FFN_HIDDEN_SIZE="$FFN_HIDDEN_SIZE" \
        MOE_FFN_HIDDEN_SIZE="$MOE_FFN_HIDDEN_SIZE" \
        SOURCE_NUM_EXPERTS="$SOURCE_NUM_EXPERTS" \
        NUM_EXPERTS="$NUM_EXPERTS" \
        MOE_ROUTER_TOPK="$MOE_ROUTER_TOPK" \
        MOE_LAYER_FREQ="$MOE_LAYER_FREQ" \
        ATTN_LORA_RANK="$ATTN_LORA_RANK" \
        ATTN_LORA_ALPHA="$ATTN_LORA_ALPHA" \
        ATTN_FULL_RANK_LORA_RANK="$ATTN_FULL_RANK_LORA_RANK" \
        ATTN_FULL_RANK_LORA_ALPHA="$ATTN_FULL_RANK_LORA_ALPHA" \
        ATTN_FULL_RANK_LORA_TARGETS="$ATTN_FULL_RANK_LORA_TARGETS" \
        MODEL_CONFIG_SCRIPT="$MODEL_CONFIG_SCRIPT" \
        SEED="$SEED" \
        WANDB_PROJECT="$WANDB_PROJECT" \
        WANDB_EXP_NAME="G2 - attention-only expert dense5632 phase2 code ⭐" \
        WANDB_RUN_ID="$CODE_RUN_ID" \
        MASTER_PORT="$MASTER_PORT_CODE" \
        bash scripts/experiment/continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh
    echo "[END] attention-only code"
}

copy_code_for_phase3() {
    if [ -f "$PHASE3_WEIGHTS/latest_checkpointed_iteration.txt" ]; then
        echo "[RESUME] phase3 copy exists: $PHASE3_WEIGHTS"
        return
    fi
    if [ -e "$PHASE3_WEIGHTS" ]; then
        echo "[ERROR] phase3 destination exists without tracker: $PHASE3_WEIGHTS" >&2
        exit 1
    fi
    if ! is_completed "$CODE_WEIGHTS" "$TRAIN_ITERS"; then
        echo "[ERROR] code checkpoint is not complete: $CODE_WEIGHTS" >&2
        exit 1
    fi

    echo "[COPY] code -> phase3"
    echo "[COPY] source:      $CODE_WEIGHTS"
    echo "[COPY] destination: $PHASE3_WEIGHTS"
    mkdir -p "$PHASE3_WEIGHTS"
    rsync -aH --info=progress2 \
        --exclude 'wandb/' \
        --exclude 'events.out.tfevents*' \
        "$CODE_WEIGHTS/" "$PHASE3_WEIGHTS/"
    {
        echo "phase=3_attention_lora_router_finetune"
        echo "source=$CODE_WEIGHTS"
        echo "source_step=$(tr -d '\n\r[:space:]' < "$CODE_WEIGHTS/latest_checkpointed_iteration.txt")"
        echo "copied_at=$(date -Iseconds)"
    } > "$PHASE3_WEIGHTS/PHASE3_SOURCE.txt"
}

run_phase3() {
    copy_code_for_phase3

    local source_step target_step latest_step
    source_step="$(grep -E '^source_step=' "$PHASE3_WEIGHTS/PHASE3_SOURCE.txt" | tail -1 | cut -d= -f2-)"
    target_step=$((source_step + RETUNE_ITERS))
    latest_step="$(tr -d '\n\r[:space:]' < "$PHASE3_WEIGHTS/latest_checkpointed_iteration.txt")"
    if [ "$latest_step" -ge "$target_step" ]; then
        echo "[SKIP] phase3 already completed: latest=$latest_step target=$target_step"
        return
    fi

    echo "[START] attention-only phase3 router-only retune"
    echo "[CONFIG] weights=$PHASE3_WEIGHTS"
    echo "[CONFIG] source_step=$source_step target_step=$target_step retune_iters=$RETUNE_ITERS"
    env \
        WANDB_MODE="$WANDB_MODE" \
        RUN_ID="$PHASE3_RUN_ID" \
        TRAIN_WEIGHTS="$PHASE3_WEIGHTS" \
        SOURCE_STEP="$source_step" \
        TRAIN_ITERS="$target_step" \
        RETUNE_ITERS="$RETUNE_ITERS" \
        SAVE_INTERVAL="$SAVE_INTERVAL" \
        LOG_INTERVAL="$LOG_INTERVAL" \
        PROBE_EVAL_INTERVAL="$PROBE_EVAL_INTERVAL" \
        SECONDARY_PROBE_EVAL_INTERVAL="$SECONDARY_PROBE_EVAL_INTERVAL" \
        MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" \
        GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
        FFN_HIDDEN_SIZE="$FFN_HIDDEN_SIZE" \
        MOE_FFN_HIDDEN_SIZE="$MOE_FFN_HIDDEN_SIZE" \
        SOURCE_NUM_EXPERTS="$SOURCE_NUM_EXPERTS" \
        RESUME_FROM_NUM_EXPERTS="$SOURCE_NUM_EXPERTS" \
        NUM_EXPERTS="$NUM_EXPERTS" \
        MOE_ROUTER_TOPK="$MOE_ROUTER_TOPK" \
        MOE_LAYER_FREQ="$MOE_LAYER_FREQ" \
        ATTN_LORA_RANK="$ATTN_LORA_RANK" \
        ATTN_LORA_ALPHA="$ATTN_LORA_ALPHA" \
        ATTN_FULL_RANK_LORA_RANK="$ATTN_FULL_RANK_LORA_RANK" \
        ATTN_FULL_RANK_LORA_ALPHA="$ATTN_FULL_RANK_LORA_ALPHA" \
        ATTN_FULL_RANK_LORA_TARGETS="$ATTN_FULL_RANK_LORA_TARGETS" \
        MODEL_CONFIG_SCRIPT="$MODEL_CONFIG_SCRIPT" \
        SEED="$SEED" \
        WANDB_PROJECT="$WANDB_PROJECT" \
        WANDB_EXP_NAME="G2 - attention-only expert dense5632 phase3 router finetune ⭐" \
        WANDB_RUN_ID="$PHASE3_RUN_ID" \
        MASTER_PORT="$MASTER_PORT_PHASE3" \
        bash scripts/experiment/a100/phase3_router_only_retune_shared_router_hybrid_mixed_local_bf16.sh
    echo "[END] attention-only phase3 router-only retune"
}

echo "[CONFIG] G2 attention-only expert chain"
echo "[CONFIG] stage=$STAGE"
echo "[CONFIG] wiki_weights=$WIKI_WEIGHTS"
echo "[CONFIG] code_weights=$CODE_WEIGHTS"
echo "[CONFIG] phase3_weights=$PHASE3_WEIGHTS"
echo "[CONFIG] ffn=dense hidden $FFN_HIDDEN_SIZE; QKVO attention experts ${SOURCE_NUM_EXPERTS}->${NUM_EXPERTS}; topk=$MOE_ROUTER_TOPK; rank=$ATTN_FULL_RANK_LORA_RANK"

case "$STAGE" in
    wiki)
        run_wiki
        ;;
    code)
        run_code
        ;;
    phase3)
        run_phase3
        ;;
    all)
        run_wiki
        pause_between
        run_code
        pause_between
        run_phase3
        ;;
esac

echo "[ALL DONE] G2 attention-only expert chain"
