#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

source "$SCRIPT_DIR/common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29831}"
export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
export G2_ROOT="${G2_ROOT:-$PROJECT_ROOT/.local/weights/a100/mha/g2-checkpoints}"
export BASE_WEIGHTS_DIR="${BASE_WEIGHTS_DIR:-$PROJECT_ROOT/.local/weights/a100/mha}"
export OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/.local/analysis/g2matched-ffn-only-hidden-space-wiki-probe}"

export NUM_LAYERS="${NUM_LAYERS:-9}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-5472}"
export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export MOE_LAYER_FREQ="${MOE_LAYER_FREQ:-[0]*1+[1]*8}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-4}"
export HIDDEN_SPACE_MAX_TOKENS="${HIDDEN_SPACE_MAX_TOKENS:-2048}"
export HIDDEN_SPACE_LAYERS="${HIDDEN_SPACE_LAYERS:-all}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-EleutherAI/pythia-12b}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-scripts/experiment/a100/flame-moe-bf16-no-shared.sh}"

WIKI_REGISTRY="$G2_ROOT/wiki/g2matched-top4-e8-ffn352-wiki-ffn-moe-mha-a100-bf16-mb128-1800"
WIKI_ORIGINAL="$BASE_WEIGHTS_DIR/wiki-a-moe-g2matched-bf16/g2matched-top4-e8-ffn352-wiki-ffn-moe-mha-a100-bf16-mb128-1800"
CODE_DIR="$G2_ROOT/code/g2matched/g2matched-top4-e8to16-ffn352-wiki-to-code-ffn-moe-attn-freeze-mha-a100-bf16-mb96-1800"
RETUNE_DIR="$G2_ROOT/code/phase3/g2matched-attn-freeze-phase3-router-only-retune-wikicode-no-reinit-mb96-1800"
WIKI_PROBE_PREFIX="$PROJECT_ROOT/data/wiki/test/test_text_document"

pick_existing_dir() {
    local first="$1"
    local second="$2"
    if [ -f "$first/latest_checkpointed_iteration.txt" ]; then
        echo "$first"
        return
    fi
    if [ -f "$second/latest_checkpointed_iteration.txt" ]; then
        echo "$second"
        return
    fi
    echo "[ERROR] missing checkpoint dir:" >&2
    echo "  $first" >&2
    echo "  $second" >&2
    exit 1
}

require_checkpoint() {
    local path="$1"
    local label="$2"
    if [ ! -f "$path/latest_checkpointed_iteration.txt" ]; then
        echo "[ERROR] missing $label checkpoint tracker: $path/latest_checkpointed_iteration.txt" >&2
        exit 1
    fi
}

WIKI_DIR="$(pick_existing_dir "$WIKI_REGISTRY" "$WIKI_ORIGINAL")"
require_checkpoint "$CODE_DIR" "code-trained"
require_checkpoint "$RETUNE_DIR" "router-retuned"

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/hidden"

echo "[CONFIG] FFN-only hidden-space probe"
echo "[CONFIG] wiki_only=$WIKI_DIR"
echo "[CONFIG] code_trained=$CODE_DIR"
echo "[CONFIG] router_retuned=$RETUNE_DIR"
echo "[CONFIG] wiki_probe=$WIKI_PROBE_PREFIX"
echo "[CONFIG] out_dir=$OUT_DIR"
echo "[CONFIG] mb=$MICRO_BATCH_SIZE, gbs=$GLOBAL_BATCH_SIZE, max_tokens=$HIDDEN_SPACE_MAX_TOKENS"

run_dump() {
    local label="$1"
    local load_dir="$2"
    local num_experts="$3"
    local extra_resume_arg=()
    if [ "$num_experts" -gt 8 ]; then
        extra_resume_arg=(--moe-resume-from-num-experts 8)
    fi

    export NUM_EXPERTS="$num_experts"
    # shellcheck source=/dev/null
    source "$MODEL_CONFIG_SCRIPT"

    local out_npz="$OUT_DIR/hidden/${label}.npz"
    local out_log="$OUT_DIR/logs/${label}.log"
    echo "[RUN] $label hidden dump -> $out_npz"

    torchrun \
        --nproc_per_node "$NPROC_PER_NODE" \
        --master_addr "$MASTER_ADDR" \
        --master_port "$MASTER_PORT" \
        Megatron-LM/pretrain_gpt.py \
        "${MODEL_ARGS[@]}" \
        --transformer-impl local \
        --pipeline-model-parallel-size 1 \
        --expert-model-parallel-size 1 \
        --distributed-timeout-minutes 30 \
        --no-persist-layer-norm \
        --bf16 \
        --micro-batch-size "$MICRO_BATCH_SIZE" \
        --global-batch-size "$GLOBAL_BATCH_SIZE" \
        --seq-length 512 \
        --data-path 1.0 "$WIKI_PROBE_PREFIX" \
        --split 100,0,0 \
        --train-iters 1 \
        --skip-train \
        --load "$load_dir" \
        --no-load-optim \
        --no-load-rng \
        "${extra_resume_arg[@]}" \
        --eval-interval 1 \
        --probe-name wiki_probe \
        --probe-eval-iters "$PROBE_EVAL_ITERS" \
        --probe-eval-interval 1 \
        --probe-data-path 1.0 "$WIKI_PROBE_PREFIX" \
        --run-initial-probe-eval \
        --hidden-space-dump-path "$out_npz" \
        --hidden-space-dump-label "$label" \
        --hidden-space-dump-max-tokens "$HIDDEN_SPACE_MAX_TOKENS" \
        --hidden-space-dump-layers "$HIDDEN_SPACE_LAYERS" \
        > "$out_log" 2>&1

    echo "[DONE] $label"
}

run_dump wiki_only "$WIKI_DIR" 8
MASTER_PORT=$((MASTER_PORT + 1)) run_dump code_trained "$CODE_DIR" 16
MASTER_PORT=$((MASTER_PORT + 2)) run_dump router_retuned "$RETUNE_DIR" 16

python scripts/analysis/plot_hidden_space_ffn_only.py \
    --wiki-only "$OUT_DIR/hidden/wiki_only.npz" \
    --code-trained "$OUT_DIR/hidden/code_trained.npz" \
    --router-retuned "$OUT_DIR/hidden/router_retuned.npz" \
    --out-dir "$OUT_DIR/plots" \
    --method "${HIDDEN_SPACE_PLOT_METHOD:-pca}" \
    --max-points-per-stage "${HIDDEN_SPACE_PLOT_MAX_POINTS_PER_STAGE:-1200}"

echo "[DONE] hidden-space analysis"
echo "[PLOTS] $OUT_DIR/plots"
