#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

resolve_python() {
  command -v python3 || command -v python
}

make_step_load_dir() {
  local source_dir="$1"
  local step="$2"
  local output_dir="$3"
  mkdir -p "$output_dir"
  printf "%s\n" "$step" > "$output_dir/latest_checkpointed_iteration.txt"
  local iter_name
  iter_name=$(printf "iter_%07d" "$step")
  ln -sfn "$source_dir/$iter_name" "$output_dir/$iter_name"
}

run_dump() {
  local load_dir="$1"
  local step_label="$2"
  local output_pt="$3"
  local num_experts="$4"
  local source_num_experts="$5"
  local port="$6"

  CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nnodes 1 \
    --nproc_per_node 1 \
    --master_port "$port" \
    analysis/dump_shared_router_hidden_states.py \
    --load "$load_dir" \
    --output-pt "$output_pt" \
    --compare-label "g1_hidden_step_${step_label}" \
    --checkpoint-step-label "$step_label" \
    --data-path 1.0 "$ROUTER_MEMORY_PREFIX" \
    --dataset-split 100,0,0 \
    --dataset-split-name train \
    --consumed-samples 0 \
    --max-batches "$MAX_BATCHES" \
    --max-tokens-per-layer "$MAX_TOKENS_PER_LAYER" \
    --micro-batch-size "$MICRO_BATCH_SIZE" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --seq-length "$SEQ_LENGTH" \
    --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --tensor-model-parallel-size 1 \
    --transformer-impl local \
    --spec megatron.core.models.gpt.shared_router_hybrid_layer_specs gpt_shared_router_hybrid_local_spec \
    --shared-router-hybrid-model \
    --hidden-size 1024 \
    --ffn-hidden-size 5472 \
    --num-layers 9 \
    --num-attention-heads 16 \
    --group-query-attention \
    --num-query-groups 16 \
    --swiglu \
    --max-position-embeddings 2048 \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --disable-bias-linear \
    --hidden-dropout 0.0 \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model EleutherAI/pythia-12b \
    --moe-ffn-hidden-size 704 \
    --num-experts "$num_experts" \
    --moe-router-topk 2 \
    --moe-layer-freq "[0,1,1,1,1,1,1,1,1]" \
    --moe-router-pre-softmax \
    --moe-router-score-function softmax \
    --moe-aux-loss-coeff 0.01 \
    --moe-z-loss-coeff 0.001 \
    --attn-lora-num-experts "$num_experts" \
    --attn-lora-rank 512 \
    --attn-lora-topk 2 \
    --attn-lora-alpha 512 \
    --attn-full-rank-lora-rank 512 \
    --attn-full-rank-lora-alpha 512 \
    --attn-full-rank-lora-targets qkvo \
    --attn-full-rank-lora-active-targets "" \
    --moe-router-dtype fp32 \
    --moe-grouped-gemm \
    --attn-lora-grouped-gemm \
    --no-persist-layer-norm \
    --bf16 \
    --no-load-optim \
    --no-load-rng \
    --exit-on-missing-checkpoint
}

export PYTHON_BIN="${PYTHON_BIN:-$(resolve_python)}"
export GPU_DEVICE="${GPU_DEVICE:-0}"
export SEQ_LENGTH="${SEQ_LENGTH:-512}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
export MAX_BATCHES="${MAX_BATCHES:-2}"
export MAX_TOKENS_PER_LAYER="${MAX_TOKENS_PER_LAYER:-2048}"
export MASTER_PORT_BASE="${MASTER_PORT_BASE:-29840}"
export BASE_STAGE_DIR="${BASE_STAGE_DIR:-$PROJECT_ROOT/.local/weights/a100/mha/shared-router-granularity-qkvo}"
export WIKI_RUN_DIR="${WIKI_RUN_DIR:-$BASE_STAGE_DIR/wiki/g1-top2-e4-ffn704-r512-wiki-shared-router-qkvo-mha-a100-bf16-mb72-1800}"
export CODE_RUN_DIR="${CODE_RUN_DIR:-$BASE_STAGE_DIR/code/g1-top2-e4to8-ffn704-r512-wiki-to-code-shared-router-qkvo-mha-a100-bf16-mb96-router-memory-fixed5-kl0p1-1800}"
export ROUTER_MEMORY_PREFIX="${ROUTER_MEMORY_PREFIX:-$PROJECT_ROOT/data/wiki/router_memory_5pct/train_text_document}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/analysis_outputs/g1_router_replay_hidden_drift}"
export STEPS="${STEPS:-300 600 900 1200 1500 1800}"

mkdir -p "$OUTPUT_ROOT/dumps" "$OUTPUT_ROOT/load_views"

echo "[hidden-drift] dumping Wiki baseline"
run_dump \
  "$WIKI_RUN_DIR" \
  0 \
  "$OUTPUT_ROOT/dumps/wiki_1800.pt" \
  4 \
  4 \
  "$MASTER_PORT_BASE"

plot_inputs=("--input" "wiki_1800=$OUTPUT_ROOT/dumps/wiki_1800.pt")

idx=1
for step in $STEPS; do
  iter_name=$(printf "iter_%07d" "$step")
  if [ ! -d "$CODE_RUN_DIR/$iter_name" ]; then
    echo "ERROR: missing checkpoint $CODE_RUN_DIR/$iter_name" >&2
    exit 1
  fi
  step_load_dir="$OUTPUT_ROOT/load_views/code_${step}"
  make_step_load_dir "$CODE_RUN_DIR" "$step" "$step_load_dir"
  echo "[hidden-drift] dumping code checkpoint step=$step"
  run_dump \
    "$step_load_dir" \
    "$step" \
    "$OUTPUT_ROOT/dumps/code_${step}.pt" \
    8 \
    4 \
    "$((MASTER_PORT_BASE + idx))"
  plot_inputs+=("--input" "code_${step}=$OUTPUT_ROOT/dumps/code_${step}.pt")
  idx=$((idx + 1))
done

"$PYTHON_BIN" analysis/plot_shared_router_hidden_drift.py \
  "${plot_inputs[@]}" \
  --baseline-label wiki_1800 \
  --output-dir "$OUTPUT_ROOT/plots" \
  --pca-layers 2,5,9 \
  --pca-token-count 512

echo "Saved hidden drift outputs under: $OUTPUT_ROOT"
echo "Heatmaps:"
echo "  $OUTPUT_ROOT/plots/cosine_distance_mean_heatmap.png"
echo "  $OUTPUT_ROOT/plots/rms_l2_drift_heatmap.png"
echo "  $OUTPUT_ROOT/plots/norm_ratio_mean_heatmap.png"
echo "PCA examples:"
echo "  $OUTPUT_ROOT/plots/layer_02_pca_hidden_cloud.png"
echo "  $OUTPUT_ROOT/plots/layer_05_pca_hidden_cloud.png"
echo "  $OUTPUT_ROOT/plots/layer_09_pca_hidden_cloud.png"
