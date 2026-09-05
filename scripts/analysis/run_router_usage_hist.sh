#!/usr/bin/env bash
# Routing histograms of a hybrid checkpoint on one corpus.
#   GPU=0 CKPT=<dir> NE=16 SRC=8 DATA=<prefix> OUT=<json> [ITERS=16 MAXTOK=500000] \
#     bash scripts/analysis/run_router_usage_hist.sh
set -euo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$P"
export PYTHONNOUSERSITE=1
export PATH=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin:/usr/bin:/bin
export TOKENIZER_MODEL=/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6
CKPT="${CKPT:?}"; DATA="${DATA:?dataset prefix}"; OUT="${OUT:?json out}"
NE="${NE:-16}"; SRC="${SRC:-8}"; GPU="${GPU:-0}"; PORT="${PORT:-$((42100 + GPU))}"
ITERS="${ITERS:-16}"; MAXTOK="${MAXTOK:-500000}"; BATCH="${BATCH:-64}"; LABEL="${LABEL:-$(basename "$OUT" .json)}"
export NUM_LAYERS=9 HIDDEN_SIZE=1024 FFN_HIDDEN_SIZE=5472 NUM_QUERY_GROUPS=16 MOE_FFN_HIDDEN_SIZE=352 \
       MOE_LAYER_FREQ="[0]*1+[1]*8" MOE_ROUTER_TOPK=4 NUM_EXPERTS="$NE" MOE_ROUTER_DTYPE=fp32 \
       ATTN_LORA_RANK=256 ATTN_LORA_ALPHA=256 ATTN_FULL_RANK_LORA_RANK=256 ATTN_FULL_RANK_LORA_ALPHA=256 \
       ATTN_FULL_RANK_LORA_TARGETS=qkvo MOE_GROUPED_GEMM=1 ATTN_LORA_GROUPED_GEMM=1
# shellcheck source=/dev/null
source configs/model/flame-shared-router-hybrid-experts.sh
resume=(); [ "$NE" -gt "$SRC" ] && resume=(--shared-router-hybrid-resume-from-num-experts "$SRC" --shared-router-hybrid-train-new-experts-and-router-only)
mkdir -p "$(dirname "$OUT")"
CUDA_VISIBLE_DEVICES="$GPU" torchrun --nproc_per_node 1 --master_addr 127.0.0.1 --master_port "$PORT" \
  scripts/analysis/router_usage_hist.py "${MODEL_ARGS[@]}" \
  --transformer-impl local --pipeline-model-parallel-size 1 --expert-model-parallel-size 1 \
  --distributed-timeout-minutes 30 --no-persist-layer-norm --bf16 \
  --micro-batch-size "$BATCH" --global-batch-size "$BATCH" --seed 1234 \
  --lr 3e-4 --min-lr 3e-5 --lr-decay-style WSD --lr-decay-iters 1 --lr-warmup-fraction 0.0 --lr-wsd-decay-iters 1 \
  --seq-length 512 --data-path 1.0 "$DATA" --split 100,0,0 --train-iters 1 --dataloader-type single \
  --load "$CKPT" --no-load-optim --no-load-rng --finetune "${resume[@]}" \
  --ru-data-path 1.0 "$DATA" --ru-eval-iters "$ITERS" --ru-max-tokens "$MAXTOK" --ru-out "$OUT" --ru-label "$LABEL" \
  2>&1 | tee "${OUT%.json}.log" | grep -E "^\[ru\]|Error|Traceback" || true
[ -s "$OUT" ] && echo "[ru] DONE $OUT" || { echo "[ru] FAILED (see ${OUT%.json}.log)"; exit 1; }
