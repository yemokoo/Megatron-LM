#!/usr/bin/env bash
# BoS-only sampling from a shared-router hybrid (FFN + QKVO attention experts)
# checkpoint.  Same model construction as the hidden-dump `run_hyb` path.
#
#   GPU=0 CKPT=<dir> NE=16 SRC=8 OUT=<dir> TEMP=1.0 TOPP=0.95 N=2048 T=512 BATCH=64 \
#     bash scripts/analysis/run_bos_sample_hybrid.sh
set -euo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$P"
export PYTHONNOUSERSITE=1
export PATH=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin:/usr/bin:/bin
export TOKENIZER_MODEL=/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6

CKPT="${CKPT:?checkpoint dir}"; OUT="${OUT:?output dir}"
NE="${NE:-16}"; SRC="${SRC:-8}"; GPU="${GPU:-0}"; PORT="${PORT:-$((42000 + GPU))}"
TEMP="${TEMP:-1.0}"; TOPP="${TOPP:-0.95}"; TOPK="${TOPK:-0}"; N="${N:-2048}"; T="${T:-512}"; BATCH="${BATCH:-64}"; SEED="${SEED:-0}"
LABEL="${LABEL:-$(basename "$CKPT")_t${TEMP}_p${TOPP}}"
PREFIX="${PREFIX:-}"; ROUTE_ALLOW="${ROUTE_ALLOW:-}"; ROUTE_TAU="${ROUTE_TAU:-0}"; ROUTE_GROUPS="${ROUTE_GROUPS:-}"; ROUTE_BALANCE="${ROUTE_BALANCE:-0}"; KV="${KV:-0}"; KV_CHECK="${KV_CHECK:-0}"; ANCHOR_JSON="${ANCHOR_JSON:-}"
kv=(); [ "$KV" = 1 ] && kv=(--bos-kv-cache); [ "$KV_CHECK" != 0 ] && kv+=(--bos-kv-check "$KV_CHECK"); [ -n "$ANCHOR_JSON" ] && kv+=(--bos-anchor-json "$ANCHOR_JSON")
WIKI_PROBE=/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/wiki/test/test_text_document

export NUM_LAYERS=9 HIDDEN_SIZE=1024 FFN_HIDDEN_SIZE=5472 NUM_QUERY_GROUPS=16 MOE_FFN_HIDDEN_SIZE=352 \
       MOE_LAYER_FREQ="[0]*1+[1]*8" MOE_ROUTER_TOPK=4 NUM_EXPERTS="$NE" MOE_ROUTER_DTYPE=fp32 \
       ATTN_LORA_RANK=256 ATTN_LORA_ALPHA=256 ATTN_FULL_RANK_LORA_RANK=256 ATTN_FULL_RANK_LORA_ALPHA=256 \
       ATTN_FULL_RANK_LORA_TARGETS=qkvo MOE_GROUPED_GEMM=1 ATTN_LORA_GROUPED_GEMM=1
# shellcheck source=/dev/null
source configs/model/flame-shared-router-hybrid-experts.sh
resume=(); [ "$NE" -gt "$SRC" ] && resume=(--shared-router-hybrid-resume-from-num-experts "$SRC" --shared-router-hybrid-train-new-experts-and-router-only)

mkdir -p "$OUT"
echo "[bos] ckpt=$CKPT experts=$NE src=$SRC gpu=$GPU temp=$TEMP top_p=$TOPP N=$N T=$T -> $OUT"
CUDA_VISIBLE_DEVICES="$GPU" torchrun --nproc_per_node 1 --master_addr 127.0.0.1 --master_port "$PORT" \
  scripts/analysis/bos_sample_hybrid.py "${MODEL_ARGS[@]}" \
  --transformer-impl local --pipeline-model-parallel-size 1 --expert-model-parallel-size 1 \
  --distributed-timeout-minutes 30 --no-persist-layer-norm --bf16 \
  --micro-batch-size "$BATCH" --global-batch-size "$BATCH" --seed 1234 \
  --lr 3e-4 --min-lr 3e-5 --lr-decay-style WSD --lr-decay-iters 1 --lr-warmup-fraction 0.0 --lr-wsd-decay-iters 1 \
  --seq-length 512 --data-path 1.0 "$WIKI_PROBE" --split 100,0,0 --train-iters 1 \
  --load "$CKPT" --no-load-optim --no-load-rng --finetune "${resume[@]}" \
  --bos-out-dir "$OUT" --bos-num-seqs "$N" --bos-seq-len "$T" --bos-batch "$BATCH" \
  --bos-temperature "$TEMP" --bos-top-p "$TOPP" --bos-top-k "$TOPK" --bos-seed "$SEED" --bos-label "$LABEL" --bos-prefix-text "$PREFIX" --bos-route-allow "$ROUTE_ALLOW" --bos-route-gumbel-tau "$ROUTE_TAU" --bos-route-groups "$ROUTE_GROUPS" --bos-route-balance-eta "$ROUTE_BALANCE" "${kv[@]}" \
  2>&1 | tee "$OUT/run.log" | grep -E "^\[bos\]|Error|Traceback|error:" || true
[ -s "$OUT/gen_text_document.idx" ] && echo "[bos] DONE $OUT" || { echo "[bos] FAILED (see $OUT/run.log)"; exit 1; }
