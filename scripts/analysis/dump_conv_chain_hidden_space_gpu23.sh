#!/usr/bin/env bash
set -uo pipefail
# Dump L2..L9 hidden states of the conv-chain KD-init (24E, before conv
# training) on wiki / code / conv test probes, for a joint PCA of the three
# domains.  One model, three data sources; the dump path is the same hook the
# earlier hidden-space plots used.
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$R"
export PYTHONNOUSERSITE=1
export PATH=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin:/usr/bin:/bin
export TOKENIZER_MODEL=/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6
V=/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup
LOAD=/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_conversation_chain_20260817/01_expansion_kd_init_e16_to_e24_step600
OUT=/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_conversation_chain_20260817/hidden_space_pca
mkdir -p "$OUT/hidden" "$OUT/logs"
export NUM_LAYERS=9 HIDDEN_SIZE=1024 FFN_HIDDEN_SIZE=5472 NUM_QUERY_GROUPS=16 MOE_FFN_HIDDEN_SIZE=352 MOE_LAYER_FREQ="[0]*1+[1]*8" MOE_ROUTER_TOPK=4 NUM_EXPERTS=24 SOURCE_NUM_EXPERTS=16
source scripts/experiment/a100/flame-moe-bf16-no-shared.sh
declare -A PROBE=( [wiki]="$V/wiki/test/test_text_document" [code]="$V/code/test/test_text_document" [conversation]="$V/conversation/test/shard_00000_text_document" )
port=36950
for task in wiki code conversation; do
  npz="$OUT/hidden/${task}.npz"; log="$OUT/logs/${task}.log"
  [ -s "$npz" ] && { echo "[SKIP] $npz"; continue; }
  echo "[RUN] $task"; port=$((port+1))
  CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node 1 --master_addr 127.0.0.1 --master_port $port \
    Megatron-LM/pretrain_gpt.py "${MODEL_ARGS[@]}" \
    --transformer-impl local --pipeline-model-parallel-size 1 --expert-model-parallel-size 1 \
    --distributed-timeout-minutes 30 --no-persist-layer-norm --bf16 \
    --micro-batch-size 64 --global-batch-size 64 --lr 3e-4 --min-lr 3e-5 --lr-decay-style WSD \
    --lr-decay-iters 1 --lr-warmup-fraction 0.0 --lr-wsd-decay-iters 1 --seq-length 512 \
    --data-path 1.0 "${PROBE[$task]}" --split 100,0,0 --train-iters 1 --skip-train \
    --load "$LOAD" --no-load-optim --no-load-rng --moe-resume-from-num-experts 16 \
    --diagnostic-override-train-iteration 0 --diagnostic-override-consumed-train-samples 0 \
    --eval-interval 1 --probe-name "${task}_probe" --probe-eval-iters 2 --probe-eval-interval 1 \
    --probe-data-path 1.0 "${PROBE[$task]}" --run-initial-probe-eval \
    --hidden-space-dump-path "$npz" --hidden-space-dump-label "kdinit24e_${task}" \
    --hidden-space-dump-max-tokens 20000 --hidden-space-dump-layers all \
    > "$log" 2>&1 && echo "[DONE] $npz" || echo "[FAIL] $task (see $log)"
done
