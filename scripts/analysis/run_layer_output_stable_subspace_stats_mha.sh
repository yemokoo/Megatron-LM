#!/usr/bin/env bash
set -euo pipefail

# Read-only checkpoint comparison.  This runs no optimizer/training step.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PY_ENV="${PY_ENV:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100}"
TORCHRUN="${TORCHRUN:-$PY_ENV/bin/torchrun}"
GPU="${GPU:?set GPU to one of 0,1,2,3}"
case "$GPU" in 0|1|2|3) ;; *) echo "[ERROR] GPU must be 0,1,2,3" >&2; exit 2 ;; esac

TARGET_LABEL="${TARGET_LABEL:?set a semantic target label}"
TARGET_LOAD="${TARGET_LOAD:?set the read-only target checkpoint}"
DOMAIN="${DOMAIN:?set DOMAIN=wiki or code}"
case "$DOMAIN" in wiki|code) ;; *) echo "[ERROR] DOMAIN must be wiki or code" >&2; exit 2 ;; esac

REFERENCE_LOAD="${REFERENCE_LOAD:-/data2/seonghyeonnoh/LLM-continual-learning-runs/g2_olddata_kd_9run_20260808/01_common_kd_init/code_e8_to_e16_wiki_kd_step600}"
OUT_ROOT="${OUT_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/layer_output_stable_subspace_20260810}"
DATA_ROOT="${DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6}"
TARGET_TOKENS="${TARGET_TOKENS:-10000000}"
BLOCK_TOKENS="${BLOCK_TOKENS:-2000000}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-3000}"
RESERVOIR_SIZE="${RESERVOIR_SIZE:-4096}"
MASTER_PORT="${MASTER_PORT:-$((31800 + GPU))}"

probe_prefix="$DATA_ROOT/$DOMAIN/train/train_text_document"
final_dir="$OUT_ROOT/streaming_stats/$DOMAIN/$TARGET_LABEL"
tmp_dir="$final_dir.inprogress"
log_path="$OUT_ROOT/logs/stream_${DOMAIN}_${TARGET_LABEL}.log"
manifest_path="$OUT_ROOT/manifests/${DOMAIN}_10m_seed1234.json"

for checkpoint in "$REFERENCE_LOAD" "$TARGET_LOAD"; do
    [[ -f "$checkpoint/latest_checkpointed_iteration.txt" ]] || {
        echo "[ERROR] checkpoint tracker missing: $checkpoint" >&2; exit 1;
    }
done
[[ -f "$probe_prefix.bin" && -f "$probe_prefix.idx" ]] || {
    echo "[ERROR] indexed corpus missing: $probe_prefix" >&2; exit 1;
}
if [[ -f "$final_dir/metadata.json" && "${FORCE:-0}" != 1 ]]; then
    echo "[SKIP] $final_dir"
    exit 0
fi
if [[ -e "$tmp_dir" ]]; then
    mv "$tmp_dir" "$tmp_dir.failed.$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$(dirname "$final_dir")" "$(dirname "$log_path")" "$(dirname "$manifest_path")"

gpu_uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v gpu="$GPU" '$1 == gpu {print $2}')"
[[ -n "$gpu_uuid" ]] || { echo "[ERROR] physical GPU $GPU not found" >&2; exit 1; }
gpu_pids="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | awk -F', ' -v uuid="$gpu_uuid" '$1 == uuid {print $2}')"
[[ -z "$gpu_pids" ]] || { echo "[COLLISION] GPU $GPU is occupied by PIDs: $gpu_pids" >&2; exit 75; }

export PATH="$PY_ENV/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/Megatron-LM"
export HF_HOME="${HF_HOME:-/data2/seonghyeonnoh/homecache/huggingface}"
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1

model_args=(
    --hidden-size 1024 --ffn-hidden-size 5472 --num-layers 9
    --num-attention-heads 16 --group-query-attention --num-query-groups 16
    --swiglu --max-position-embeddings 2048 --normalization RMSNorm --norm-epsilon 1e-6
    --untie-embeddings-and-output-weights --position-embedding-type rope --disable-bias-linear
    --moe-ffn-hidden-size 352 --num-experts 16 --moe-router-topk 4
    --moe-layer-freq '[0]*1+[1]*8' --moe-router-dtype fp32 --moe-router-pre-softmax
    --moe-router-score-function softmax --moe-aux-loss-coeff 0.01 --moe-z-loss-coeff 0.001
    --hidden-dropout 0.0 --attention-dropout 0.0 --init-method-std 0.02
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$TOKENIZER_MODEL"
)

echo "[RUN] reference=expansion KD-init complete(step 600) target=$TARGET_LABEL domain=$DOMAIN tokens=$TARGET_TOKENS GPU=$GPU"
set +e
CUDA_VISIBLE_DEVICES="$GPU" "$TORCHRUN" --nproc_per_node 1 --master_addr 127.0.0.1 --master_port "$MASTER_PORT" \
    Megatron-LM/pretrain_gpt.py "${model_args[@]}" \
    --transformer-impl local --pipeline-model-parallel-size 1 --expert-model-parallel-size 1 \
    --distributed-timeout-minutes 30 --no-persist-layer-norm --bf16 \
    --micro-batch-size "$MICRO_BATCH_SIZE" --global-batch-size "$MICRO_BATCH_SIZE" \
    --lr 3e-4 --min-lr 3e-5 --lr-decay-style WSD --lr-decay-iters 1 \
    --lr-warmup-fraction 0.0 --lr-wsd-decay-iters 1 --seq-length 512 \
    --data-path 1.0 "$probe_prefix" --split 100,0,0 --train-iters 1 --skip-train \
    --load "$TARGET_LOAD" --no-load-optim --no-load-rng \
    --moe-resume-from-num-experts 8 \
    --moe-old-model-kl-load "$REFERENCE_LOAD" --moe-old-model-kl-num-experts 16 \
    --diagnostic-override-train-iteration 0 --diagnostic-override-consumed-train-samples 0 \
    --eval-interval 1 --probe-name "${DOMAIN}_layer_output_stability" \
    --probe-eval-iters "$PROBE_EVAL_ITERS" --probe-eval-interval 1 \
    --probe-data-path 1.0 "$probe_prefix" --run-initial-probe-eval \
    --layer-output-streaming-stats-path "$tmp_dir" \
    --layer-output-streaming-target-tokens "$TARGET_TOKENS" \
    --layer-output-streaming-block-tokens "$BLOCK_TOKENS" \
    --layer-output-streaming-layers 2,3,4,5,6,7,8,9 \
    --layer-output-streaming-reservoir-size "$RESERVOIR_SIZE" \
    --layer-output-streaming-label "$TARGET_LABEL:$DOMAIN" \
    --layer-output-streaming-manifest-path "$manifest_path" \
    --layer-output-streaming-ffn-diagnostic \
    >"$log_path" 2>&1
rc=$?
set -e
if [[ "$rc" -ne 0 ]]; then
    echo "[ERROR] analysis failed rc=$rc log=$log_path" >&2
    exit "$rc"
fi
[[ -f "$tmp_dir/metadata.json" ]] || { echo "[ERROR] no metadata produced: $tmp_dir" >&2; exit 1; }
mv "$tmp_dir" "$final_dir"
echo "[DONE] $final_dir"
