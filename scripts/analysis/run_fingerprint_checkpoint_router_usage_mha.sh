#!/usr/bin/env bash
set -euo pipefail

# Read-only natural-routing probe for one 16-expert checkpoint and one domain.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PY_ENV="${PY_ENV:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100}"
TORCHRUN="${TORCHRUN:-$PY_ENV/bin/torchrun}"
GPU="${GPU:?set physical GPU 0,1,2,or 3}"
case "$GPU" in 0|1|2|3) ;; *) echo "[ERROR] GPU must be one of 0,1,2,3" >&2; exit 2 ;; esac

LABEL="${LABEL:?set a semantic checkpoint label}"
LOAD_DIR="${LOAD_DIR:?set the read-only checkpoint path}"
EXPECTED_STEP="${EXPECTED_STEP:-200}"
DOMAIN="${DOMAIN:?set DOMAIN=wiki or code}"
case "$DOMAIN" in wiki|code) ;; *) echo "[ERROR] DOMAIN must be wiki or code" >&2; exit 2 ;; esac

OUT_ROOT="${OUT_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/layer_output_fingerprint_kd_20260810/router_usage_readonly}"
DATA_ROOT="${DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-25}"
MASTER_PORT="${MASTER_PORT:-$((31900 + GPU))}"

tracker="$LOAD_DIR/latest_checkpointed_iteration.txt"
[[ -f "$tracker" ]] || { echo "[ERROR] checkpoint tracker missing: $tracker" >&2; exit 1; }
actual_step="$(tr -d '[:space:]' < "$tracker")"
[[ "$actual_step" == "$EXPECTED_STEP" ]] || {
    echo "[ERROR] expected checkpoint step $EXPECTED_STEP, got $actual_step: $LOAD_DIR" >&2
    exit 1
}

probe_prefix="$DATA_ROOT/$DOMAIN/test/test_text_document"
[[ -f "$probe_prefix.bin" && -f "$probe_prefix.idx" ]] || {
    echo "[ERROR] indexed test corpus missing: $probe_prefix" >&2; exit 1;
}

final_dir="$OUT_ROOT/$DOMAIN/$LABEL"
tmp_dir="$final_dir.inprogress"
log_path="$tmp_dir/probe.log"
if [[ -e "$final_dir" ]]; then
    if [[ "${FORCE:-0}" != 1 ]]; then
        echo "[SKIP] $final_dir"
        exit 0
    fi
    mv "$final_dir" "$final_dir.replaced.$(date +%Y%m%d-%H%M%S)"
fi
if [[ -e "$tmp_dir" ]]; then
    mv "$tmp_dir" "$tmp_dir.failed.$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$tmp_dir"

gpu_uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v gpu="$GPU" '$1 == gpu {print $2}')"
[[ -n "$gpu_uuid" ]] || { echo "[ERROR] physical GPU $GPU not found" >&2; exit 1; }
gpu_pids="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | awk -F', ' -v uuid="$gpu_uuid" '$1 == uuid {print $2}')"
[[ -z "$gpu_pids" ]] || { echo "[COLLISION] GPU $GPU occupied by PIDs: $gpu_pids" >&2; exit 75; }

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

echo "[RUN] label=$LABEL domain=$DOMAIN checkpoint_step=$actual_step GPU=$GPU natural_routing=on"
set +e
CUDA_VISIBLE_DEVICES="$GPU" "$TORCHRUN" --nproc_per_node 1 \
    --master_addr 127.0.0.1 --master_port "$MASTER_PORT" \
    Megatron-LM/pretrain_gpt.py "${model_args[@]}" \
    --transformer-impl local --pipeline-model-parallel-size 1 --expert-model-parallel-size 1 \
    --distributed-timeout-minutes 30 --no-persist-layer-norm --bf16 \
    --micro-batch-size "$MICRO_BATCH_SIZE" --global-batch-size "$MICRO_BATCH_SIZE" \
    --lr 3e-4 --min-lr 3e-5 --lr-decay-style WSD --lr-decay-iters 1 \
    --lr-warmup-fraction 0.0 --lr-wsd-decay-iters 1 --seq-length 512 \
    --data-path 1.0 "$probe_prefix" --split 100,0,0 --train-iters 1 --skip-train \
    --load "$LOAD_DIR" --no-load-optim --no-load-rng --moe-resume-from-num-experts 8 \
    --diagnostic-override-train-iteration 0 --diagnostic-override-consumed-train-samples 0 \
    --eval-interval 1 --probe-name "${DOMAIN}_router_usage" \
    --probe-eval-iters "$PROBE_EVAL_ITERS" --probe-eval-interval 1 \
    --probe-data-path 1.0 "$probe_prefix" --run-initial-probe-eval \
    --probe-router-usage --probe-router-usage-num-existing-experts 8 \
    --tensorboard-dir "$tmp_dir/tensorboard" --tensorboard-log-interval 1 \
    >"$log_path" 2>&1
rc=$?
set -e
if [[ "$rc" -ne 0 ]]; then
    echo "[ERROR] router usage probe failed rc=$rc log=$log_path" >&2
    exit "$rc"
fi
grep -q "probe ${DOMAIN}_router_usage router usage" "$log_path" || {
    echo "[ERROR] router usage line missing: $log_path" >&2; exit 1;
}
mkdir -p "$(dirname "$final_dir")"
mv "$tmp_dir" "$final_dir"
echo "[DONE] $final_dir"
