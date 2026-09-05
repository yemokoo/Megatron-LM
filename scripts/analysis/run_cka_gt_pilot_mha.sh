#!/usr/bin/env bash
set -euo pipefail

# One read-only CKA-pilot worker.  It loads the frozen before/current models on
# one GPU and processes only its deterministic document-window partition.  It
# never trains, writes raw hidden states, or terminates an occupying process.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PY_ENV="${PY_ENV:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100}"
TORCHRUN="${TORCHRUN:-$PY_ENV/bin/torchrun}"
GPU="${GPU:-${1:-}}"
MODE="${MODE:-pass2}"
DOMAIN="${DOMAIN:-code}"
SPLIT="${SPLIT:-all}"
WORKER_INDEX="${WORKER_INDEX:-0}"
WORKER_COUNT="${WORKER_COUNT:-1}"
WINDOW_BATCH_SIZE="${WINDOW_BATCH_SIZE:-4}"
SHARD_WINDOWS="${SHARD_WINDOWS:-64}"
MAX_WINDOWS="${MAX_WINDOWS:-0}"
SEED="${SEED:-1234}"

[[ "$GPU" =~ ^[0-7]$ ]] || { echo "[ERROR] GPU must be in 0..7" >&2; exit 2; }
case "$MODE" in pass1|pass2|router_smoke) ;; *) echo "[ERROR] bad MODE=$MODE" >&2; exit 2 ;; esac
case "$DOMAIN" in code|wiki) ;; *) echo "[ERROR] bad DOMAIN=$DOMAIN" >&2; exit 2 ;; esac
case "$SPLIT" in all|calibration|selection|test) ;; *) echo "[ERROR] bad SPLIT=$SPLIT" >&2; exit 2 ;; esac
[[ "$WORKER_INDEX" =~ ^[0-9]+$ && "$WORKER_COUNT" =~ ^[1-9][0-9]*$ ]] || {
    echo "[ERROR] invalid worker index/count" >&2; exit 2;
}
(( WORKER_INDEX < WORKER_COUNT && WINDOW_BATCH_SIZE > 0 && SHARD_WINDOWS > 0 )) || {
    echo "[ERROR] invalid worker/batch/shard configuration" >&2; exit 2;
}
if [[ "$MODE" == pass1 && ( "$DOMAIN" != wiki || "$SPLIT" != calibration || "$WORKER_COUNT" != 1 ) ]]; then
    echo "[ERROR] pass1 is exactly Wiki calibration with one worker" >&2
    exit 2
fi

OUT_ROOT="${OUT_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_gt_pilot_20260815/analysis/cka_gt_pilot_v1}"
PILOT_CONFIG="${PILOT_CONFIG:-$OUT_ROOT/config.json}"
MEMBERSHIP_STATS="${MEMBERSHIP_STATS:-$OUT_ROOT/membership/wiki_calibration_membership.npz}"
REFERENCE_LOAD="${REFERENCE_LOAD:-/data2/seonghyeonnoh/LLM-continual-learning-runs/flamemoe/ffn_experts_only/code/expansion_kd_init/kd_init/full_training/g2_olddata_kd_9run_20260808__code_e8_to_e16_wiki_kd_step600}"
CURRENT_LOAD="${CURRENT_LOAD:-/data2/seonghyeonnoh/LLM-continual-learning-runs/flamemoe/ffn_experts_only/code/no_replay/lm/full_training/flame_code_bootstrap_20260810__g2-ffn-only-code1800-one-slot-additive-bootstrap-q50s200-b01-persistent-opt-v2}"
CODE_PREFIX="${CODE_PREFIX:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/code/train/train_text_document}"
WIKI_PREFIX="${WIKI_PREFIX:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/wiki/train/train_text_document}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6}"
REFERENCE_STEP="${REFERENCE_STEP:-600}"
CURRENT_STEP="${CURRENT_STEP:-1800}"
MASTER_PORT="${MASTER_PORT:-$((34700 + WORKER_INDEX))}"

for p in "$TORCHRUN" "$PILOT_CONFIG" "$REFERENCE_LOAD/latest_checkpointed_iteration.txt" \
         "$CURRENT_LOAD/latest_checkpointed_iteration.txt" "$CODE_PREFIX.idx" "$CODE_PREFIX.bin" \
         "$WIKI_PREFIX.idx" "$WIKI_PREFIX.bin"; do
    [[ -e "$p" ]] || { echo "[ERROR] missing: $p" >&2; exit 1; }
done
[[ "$(tr -d '[:space:]' < "$REFERENCE_LOAD/latest_checkpointed_iteration.txt")" == "$REFERENCE_STEP" ]] || {
    echo "[ERROR] reference checkpoint is not step $REFERENCE_STEP" >&2; exit 1;
}
[[ "$(tr -d '[:space:]' < "$CURRENT_LOAD/latest_checkpointed_iteration.txt")" == "$CURRENT_STEP" ]] || {
    echo "[ERROR] current checkpoint is not step $CURRENT_STEP" >&2; exit 1;
}
if [[ "$MODE" == pass2 ]]; then
    [[ -f "$MEMBERSHIP_STATS" ]] || { echo "[ERROR] pass2 membership stats missing: $MEMBERSHIP_STATS" >&2; exit 1; }
fi

worker_name="worker_$(printf '%03d' "$WORKER_INDEX")"
worker_dir="$OUT_ROOT/runtime/$MODE/$DOMAIN/$SPLIT/$worker_name"
log_path="$OUT_ROOT/logs/${MODE}_${DOMAIN}_${SPLIT}_${worker_name}.log"
cache_path="$OUT_ROOT/data_cache/$worker_name"
echo "[PLAN] mode=$MODE domain=$DOMAIN split=$SPLIT worker=$WORKER_INDEX/$WORKER_COUNT GPU=$GPU"
echo "[PLAN] before=$REFERENCE_LOAD (step $REFERENCE_STEP)"
echo "[PLAN] after=$CURRENT_LOAD (step $CURRENT_STEP)"
echo "[PLAN] config=$PILOT_CONFIG output=$worker_dir"
if [[ "${PLAN_ONLY:-0}" == 1 ]]; then
    exit 0
fi

# Read-only collision check: an occupied GPU causes a clean refusal.
command -v nvidia-smi >/dev/null 2>&1 || { echo "[ERROR] nvidia-smi unavailable" >&2; exit 1; }
gpu_uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$GPU" '$1 == g {print $2}')"
[[ -n "$gpu_uuid" ]] || { echo "[ERROR] GPU $GPU not found" >&2; exit 1; }
gpu_pids="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | awk -F', ' -v u="$gpu_uuid" '$1 == u {print $2}')"
[[ -z "$gpu_pids" ]] || { echo "[COLLISION] GPU $GPU is occupied by PIDs: $gpu_pids" >&2; exit 75; }

mkdir -p "$worker_dir" "$(dirname "$log_path")" "$cache_path"
export PATH="$PY_ENV/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/Megatron-LM"
export HF_HOME="${HF_HOME:-/data2/seonghyeonnoh/homecache/huggingface}"
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

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

CUDA_VISIBLE_DEVICES="$GPU" "$TORCHRUN" --nproc_per_node 1 \
    --master_addr 127.0.0.1 --master_port "$MASTER_PORT" \
    Megatron-LM/pretrain_gpt.py "${model_args[@]}" \
    --transformer-impl local --pipeline-model-parallel-size 1 --expert-model-parallel-size 1 \
    --distributed-timeout-minutes 30 --no-persist-layer-norm --bf16 \
    --micro-batch-size 1 --global-batch-size 1 --seq-length 512 \
    --lr 3e-4 --min-lr 3e-5 --lr-decay-style WSD --lr-decay-iters 1 \
    --lr-warmup-fraction 0.0 --lr-wsd-decay-iters 1 --seed "$SEED" \
    --dataloader-type single --num-workers 0 --data-cache-path "$cache_path" \
    --data-path 1.0 "$CODE_PREFIX" --split 100,0,0 --train-iters 1 --skip-train \
    --load "$CURRENT_LOAD" --no-load-optim --no-load-rng \
    --moe-resume-from-num-experts 8 \
    --moe-old-model-kl-load "$REFERENCE_LOAD" --moe-old-model-kl-num-experts 16 \
    --diagnostic-override-train-iteration 0 --diagnostic-override-consumed-train-samples 0 \
    --eval-interval 1 --probe-name cka_gt_pilot --probe-eval-iters 1 --probe-eval-interval 1 \
    --probe-data-path 1.0 "$CODE_PREFIX" --run-initial-probe-eval \
    --cka-gt-pilot-path "$worker_dir" --cka-gt-pilot-config "$PILOT_CONFIG" \
    --cka-gt-pilot-mode "$MODE" --cka-gt-pilot-domain "$DOMAIN" --cka-gt-pilot-split "$SPLIT" \
    --cka-gt-pilot-worker-index "$WORKER_INDEX" --cka-gt-pilot-worker-count "$WORKER_COUNT" \
    --cka-gt-pilot-batch-size "$WINDOW_BATCH_SIZE" --cka-gt-pilot-shard-windows "$SHARD_WINDOWS" \
    --cka-gt-pilot-max-windows "$MAX_WINDOWS" \
    --cka-gt-pilot-membership-stats "$MEMBERSHIP_STATS" \
    >> "$log_path" 2>&1

echo "[DONE] mode=$MODE domain=$DOMAIN split=$SPLIT worker=$WORKER_INDEX/$WORKER_COUNT"
