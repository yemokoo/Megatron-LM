#!/usr/bin/env bash
set -euo pipefail

# One read-only, world-size-1 full Code-train CKA census worker.  The before
# and after checkpoints coexist on one H100.  Residual layer outputs 2--9 are
# compared in memory and released after compact streaming statistics are
# updated; raw hidden tensors are never written.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PY_ENV="${PY_ENV:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100}"
TORCHRUN="${TORCHRUN:-$PY_ENV/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-$PY_ENV/bin/python}"
GPU="${GPU:-${1:-}}"
WORKER_INDEX="${WORKER_INDEX:-0}"
WORKER_COUNT="${WORKER_COUNT:-4}"
WINDOW_BATCH_SIZE="${WINDOW_BATCH_SIZE:-32}"
FORWARD_SUBBATCH_SIZE="${FORWARD_SUBBATCH_SIZE:-0}"
CHECKPOINT_EVERY_BATCHES="${CHECKPOINT_EVERY_BATCHES:-100}"
HISTOGRAM_BINS="${HISTOGRAM_BINS:-4096}"
RESERVOIR_SIZE="${RESERVOIR_SIZE:-5000000}"
MAX_WINDOWS="${MAX_WINDOWS:-0}"
BENCHMARK_BATCH_SIZES="${BENCHMARK_BATCH_SIZES:-}"
BENCHMARK_MIN_GAIN="${BENCHMARK_MIN_GAIN:-0.03}"
BENCHMARK_MAX_PEAK_GIB="${BENCHMARK_MAX_PEAK_GIB:-70}"
SEED="${SEED:-1234}"
MASTER_PORT="${MASTER_PORT:-$((36870 + WORKER_INDEX))}"

OUT_ROOT="${OUT_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_conv_census_20260817}"
CENSUS_OUTPUT="${CENSUS_OUTPUT:-$OUT_ROOT/full_census}"
PILOT_ROOT="${PILOT_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_gt_pilot_20260815/analysis/cka_gt_pilot_v1}"
ANALYSIS_CONFIG="${ANALYSIS_CONFIG:-$PILOT_ROOT/analysis_config.json}"
CENSUS_MANIFEST="${CENSUS_MANIFEST:-$OUT_ROOT/manifest/manifest.json}"
REFERENCE_LOAD="${REFERENCE_LOAD:-/data2/seonghyeonnoh/LLM-continual-learning-runs/flamemoe/ffn_experts_only/conversation/expansion_kd_init/kd_init/full_training/conversation_old_like_gt_pipeline_20260812__01_expansion_kd_init_e16_to_e24_step600}"
CURRENT_LOAD="${CURRENT_LOAD:-/data2/seonghyeonnoh/LLM-continual-learning-runs/flamemoe/ffn_experts_only/conversation/replay/lm/full_training/conversation_old_like_gt_pipeline_20260812__02_conversation_wikicode_replay_lm_oracle_step1800}"
CODE_PREFIX="${DATA_PREFIX:-/data2/seonghyeonnoh/LLM-continual-learning-data/conversation_merged_train_20260817/train_text_document}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6}"
REFERENCE_STEP="${REFERENCE_STEP:-600}"
CURRENT_STEP="${CURRENT_STEP:-1800}"
worker_name="worker_$(printf '%03d' "$WORKER_INDEX")"
LOG_PATH="${LOG_PATH:-$CENSUS_OUTPUT/logs/${worker_name}.log}"
CACHE_PATH="${CACHE_PATH:-$OUT_ROOT/data_cache/${worker_name}}"

die() { echo "[ERROR] $*" >&2; exit 1; }

[[ "$GPU" =~ ^[0-7]$ ]] || die "GPU must be one of physical 0..7"
[[ "$WORKER_INDEX" =~ ^[0-9]+$ && "$WORKER_COUNT" =~ ^[1-9][0-9]*$ ]] \
    || die "invalid worker index/count"
(( WORKER_INDEX < WORKER_COUNT )) || die "worker index must be below worker count"
[[ "$WINDOW_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die "WINDOW_BATCH_SIZE must be positive"
[[ "$FORWARD_SUBBATCH_SIZE" =~ ^[0-9]+$ ]] \
    || die "FORWARD_SUBBATCH_SIZE must be nonnegative"
(( FORWARD_SUBBATCH_SIZE == 0 || FORWARD_SUBBATCH_SIZE <= WINDOW_BATCH_SIZE )) \
    || die "FORWARD_SUBBATCH_SIZE must be zero or <= WINDOW_BATCH_SIZE"
[[ "$CHECKPOINT_EVERY_BATCHES" =~ ^[1-9][0-9]*$ ]] \
    || die "CHECKPOINT_EVERY_BATCHES must be positive"
[[ "$HISTOGRAM_BINS" =~ ^[1-9][0-9]*$ ]] && (( HISTOGRAM_BINS >= 200 )) \
    || die "HISTOGRAM_BINS must be at least 200"
[[ "$RESERVOIR_SIZE" =~ ^[1-9][0-9]*$ ]] || die "RESERVOIR_SIZE must be positive"
[[ "$MAX_WINDOWS" =~ ^[0-9]+$ ]] || die "MAX_WINDOWS must be nonnegative"
[[ "$MASTER_PORT" =~ ^[1-9][0-9]*$ ]] && (( MASTER_PORT < 65536 )) \
    || die "invalid MASTER_PORT"

cat <<PLAN
[CKA FULL CENSUS WORKER]
GPU: $GPU
worker: $WORKER_INDEX/$WORKER_COUNT
batch: $WINDOW_BATCH_SIZE
GPU forward subbatch: ${FORWARD_SUBBATCH_SIZE:-0} (0 means journal batch)
max windows: $MAX_WINDOWS (0 means complete assigned partition)
global deterministic token-score reservoir: $RESERVOIR_SIZE
before: $REFERENCE_LOAD (step $REFERENCE_STEP)
after:  $CURRENT_LOAD (step $CURRENT_STEP)
manifest: $CENSUS_MANIFEST
threshold config: $ANALYSIS_CONFIG
output: $CENSUS_OUTPUT
log: $LOG_PATH
port: $MASTER_PORT
in-process benchmark batches: ${BENCHMARK_BATCH_SIZES:-disabled}
PLAN
if [[ "${PLAN_ONLY:-0}" == 1 ]]; then
    exit 0
fi

for path in "$TORCHRUN" "$PYTHON_BIN" "$ANALYSIS_CONFIG" "$CENSUS_MANIFEST" \
        "$REFERENCE_LOAD/latest_checkpointed_iteration.txt" \
        "$CURRENT_LOAD/latest_checkpointed_iteration.txt" \
        "$CODE_PREFIX.idx" "$CODE_PREFIX.bin"; do
    [[ -e "$path" ]] || die "missing required path: $path"
done
[[ "$(tr -d '[:space:]' < "$REFERENCE_LOAD/latest_checkpointed_iteration.txt")" == "$REFERENCE_STEP" ]] \
    || die "reference checkpoint tracker is not step $REFERENCE_STEP"
[[ "$(tr -d '[:space:]' < "$CURRENT_LOAD/latest_checkpointed_iteration.txt")" == "$CURRENT_STEP" ]] \
    || die "current checkpoint tracker is not step $CURRENT_STEP"

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi unavailable"
gpu_uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$GPU" '$1 == g {print $2}')"
[[ -n "$gpu_uuid" ]] || die "physical GPU $GPU not found"
gpu_pids="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | awk -F', ' -v u="$gpu_uuid" '$1 == u {print $2}')"
[[ -z "$gpu_pids" ]] || { echo "[COLLISION] GPU $GPU occupied by PIDs: $gpu_pids" >&2; exit 75; }
"$PYTHON_BIN" - "$MASTER_PORT" <<'PY'
import socket, sys
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
finally:
    sock.close()
PY

mkdir -p "$CENSUS_OUTPUT/logs" "$CACHE_PATH"
export PATH="$PY_ENV/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/Megatron-LM"
export HF_HOME="${HF_HOME:-/data2/seonghyeonnoh/homecache/huggingface}"
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

model_args=(
    --hidden-size 1024 --ffn-hidden-size 5472 --num-layers 9
    --num-attention-heads 16 --group-query-attention --num-query-groups 16
    --swiglu --max-position-embeddings 2048 --normalization RMSNorm --norm-epsilon 1e-6
    --untie-embeddings-and-output-weights --position-embedding-type rope --disable-bias-linear
    --moe-ffn-hidden-size 352 --num-experts 24 --moe-router-topk 4
    --moe-layer-freq '[0]*1+[1]*8' --moe-router-dtype fp32 --moe-router-pre-softmax
    --moe-router-score-function softmax --moe-aux-loss-coeff 0.01 --moe-z-loss-coeff 0.001
    --hidden-dropout 0.0 --attention-dropout 0.0 --init-method-std 0.02
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$TOKENIZER_MODEL"
)

benchmark_args=()
if [[ -n "$BENCHMARK_BATCH_SIZES" ]]; then
    benchmark_args+=(
        --cka-gt-full-census-benchmark-batch-sizes "$BENCHMARK_BATCH_SIZES"
        --cka-gt-full-census-benchmark-min-gain "$BENCHMARK_MIN_GAIN"
        --cka-gt-full-census-benchmark-max-peak-gib "$BENCHMARK_MAX_PEAK_GIB"
    )
fi

CUDA_VISIBLE_DEVICES="$GPU" "$TORCHRUN" --nproc_per_node 1 \
    --master_addr 127.0.0.1 --master_port "$MASTER_PORT" \
    Megatron-LM/pretrain_gpt.py "${model_args[@]}" \
    --transformer-impl local --pipeline-model-parallel-size 1 --expert-model-parallel-size 1 \
    --distributed-timeout-minutes 30 --no-persist-layer-norm --bf16 \
    --micro-batch-size 1 --global-batch-size 1 --seq-length 512 \
    --lr 3e-4 --min-lr 3e-5 --lr-decay-style WSD --lr-decay-iters 1 \
    --lr-warmup-fraction 0.0 --lr-wsd-decay-iters 1 --seed "$SEED" \
    --dataloader-type single --num-workers 0 --data-cache-path "$CACHE_PATH" \
    --data-path 1.0 "$CODE_PREFIX" --split 100,0,0 --train-iters 1 --skip-train \
    --load "$CURRENT_LOAD" --no-load-optim --no-load-rng \
    --moe-resume-from-num-experts 16 \
    --moe-old-model-kl-load "$REFERENCE_LOAD" --moe-old-model-kl-num-experts 24 \
    --diagnostic-override-train-iteration 0 --diagnostic-override-consumed-train-samples 0 \
    --eval-interval 1 --probe-name cka_conv_census --probe-eval-iters 1 \
    --probe-eval-interval 1 --probe-data-path 1.0 "$CODE_PREFIX" --run-initial-probe-eval \
    --cka-gt-full-census-path "$CENSUS_OUTPUT" \
    --cka-gt-full-census-config "$ANALYSIS_CONFIG" \
    --cka-gt-full-census-manifest "$CENSUS_MANIFEST" \
    --cka-gt-full-census-worker-index "$WORKER_INDEX" \
    --cka-gt-full-census-worker-count "$WORKER_COUNT" \
    --cka-gt-full-census-batch-size "$WINDOW_BATCH_SIZE" \
    --cka-gt-full-census-forward-subbatch-size "$FORWARD_SUBBATCH_SIZE" \
    --cka-gt-full-census-checkpoint-every-batches "$CHECKPOINT_EVERY_BATCHES" \
    --cka-gt-full-census-histogram-bins "$HISTOGRAM_BINS" \
    --cka-gt-full-census-reservoir-size "$RESERVOIR_SIZE" \
    --cka-gt-full-census-max-windows "$MAX_WINDOWS" \
    "${benchmark_args[@]}" \
    >> "$LOG_PATH" 2>&1

echo "[DONE] CKA census worker=$WORKER_INDEX/$WORKER_COUNT GPU=$GPU batch=$WINDOW_BATCH_SIZE"
