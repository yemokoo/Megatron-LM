#!/usr/bin/env bash
set -euo pipefail

# Exact, read-only second pass over B-candidate Code windows.  One process on
# one GPU holds the frozen before/after models, computes B/T/M in fp32, and
# emits only packed token masks plus occurrence identities for bundles
# 95/97/99.  It neither opens the sealed pilot test split nor locks a bundle.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PY_ENV="${PY_ENV:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100}"
TORCHRUN="${TORCHRUN:-$PY_ENV/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-$PY_ENV/bin/python}"
GPU="${GPU:-${1:-4}}"
WINDOW_BATCH_SIZE="${WINDOW_BATCH_SIZE:-192}"
FORWARD_SUBBATCH_SIZE="${FORWARD_SUBBATCH_SIZE:-128}"
CHECKPOINT_EVERY_BATCHES="${CHECKPOINT_EVERY_BATCHES:-10}"
SEED="${SEED:-1234}"
MASTER_PORT="${MASTER_PORT:-35784}"

OUT_ROOT="${OUT_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_gt_full_census_20260816/analysis/cka_gt_full_census_v1}"
CENSUS_ROOT="${CENSUS_ROOT:-$OUT_ROOT/full_census}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-$OUT_ROOT/gt_candidates_b99_v2}"
TARGET_OUTPUT="${TARGET_OUTPUT:-$CENSUS_ROOT/exact_targeted_v2}"
PILOT_ROOT="${PILOT_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_gt_pilot_20260815/analysis/cka_gt_pilot_v1}"
ANALYSIS_CONFIG="${ANALYSIS_CONFIG:-$PILOT_ROOT/analysis_config.json}"
CANDIDATE_MANIFEST="${CANDIDATE_MANIFEST:-$CANDIDATE_ROOT/manifest.json}"
CANDIDATE_WINDOWS="${CANDIDATE_WINDOWS:-$CANDIDATE_ROOT/candidate_windows.npy}"
CENSUS_SUMMARY="${CENSUS_SUMMARY:-$CENSUS_ROOT/summary.json}"
AUTHORITATIVE_B="${AUTHORITATIVE_B:-$OUT_ROOT/gt_candidates_b99_v2_authoritative_b_v1/authoritative_b_masks.npz}"
REFERENCE_LOAD="${REFERENCE_LOAD:-/data2/seonghyeonnoh/LLM-continual-learning-runs/flamemoe/ffn_experts_only/code/expansion_kd_init/kd_init/full_training/g2_olddata_kd_9run_20260808__code_e8_to_e16_wiki_kd_step600}"
CURRENT_LOAD="${CURRENT_LOAD:-/data2/seonghyeonnoh/LLM-continual-learning-runs/flamemoe/ffn_experts_only/code/no_replay/lm/full_training/flame_code_bootstrap_20260810__g2-ffn-only-code1800-one-slot-additive-bootstrap-q50s200-b01-persistent-opt-v2}"
CODE_PREFIX="${CODE_PREFIX:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/code/train/train_text_document}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6}"
REFERENCE_STEP="${REFERENCE_STEP:-600}"
CURRENT_STEP="${CURRENT_STEP:-1800}"
LOG_PATH="${LOG_PATH:-$TARGET_OUTPUT/logs/targeted_gpu${GPU}.log}"
CACHE_PATH="${CACHE_PATH:-$OUT_ROOT/data_cache/targeted_gpu${GPU}}"

die() { echo "[ERROR] $*" >&2; exit 1; }

[[ "$GPU" =~ ^[0-7]$ ]] || die "GPU must be physical 0..7"
[[ "$WINDOW_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die "WINDOW_BATCH_SIZE must be positive"
[[ "$FORWARD_SUBBATCH_SIZE" =~ ^[0-9]+$ ]] || die "FORWARD_SUBBATCH_SIZE must be nonnegative"
(( FORWARD_SUBBATCH_SIZE == 0 || FORWARD_SUBBATCH_SIZE <= WINDOW_BATCH_SIZE )) \
    || die "FORWARD_SUBBATCH_SIZE must be zero or <= WINDOW_BATCH_SIZE"
[[ "$CHECKPOINT_EVERY_BATCHES" =~ ^[1-9][0-9]*$ ]] \
    || die "CHECKPOINT_EVERY_BATCHES must be positive"
[[ "$MASTER_PORT" =~ ^[1-9][0-9]*$ ]] && (( MASTER_PORT < 65536 )) \
    || die "invalid MASTER_PORT"

cat <<PLAN
[CKA EXACT TARGETED GT]
GPU: $GPU (world size 1)
batch / forward subbatch: $WINDOW_BATCH_SIZE / $FORWARD_SUBBATCH_SIZE
before: $REFERENCE_LOAD (step $REFERENCE_STEP)
after:  $CURRENT_LOAD (step $CURRENT_STEP)
B candidates: $CANDIDATE_WINDOWS
candidate manifest: $CANDIDATE_MANIFEST
frozen bundles: $ANALYSIS_CONFIG
full denominator: $CENSUS_SUMMARY
authoritative B: $AUTHORITATIVE_B
output: $TARGET_OUTPUT
log: $LOG_PATH
port: $MASTER_PORT
policy: exact 95/97/99 outputs; no threshold lock; sealed test remains closed
PLAN
if [[ "${PLAN_ONLY:-0}" == 1 ]]; then
    exit 0
fi

for path in "$TORCHRUN" "$PYTHON_BIN" "$ANALYSIS_CONFIG" "$CANDIDATE_MANIFEST" \
        "$CANDIDATE_WINDOWS" "$CENSUS_SUMMARY" "$AUTHORITATIVE_B" \
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

mkdir -p "$TARGET_OUTPUT/logs" "$CACHE_PATH"
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
    --dataloader-type single --num-workers 0 --data-cache-path "$CACHE_PATH" \
    --data-path 1.0 "$CODE_PREFIX" --split 100,0,0 --train-iters 1 --skip-train \
    --load "$CURRENT_LOAD" --no-load-optim --no-load-rng \
    --moe-resume-from-num-experts 8 \
    --moe-old-model-kl-load "$REFERENCE_LOAD" --moe-old-model-kl-num-experts 16 \
    --diagnostic-override-train-iteration 0 --diagnostic-override-consumed-train-samples 0 \
    --eval-interval 1 --probe-name cka_gt_targeted --probe-eval-iters 1 \
    --probe-eval-interval 1 --probe-data-path 1.0 "$CODE_PREFIX" --run-initial-probe-eval \
    --cka-gt-targeted-path "$TARGET_OUTPUT" \
    --cka-gt-targeted-config "$ANALYSIS_CONFIG" \
    --cka-gt-targeted-candidate-manifest "$CANDIDATE_MANIFEST" \
    --cka-gt-targeted-candidate-windows "$CANDIDATE_WINDOWS" \
    --cka-gt-targeted-census-summary "$CENSUS_SUMMARY" \
    --cka-gt-targeted-authoritative-b "$AUTHORITATIVE_B" \
    --cka-gt-targeted-batch-size "$WINDOW_BATCH_SIZE" \
    --cka-gt-targeted-forward-subbatch-size "$FORWARD_SUBBATCH_SIZE" \
    --cka-gt-targeted-checkpoint-every-batches "$CHECKPOINT_EVERY_BATCHES" \
    >> "$LOG_PATH" 2>&1

echo "[DONE] exact targeted CKA GT GPU=$GPU bundles=95,97,99 (not locked)"
