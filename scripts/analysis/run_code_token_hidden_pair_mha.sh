#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Read-only paired forward for one contiguous task-train partition.  Each
# invocation owns one physical GPU and loads both checkpoints in one process,
# so reference/current metrics are always computed from the exact same batch.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PY_ENV="${PY_ENV:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100}"
PYTHON_BIN="${PYTHON_BIN:-$PY_ENV/bin/python}"
TORCHRUN="${TORCHRUN:-$PY_ENV/bin/torchrun}"

GPU="${GPU:-${1:-}}"
WORKER_INDEX="${WORKER_INDEX:-${2:-}}"
WORKER_COUNT="${WORKER_COUNT:-8}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-4147200}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-512}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-48}"
SHARD_SAMPLES="${SHARD_SAMPLES:-4800}"
RESERVOIR_SIZE="${RESERVOIR_SIZE:-512}"
SEED="${SEED:-1234}"

[[ "$GPU" =~ ^[0-7]$ ]] || {
    echo "[ERROR] set GPU to one physical GPU index in 0..7" >&2
    exit 2
}
[[ "$WORKER_INDEX" =~ ^[0-9]+$ ]] || {
    echo "[ERROR] set WORKER_INDEX to an integer in [0, WORKER_COUNT)" >&2
    exit 2
}
[[ "$WORKER_COUNT" =~ ^[1-9][0-9]*$ ]] || {
    echo "[ERROR] WORKER_COUNT must be positive" >&2
    exit 2
}
(( WORKER_INDEX < WORKER_COUNT )) || {
    echo "[ERROR] WORKER_INDEX=$WORKER_INDEX is outside WORKER_COUNT=$WORKER_COUNT" >&2
    exit 2
}
(( TOTAL_SAMPLES > 0 && SEQUENCE_LENGTH > 0 && MICRO_BATCH_SIZE > 0 )) || {
    echo "[ERROR] sample, sequence, and batch sizes must be positive" >&2
    exit 2
}
(( SHARD_SAMPLES > 0 && RESERVOIR_SIZE >= 0 )) || {
    echo "[ERROR] invalid shard/reservoir size" >&2
    exit 2
}

# General quotient/remainder partitioning keeps ranges contiguous and complete,
# including when TOTAL_SAMPLES is not divisible by WORKER_COUNT.
samples_per_worker=$(( TOTAL_SAMPLES / WORKER_COUNT ))
remainder=$(( TOTAL_SAMPLES % WORKER_COUNT ))
if (( WORKER_INDEX < remainder )); then
    partition_samples=$(( samples_per_worker + 1 ))
    partition_start=$(( WORKER_INDEX * samples_per_worker + WORKER_INDEX ))
else
    partition_samples=$samples_per_worker
    partition_start=$(( WORKER_INDEX * samples_per_worker + remainder ))
fi
partition_end=$(( partition_start + partition_samples ))
(( partition_samples > 0 )) || {
    echo "[ERROR] worker partition is empty; TOTAL_SAMPLES must be at least WORKER_COUNT" >&2
    exit 2
}
(( partition_start % MICRO_BATCH_SIZE == 0 )) || {
    echo "[ERROR] partition start $partition_start must align to micro batch $MICRO_BATCH_SIZE" >&2
    exit 2
}

REFERENCE_LOAD="${REFERENCE_LOAD:-/data2/seonghyeonnoh/LLM-continual-learning-runs/g2_olddata_kd_9run_20260808/01_common_kd_init/code_e8_to_e16_wiki_kd_step600}"
CURRENT_LOAD="${CURRENT_LOAD:-/data2/seonghyeonnoh/LLM-continual-learning-runs/flame_code_bootstrap_20260810/g2-ffn-only-code1800-one-slot-additive-bootstrap-q50s200-b01-persistent-opt-v2}"
REFERENCE_REQUIRED_STEP="${REFERENCE_REQUIRED_STEP:-600}"
CURRENT_REQUIRED_STEP="${CURRENT_REQUIRED_STEP:-1800}"
TASK_NAME="${TASK_NAME:-code}"
DATASET_DIR="${DATASET_DIR:-}"
DATASET_PREFIX="${DATASET_PREFIX:-${CODE_PREFIX:-}}"
DATASET_BLEND_MODE="${DATASET_BLEND_MODE:-equal_weights}"
if [[ -z "$DATASET_DIR" && -z "$DATASET_PREFIX" ]]; then
    DATASET_PREFIX=/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/code/train/train_text_document
fi
NUM_EXPERTS="${NUM_EXPERTS:-16}"
RESUME_FROM_NUM_EXPERTS="${RESUME_FROM_NUM_EXPERTS:-8}"
REFERENCE_NUM_EXPERTS="${REFERENCE_NUM_EXPERTS:-$NUM_EXPERTS}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6}"
OUT_ROOT="${OUT_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/code_token_representation_gt_20260811}"
PAIR_LABEL="${PAIR_LABEL:-code_train_post_wiki_kd_init_step600_vs_code_wiki_replay_one_phase_step1800}"
PROBE_NAME="${PROBE_NAME:-${TASK_NAME}_token_hidden_pair}"
MASTER_PORT="${MASTER_PORT:-$((32900 + WORKER_INDEX))}"

rank_name="$(printf 'rank_%03d' "$WORKER_INDEX")"
worker_dir="$OUT_ROOT/$rank_name"
log_dir="$OUT_ROOT/logs"
log_path="$log_dir/${rank_name}.log"
data_cache_path="$OUT_ROOT/data_cache_omp8_noworkers/$rank_name"

for executable in "$PYTHON_BIN" "$TORCHRUN"; do
    [[ -x "$executable" ]] || {
        echo "[ERROR] executable missing: $executable" >&2
        exit 1
    }
done
for checkpoint in "$REFERENCE_LOAD" "$CURRENT_LOAD"; do
    [[ -f "$checkpoint/latest_checkpointed_iteration.txt" ]] || {
        echo "[ERROR] checkpoint tracker missing: $checkpoint" >&2
        exit 1
    }
done
reference_step="$(tr -d '[:space:]' < "$REFERENCE_LOAD/latest_checkpointed_iteration.txt")"
current_step="$(tr -d '[:space:]' < "$CURRENT_LOAD/latest_checkpointed_iteration.txt")"
[[ "$reference_step" == "$REFERENCE_REQUIRED_STEP" ]] || {
    echo "[ERROR] reference tracker is $reference_step, expected $REFERENCE_REQUIRED_STEP: $REFERENCE_LOAD" >&2
    exit 1
}
[[ "$current_step" == "$CURRENT_REQUIRED_STEP" ]] || {
    echo "[ERROR] current tracker is $current_step, expected $CURRENT_REQUIRED_STEP: $CURRENT_LOAD" >&2
    exit 1
}
data_path_args=()
if [[ -n "$DATASET_DIR" ]]; then
    [[ -d "$DATASET_DIR" ]] || { echo "[ERROR] dataset directory missing: $DATASET_DIR" >&2; exit 1; }
    dataset_bins=("$DATASET_DIR"/*.bin)
    (( ${#dataset_bins[@]} > 0 )) || { echo "[ERROR] no indexed shards in $DATASET_DIR" >&2; exit 1; }
    for bin_path in "${dataset_bins[@]}"; do
        prefix="${bin_path%.bin}"
        [[ -s "$prefix.idx" ]] || { echo "[ERROR] indexed shard missing: $prefix.idx" >&2; exit 1; }
        case "$DATASET_BLEND_MODE" in
            equal_weights) data_path_args+=(1.0 "$prefix") ;;
            exhaustive) data_path_args+=("$prefix") ;;
            *) echo "[ERROR] DATASET_BLEND_MODE must be equal_weights or exhaustive" >&2; exit 2 ;;
        esac
    done
else
    [[ -f "$DATASET_PREFIX.bin" && -f "$DATASET_PREFIX.idx" ]] || {
        echo "[ERROR] indexed corpus missing: $DATASET_PREFIX.{bin,idx}" >&2
        exit 1
    }
    data_path_args+=(1.0 "$DATASET_PREFIX")
fi
[[ -d "$TOKENIZER_MODEL" ]] || {
    echo "[ERROR] tokenizer snapshot missing: $TOKENIZER_MODEL" >&2
    exit 1
}

mkdir -p "$worker_dir" "$log_dir" "$data_cache_path"

# A final metadata file is written only after all shards are committed.  Skip a
# completed, exactly matching partition before paying the checkpoint-load cost;
# refuse mismatched final metadata instead of silently overwriting it.
if [[ -f "$worker_dir/metadata.json" ]]; then
    if "$PYTHON_BIN" -c '
import json, sys
p, start, count, total, wi, wc, ref, cur, ref_step, cur_step, seq = sys.argv[1:]
with open(p, encoding="utf-8") as f:
    m = json.load(f)
expected = {
    "completed": True,
    "partition_start_sample": int(start),
    "partition_samples": int(count),
    "total_samples": int(total),
    "worker_index": int(wi),
    "worker_count": int(wc),
    "reference_load": ref,
    "current_load": cur,
    "reference_tracker_step": int(ref_step),
    "current_tracker_step": int(cur_step),
    "sequence_length": int(seq),
    "layers": list(range(1, 10)),
}
sys.exit(0 if all(m.get(k) == v for k, v in expected.items()) else 1)
' "$worker_dir/metadata.json" "$partition_start" "$partition_samples" \
        "$TOTAL_SAMPLES" "$WORKER_INDEX" "$WORKER_COUNT" \
        "$REFERENCE_LOAD" "$CURRENT_LOAD" \
        "$REFERENCE_REQUIRED_STEP" "$CURRENT_REQUIRED_STEP" "$SEQUENCE_LENGTH"; then
        echo "[SKIP] completed exact partition $rank_name [$partition_start,$partition_end)"
        exit 0
    fi
    echo "[ERROR] completed metadata does not match requested partition/config: $worker_dir/metadata.json" >&2
    exit 1
fi

echo "[PLAN] $rank_name GPU=$GPU samples=[$partition_start,$partition_end) count=$partition_samples"
echo "[PLAN] reference=$REFERENCE_LOAD (step $reference_step)"
echo "[PLAN] current=$CURRENT_LOAD (step $current_step)"
if [[ "$DATASET_BLEND_MODE" == exhaustive ]]; then
    data_shard_count=${#data_path_args[@]}
else
    data_shard_count=$((${#data_path_args[@]} / 2))
fi
echo "[PLAN] task=$TASK_NAME data_shards=$data_shard_count blend_mode=$DATASET_BLEND_MODE experts=$NUM_EXPERTS reference_experts=$REFERENCE_NUM_EXPERTS"
echo "[PLAN] output=$worker_dir log=$log_path"
if [[ "${PLAN_ONLY:-0}" == 1 ]]; then
    exit 0
fi

# Never kill or reuse an occupied GPU.  The top-level launcher performs the same
# all-GPU preflight so a collision normally prevents every worker from starting.
command -v nvidia-smi >/dev/null 2>&1 || {
    echo "[ERROR] nvidia-smi is unavailable" >&2
    exit 1
}
gpu_uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v gpu="$GPU" '$1 == gpu {print $2}')"
[[ -n "$gpu_uuid" ]] || {
    echo "[ERROR] physical GPU $GPU was not found" >&2
    exit 1
}
gpu_pids="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | awk -F', ' -v uuid="$gpu_uuid" '$1 == uuid {print $2}')"
[[ -z "$gpu_pids" ]] || {
    echo "[COLLISION] GPU $GPU is occupied by PIDs: $gpu_pids" >&2
    exit 75
}

export PATH="$PY_ENV/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/Megatron-LM"
export HF_HOME="${HF_HOME:-/data2/seonghyeonnoh/homecache/huggingface}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

model_args=(
    --hidden-size 1024 --ffn-hidden-size 5472 --num-layers 9
    --num-attention-heads 16 --group-query-attention --num-query-groups 16
    --swiglu --max-position-embeddings 2048 --normalization RMSNorm --norm-epsilon 1e-6
    --untie-embeddings-and-output-weights --position-embedding-type rope --disable-bias-linear
    --moe-ffn-hidden-size 352 --num-experts "$NUM_EXPERTS" --moe-router-topk 4
    --moe-layer-freq '[0]*1+[1]*8' --moe-router-dtype fp32 --moe-router-pre-softmax
    --moe-router-score-function softmax --moe-aux-loss-coeff 0.01 --moe-z-loss-coeff 0.001
    --hidden-dropout 0.0 --attention-dropout 0.0 --init-method-std 0.02
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$TOKENIZER_MODEL"
)

{
    echo "[$(date --iso-8601=seconds)] START $rank_name GPU=$GPU samples=[$partition_start,$partition_end)"
    echo "reference=$REFERENCE_LOAD step=$reference_step"
    echo "current=$CURRENT_LOAD step=$current_step"
} >> "$log_path"

set +e
CUDA_VISIBLE_DEVICES="$GPU" "$TORCHRUN" --nproc_per_node 1 \
    --master_addr 127.0.0.1 --master_port "$MASTER_PORT" \
    Megatron-LM/pretrain_gpt.py "${model_args[@]}" \
    --transformer-impl local --pipeline-model-parallel-size 1 --expert-model-parallel-size 1 \
    --distributed-timeout-minutes 30 --no-persist-layer-norm --bf16 \
    --micro-batch-size "$MICRO_BATCH_SIZE" --global-batch-size "$MICRO_BATCH_SIZE" \
    --lr 3e-4 --min-lr 3e-5 --lr-decay-style WSD --lr-decay-iters 1 \
    --lr-warmup-fraction 0.0 --lr-wsd-decay-iters 1 --seq-length "$SEQUENCE_LENGTH" \
    --seed "$SEED" --dataloader-type single --num-workers 0 \
    --data-cache-path "$data_cache_path" \
    --data-path "${data_path_args[@]}" --split 100,0,0 --train-iters 1 --skip-train \
    --load "$CURRENT_LOAD" --no-load-optim --no-load-rng \
    --moe-resume-from-num-experts "$RESUME_FROM_NUM_EXPERTS" \
    --moe-old-model-kl-load "$REFERENCE_LOAD" --moe-old-model-kl-num-experts "$REFERENCE_NUM_EXPERTS" \
    --diagnostic-override-train-iteration 0 --diagnostic-override-consumed-train-samples 0 \
    --eval-interval 1 --probe-name "$PROBE_NAME" \
    --probe-eval-iters 1 --probe-eval-interval 1 \
    --probe-data-path "${data_path_args[@]}" --run-initial-probe-eval \
    --code-token-hidden-pair-path "$worker_dir" \
    --code-token-hidden-pair-total-samples "$TOTAL_SAMPLES" \
    --code-token-hidden-pair-start-sample "$partition_start" \
    --code-token-hidden-pair-samples "$partition_samples" \
    --code-token-hidden-pair-shard-samples "$SHARD_SAMPLES" \
    --code-token-hidden-pair-layers 1,2,3,4,5,6,7,8,9 \
    --code-token-hidden-pair-reservoir-size "$RESERVOIR_SIZE" \
    --code-token-hidden-pair-worker-index "$WORKER_INDEX" \
    --code-token-hidden-pair-worker-count "$WORKER_COUNT" \
    --code-token-hidden-pair-label "$PAIR_LABEL" \
    >> "$log_path" 2>&1
rc=$?
set -e

if (( rc != 0 )); then
    echo "[ERROR] $rank_name failed rc=$rc; inspect $log_path" >&2
    exit "$rc"
fi
[[ -f "$worker_dir/metadata.json" ]] || {
    echo "[ERROR] process exited without final metadata: $worker_dir/metadata.json" >&2
    exit 1
}
echo "[$(date --iso-8601=seconds)] DONE $rank_name" >> "$log_path"
echo "[DONE] $rank_name [$partition_start,$partition_end) -> $worker_dir"
