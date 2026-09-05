#!/usr/bin/env bash
set -euo pipefail

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/../.." && pwd)"
cd "$R"

PY_ENV="${PY_ENV:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100}"
TORCHRUN="${TORCHRUN:-$PY_ENV/bin/torchrun}"
GPU="${GPU:-5}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29871}"
OUT_ROOT="${OUT_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/fingerprint_router_geometry_20260809}"
SOURCE_ROOT="${SOURCE_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/g2_olddata_kd_9run_20260808}"
DATA_ROOT="${DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6}"

SAMPLES="${SAMPLES:-64}"
TOKENS_PER_SAMPLE="${TOKENS_PER_SAMPLE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-$((SAMPLES / MICRO_BATCH_SIZE))}"
MAX_TOKENS="${MAX_TOKENS:-$((SAMPLES * TOKENS_PER_SAMPLE))}"
LAYERS="${LAYERS:-2,5,9}"
# This fixed 9-layer recipe has one dense first layer and MoE/router layers
# 2--9 (moe_layer_freq=[0,1,1,1,1,1,1,1,1]).  Detailed routing arrays are
# undefined for the dense layer, so "all" means all router-bearing layers.
if [[ "$LAYERS" == "all" ]]; then
    LAYERS="2,3,4,5,6,7,8,9"
fi
DOMAINS="${DOMAINS:-wiki code}"
STAGES="${STAGES:-A B C_vocabkl}"

A_DIR="$SOURCE_ROOT/00_sources/wiki_ffn_only_e8_step1800"
B_DIR="$SOURCE_ROOT/01_common_kd_init/code_e8_to_e16_wiki_kd_step600"
C_DIR="$SOURCE_ROOT/02_nine_runs/local/weights/a100/mha/g2-checkpoints/code/joint_old_data_kd/post_kd_teacher_c10_fixed/r2-code-vocabkl-c10-fixed-post16-mb48-1800-probe3i100"
G_DIR="$SOURCE_ROOT/02_nine_runs/local/weights/a100/mha/g2-checkpoints/conversation/expansion_distill_init_3objective_c10_l2to9/vocab_kl/r3-conv-expand-outputkd-c1-vocabkl-branch-e16to24-mb32-600-probe3i100"
H_DIR="$SOURCE_ROOT/02_nine_runs/local/weights/a100/mha/g2-checkpoints/conversation/joint_old_data_kd/post_kd_teacher_c10_fixed/r4-conv-vocabkl-c10-fixed-post24-mb36-1800-probe3i100"

export PATH="$PY_ENV/bin:${PATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$R/Megatron-LM:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/data2/seonghyeonnoh/homecache/huggingface}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1

checkpoint_step() {
    local path="$1" expected="$2" label="$3" actual=""
    if [[ -f "$path/latest_checkpointed_iteration.txt" ]]; then
        actual="$(tr -d '[:space:]' < "$path/latest_checkpointed_iteration.txt")"
    fi
    if [[ "$actual" != "$expected" ]]; then
        echo "[ERROR] $label expected step $expected, got ${actual:-missing}: $path" >&2
        return 1
    fi
}

stage_path() {
    case "$1" in
        A) printf '%s\n' "$A_DIR" ;;
        B) printf '%s\n' "$B_DIR" ;;
        C_vocabkl) printf '%s\n' "$C_DIR" ;;
        F) printf '%s\n' "$C_DIR" ;;
        G) printf '%s\n' "$G_DIR" ;;
        H) printf '%s\n' "$H_DIR" ;;
        *) echo "[ERROR] unsupported stage: $1" >&2; return 1 ;;
    esac
}

stage_experts() {
    case "$1" in
        A) printf '8\n' ;;
        B|C_vocabkl|F) printf '16\n' ;;
        G|H) printf '24\n' ;;
        *) return 1 ;;
    esac
}

stage_source_experts() {
    case "$1" in
        A) printf '0\n' ;;
        B|C_vocabkl|F) printf '8\n' ;;
        G|H) printf '16\n' ;;
        *) return 1 ;;
    esac
}

domain_probe_prefix() {
    case "$1" in
        wiki|code) printf '%s\n' "$DATA_ROOT/$1/test/test_text_document" ;;
        conversation) printf '%s\n' "$DATA_ROOT/conversation/test/shard_00000_text_document" ;;
        *) echo "[ERROR] unsupported domain: $1" >&2; return 1 ;;
    esac
}

gpu_uuid() {
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
        | awk -F', ' -v gpu="$GPU" '$1 == gpu {print $2}'
}

gpu_pids() {
    local uuid="$1"
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits \
        | awk -F', ' -v uuid="$uuid" '$1 == uuid {print $2}'
}

assert_gpu_free() {
    local uuid="$1" pids
    pids="$(gpu_pids "$uuid")"
    if [[ -n "$pids" ]]; then
        echo "[COLLISION] physical GPU $GPU ($uuid) is occupied by PIDs: $pids" >&2
        return 75
    fi
}

watch_job() {
    local launcher_pid="$1" uuid="$2" marker="$3"
    local foreign pid pid_pgid cmd status=0
    while kill -0 "$launcher_pid" 2>/dev/null; do
        foreign=""
        while read -r pid; do
            [[ -n "$pid" ]] || continue
            pid_pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
            [[ "$pid_pgid" == "$launcher_pid" ]] && continue
            cmd="$(ps -ww -o args= -p "$pid" 2>/dev/null || true)"
            # nvidia-smi can retain a just-exited CUDA PID for one polling
            # interval after /proc has disappeared.  It is not a new job.
            [[ -n "$cmd" ]] || continue
            case "$cmd" in
                *pretrain_gpt.py*"$marker"*) ;;
                *) foreign+=" $pid" ;;
            esac
        done < <(gpu_pids "$uuid")
        if [[ -n "$foreign" ]]; then
            echo "[COLLISION] foreign evaluation appeared on GPU $GPU; stopping only analysis process group. PIDs:$foreign" >&2
            kill -TERM -- "-$launcher_pid" 2>/dev/null || true
            status=75
            break
        fi
        sleep 5
    done
    set +e
    wait "$launcher_pid"
    local rc=$?
    set -e
    if [[ "$status" -ne 0 ]]; then
        return "$status"
    fi
    return "$rc"
}

run_dump() {
    local stage="$1" domain="$2" port="$3"
    local load_dir experts source_experts out_dir out_npz tmp_npz log_path probe_prefix uuid
    local resume_args=()
    load_dir="$(stage_path "$stage")"
    experts="$(stage_experts "$stage")"
    source_experts="$(stage_source_experts "$stage")"
    out_dir="$OUT_ROOT/dumps/$domain"
    out_npz="$out_dir/${stage}.npz"
    tmp_npz="$out_dir/.${stage}.inprogress.npz"
    log_path="$OUT_ROOT/logs/dump_${domain}_${stage}.log"
    probe_prefix="$(domain_probe_prefix "$domain")"
    uuid="$(gpu_uuid)"

    if [[ -s "$out_npz" && "${FORCE_DUMP:-0}" != 1 ]]; then
        echo "[SKIP] $out_npz"
        return 0
    fi
    if [[ -e "$tmp_npz" ]]; then
        mv "$tmp_npz" "$tmp_npz.failed.$(date +%Y%m%d-%H%M%S)"
    fi
    if [[ "$source_experts" -gt 0 ]]; then
        resume_args=(--moe-resume-from-num-experts "$source_experts")
    fi

    assert_gpu_free "$uuid"
    echo "[RUN] stage=$stage domain=$domain physical_gpu=$GPU uuid=$uuid experts=$experts"

    local model_args=(
        --hidden-size 1024
        --ffn-hidden-size 5472
        --num-layers 9
        --num-attention-heads 16
        --group-query-attention
        --num-query-groups 16
        --swiglu
        --max-position-embeddings 2048
        --normalization RMSNorm
        --norm-epsilon 1e-6
        --untie-embeddings-and-output-weights
        --position-embedding-type rope
        --disable-bias-linear
        --moe-ffn-hidden-size 352
        --num-experts "$experts"
        --moe-router-topk 4
        --moe-layer-freq '[0]*1+[1]*8'
        --moe-router-dtype fp32
        --moe-router-pre-softmax
        --moe-router-score-function softmax
        --moe-aux-loss-coeff 0.01
        --moe-z-loss-coeff 0.001
        --hidden-dropout 0.0
        --attention-dropout 0.0
        --init-method-std 0.02
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model "$TOKENIZER_MODEL"
    )

    setsid env CUDA_VISIBLE_DEVICES="$GPU" "$TORCHRUN" \
        --nproc_per_node 1 \
        --master_addr 127.0.0.1 \
        --master_port "$port" \
        Megatron-LM/pretrain_gpt.py \
        "${model_args[@]}" \
        --transformer-impl local \
        --pipeline-model-parallel-size 1 \
        --expert-model-parallel-size 1 \
        --distributed-timeout-minutes 30 \
        --no-persist-layer-norm \
        --bf16 \
        --micro-batch-size "$MICRO_BATCH_SIZE" \
        --global-batch-size "$MICRO_BATCH_SIZE" \
        --lr 3e-4 \
        --min-lr 3e-5 \
        --lr-decay-style WSD \
        --lr-decay-iters 1 \
        --lr-warmup-fraction 0.0 \
        --lr-wsd-decay-iters 1 \
        --seq-length 512 \
        --data-path 1.0 "$probe_prefix" \
        --split 100,0,0 \
        --train-iters 1 \
        --skip-train \
        --load "$load_dir" \
        --no-load-optim \
        --no-load-rng \
        --diagnostic-override-train-iteration 0 \
        --diagnostic-override-consumed-train-samples 0 \
        "${resume_args[@]}" \
        --eval-interval 1 \
        --probe-name "${domain}_fingerprint_smoke" \
        --probe-eval-iters "$PROBE_EVAL_ITERS" \
        --probe-eval-interval 1 \
        --probe-data-path 1.0 "$probe_prefix" \
        --run-initial-probe-eval \
        --hidden-space-dump-path "$tmp_npz" \
        --hidden-space-dump-label "$stage" \
        --hidden-space-dump-max-tokens "$MAX_TOKENS" \
        --hidden-space-dump-tokens-per-sample "$TOKENS_PER_SAMPLE" \
        --hidden-space-dump-layers "$LAYERS" \
        --hidden-space-dump-routing-details \
        --hidden-space-dump-expert-outputs \
        > "$log_path" 2>&1 &
    local launcher_pid=$!

    set +e
    watch_job "$launcher_pid" "$uuid" "$tmp_npz"
    local rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
        echo "[ERROR] dump failed stage=$stage domain=$domain rc=$rc log=$log_path" >&2
        return "$rc"
    fi
    if [[ ! -s "$tmp_npz" ]]; then
        echo "[ERROR] successful launcher produced no dump: $tmp_npz" >&2
        return 1
    fi
    mv "$tmp_npz" "$out_npz"
    echo "[DONE] $out_npz"
}

if (( SAMPLES <= 0 || TOKENS_PER_SAMPLE <= 0 || MICRO_BATCH_SIZE <= 0 )); then
    echo "[ERROR] SAMPLES, TOKENS_PER_SAMPLE, and MICRO_BATCH_SIZE must be positive" >&2
    exit 1
fi
if (( SAMPLES % MICRO_BATCH_SIZE != 0 )); then
    echo "[ERROR] SAMPLES must be divisible by MICRO_BATCH_SIZE" >&2
    exit 1
fi

checkpoint_step "$A_DIR" 1800 A
checkpoint_step "$B_DIR" 600 B
checkpoint_step "$C_DIR" 1800 C_vocabkl
checkpoint_step "$G_DIR" 600 G
checkpoint_step "$H_DIR" 1800 H
mkdir -p "$OUT_ROOT/dumps/wiki" "$OUT_ROOT/dumps/code" "$OUT_ROOT/logs"

port="$MASTER_PORT_BASE"
for domain in $DOMAINS; do
    for stage in $STAGES; do
        run_dump "$stage" "$domain" "$port"
        port=$((port + 1))
    done
done

echo "[DONE] fingerprint router smoke dumps: $OUT_ROOT/dumps"
