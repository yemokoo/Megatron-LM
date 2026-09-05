#!/usr/bin/env bash
set -euo pipefail

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/../.." && pwd)"
cd "$R"

PY_ENV="${PY_ENV:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100}"
TORCHRUN="${TORCHRUN:-$PY_ENV/bin/torchrun}"
GPU="${GPU:-5}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29931}"
OUT_ROOT="${OUT_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/fingerprint_router_geometry_20260809}"
SOURCE_ROOT="${SOURCE_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/g2_olddata_kd_9run_20260808}"
DATA_ROOT="${DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6}"
PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
CONDITIONS="${CONDITIONS:-baseline teacher_full stable_sensitive32_only stable_sensitive32_removed pca32_only random32_only rowspace16_only rowspace16_removed}"

C_DIR="$SOURCE_ROOT/02_nine_runs/local/weights/a100/mha/g2-checkpoints/code/joint_old_data_kd/post_kd_teacher_c10_fixed/r2-code-vocabkl-c10-fixed-post16-mb48-1800-probe3i100"
FINGERPRINT_ROOT="$OUT_ROOT/metrics/fingerprints"
LOG_ROOT="$OUT_ROOT/logs/forward_interventions"

export PATH="$PY_ENV/bin:${PATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$R/Megatron-LM:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/data2/seonghyeonnoh/homecache/huggingface}"
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1 WANDB_MODE=disabled

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
    local launcher_pid="$1" uuid="$2" marker="$3" status=0 foreign pid pid_pgid cmd
    while kill -0 "$launcher_pid" 2>/dev/null; do
        foreign=""
        while read -r pid; do
            [[ -n "$pid" ]] || continue
            pid_pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
            [[ "$pid_pgid" == "$launcher_pid" ]] && continue
            cmd="$(ps -ww -o args= -p "$pid" 2>/dev/null || true)"
            # Ignore the short nvidia-smi stale-PID window after our CUDA
            # worker exits; a real foreign process still has a /proc command.
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
    [[ "$status" -eq 0 ]] || return "$status"
    return "$rc"
}

condition_args() {
    case "$1" in
        baseline) ;;
        teacher_full)
            printf '%s\n' --router-fingerprint-intervention-path "$FINGERPRINT_ROOT/router_row_space_r16_effective16.npz" --router-fingerprint-intervention-mode teacher_full
            ;;
        stable_sensitive32_only)
            printf '%s\n' --router-fingerprint-intervention-path "$FINGERPRINT_ROOT/stable_routing_sensitive_r32_effective32.npz" --router-fingerprint-intervention-mode fingerprint_only
            ;;
        stable_sensitive32_removed)
            printf '%s\n' --router-fingerprint-intervention-path "$FINGERPRINT_ROOT/stable_routing_sensitive_r32_effective32.npz" --router-fingerprint-intervention-mode fingerprint_removed
            ;;
        pca32_only)
            printf '%s\n' --router-fingerprint-intervention-path "$FINGERPRINT_ROOT/top_variance_pca_r32_effective32.npz" --router-fingerprint-intervention-mode fingerprint_only
            ;;
        random32_only)
            printf '%s\n' --router-fingerprint-intervention-path "$FINGERPRINT_ROOT/random_r32_repeat0.npz" --router-fingerprint-intervention-mode fingerprint_only
            ;;
        rowspace16_only)
            printf '%s\n' --router-fingerprint-intervention-path "$FINGERPRINT_ROOT/router_row_space_r16_effective16.npz" --router-fingerprint-intervention-mode fingerprint_only
            ;;
        rowspace16_removed)
            printf '%s\n' --router-fingerprint-intervention-path "$FINGERPRINT_ROOT/router_row_space_r16_effective16.npz" --router-fingerprint-intervention-mode fingerprint_removed
            ;;
        *) echo "[ERROR] unsupported condition: $1" >&2; return 2 ;;
    esac
}

run_condition() {
    local condition="$1" port="$2" uuid log marker
    local intervention_args=()
    uuid="$(gpu_uuid)"
    log="$LOG_ROOT/${condition}.log"
    marker="fingerprint_forward_${condition}"
    mapfile -t intervention_args < <(condition_args "$condition")

    if grep -q 'probe code_fingerprint_forward at iteration' "$log" 2>/dev/null \
        && grep -q 'probe wiki_fingerprint_forward at iteration' "$log" 2>/dev/null \
        && [[ "${FORCE_INTERVENTION:-0}" != 1 ]]; then
        echo "[SKIP] completed condition=$condition log=$log"
        return
    fi
    assert_gpu_free "$uuid"
    echo "[RUN] condition=$condition physical_gpu=$GPU uuid=$uuid"

    setsid env CUDA_VISIBLE_DEVICES="$GPU" "$TORCHRUN" \
        --nproc_per_node 1 --master_addr 127.0.0.1 --master_port "$port" \
        Megatron-LM/pretrain_gpt.py \
        --hidden-size 1024 --ffn-hidden-size 5472 --num-layers 9 \
        --num-attention-heads 16 --group-query-attention --num-query-groups 16 \
        --swiglu --max-position-embeddings 2048 --normalization RMSNorm --norm-epsilon 1e-6 \
        --untie-embeddings-and-output-weights --position-embedding-type rope --disable-bias-linear \
        --moe-ffn-hidden-size 352 --num-experts 16 --moe-router-topk 4 \
        --moe-layer-freq '[0]*1+[1]*8' --moe-router-dtype fp32 \
        --moe-router-pre-softmax --moe-router-score-function softmax \
        --moe-aux-loss-coeff 0.01 --moe-z-loss-coeff 0.001 \
        --hidden-dropout 0.0 --attention-dropout 0.0 --init-method-std 0.02 \
        --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$TOKENIZER_MODEL" \
        --transformer-impl local --pipeline-model-parallel-size 1 --expert-model-parallel-size 1 \
        --distributed-timeout-minutes 30 --no-persist-layer-norm --bf16 \
        --micro-batch-size "$MICRO_BATCH_SIZE" --global-batch-size "$MICRO_BATCH_SIZE" \
        --lr 3e-4 --min-lr 3e-5 --lr-decay-style WSD --lr-decay-iters 1 \
        --lr-warmup-fraction 0.0 --lr-wsd-decay-iters 1 --seq-length 512 \
        --data-path 1.0 "$DATA_ROOT/wiki/test/test_text_document" --split 100,0,0 \
        --train-iters 1 --skip-train --load "$C_DIR" --no-load-optim --no-load-rng \
        --diagnostic-override-train-iteration 0 --diagnostic-override-consumed-train-samples 0 \
        --moe-resume-from-num-experts 8 --eval-interval 1 --run-initial-probe-eval \
        --probe-name wiki_fingerprint_forward --probe-eval-iters "$PROBE_EVAL_ITERS" \
        --probe-eval-interval 1 --probe-data-path 1.0 "$DATA_ROOT/wiki/test/test_text_document" \
        --secondary-probe-name code_fingerprint_forward \
        --secondary-probe-eval-iters "$PROBE_EVAL_ITERS" --secondary-probe-eval-interval 1 \
        --secondary-probe-data-path 1.0 "$DATA_ROOT/code/test/test_text_document" \
        "${intervention_args[@]}" \
        --hidden-space-dump-label "$marker" \
        > "$log" 2>&1 &
    local launcher_pid=$!
    set +e
    watch_job "$launcher_pid" "$uuid" "$marker"
    local rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
        echo "[ERROR] intervention failed condition=$condition rc=$rc log=$log" >&2
        return "$rc"
    fi
    grep 'probe \(wiki\|code\)_fingerprint_forward at iteration' "$log"
}

[[ -f "$C_DIR/latest_checkpointed_iteration.txt" ]] || { echo "[ERROR] missing C checkpoint: $C_DIR" >&2; exit 1; }
[[ "$(tr -d '[:space:]' < "$C_DIR/latest_checkpointed_iteration.txt")" == 1800 ]] || { echo "[ERROR] C checkpoint is not step 1800" >&2; exit 1; }
mkdir -p "$LOG_ROOT"

port="$MASTER_PORT_BASE"
for condition in $CONDITIONS; do
    run_condition "$condition" "$port"
    port=$((port + 1))
done
echo "[DONE] actual-forward fingerprint interventions: $LOG_ROOT"
