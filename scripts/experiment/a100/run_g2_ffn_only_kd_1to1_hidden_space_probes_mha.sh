#!/bin/bash
set -euo pipefail

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/../../.." && pwd)"
cd "$R"

source "$D/common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT_BASE="${MASTER_PORT_BASE:-29841}"
export G2_ROOT="${G2_ROOT:-$R/.local/weights/a100/mha/g2-checkpoints}"
export OUT_ROOT="${OUT_ROOT:-$R/.local/analysis/g2-ffn-only-kd-1to1-hidden-space}"
export PROBE_TASKS="${PROBE_TASKS:-wiki code}"
export STAGE1_LOG="${STAGE1_LOG:-$R/.local/logs/g2_kd_ramp900_code_kd_ramp900_conversation_chain/stage1_code_joint_ramp.log}"

export NUM_LAYERS="${NUM_LAYERS:-9}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-5472}"
export NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-16}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-352}"
export MOE_LAYER_FREQ="${MOE_LAYER_FREQ:-[0]*1+[1]*8}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-4}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-4}"
export HIDDEN_SPACE_MAX_TOKENS="${HIDDEN_SPACE_MAX_TOKENS:-2048}"
export HIDDEN_SPACE_LAYERS="${HIDDEN_SPACE_LAYERS:-all}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-$R/.local/models/pythia-12b-tokenizer}"
export MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-scripts/experiment/a100/flame-moe-bf16-no-shared.sh}"

WIKI_DIR="${WIKI_DIR:-$G2_ROOT/wiki/g2matched-top4-e8-ffn352-wiki-ffn-moe-mha-a100-bf16-mb128-1800}"
KD_INIT_DIR="${KD_INIT_DIR:-$G2_ROOT/code/expansion_distill_init/g2-ffn-only-e8to16-code-expert-init-logits-wiki-distill-mha-a100-bf16-mb48-1800}"
CODE_1TO1_DIR="${CODE_1TO1_DIR:-$G2_ROOT/code/joint_lm_replay_ramp/g2-ffn-only-code-wiki-joint-lm-allrouter-newexpert-ramp900-mb96-1800}"

require_checkpoint_step() {
    local path="$1"
    local expected="$2"
    local label="$3"
    local tracker="$path/latest_checkpointed_iteration.txt"
    local actual=""
    if [ -f "$tracker" ]; then
        actual="$(tr -d '\r\n[:space:]' < "$tracker")"
    fi
    if [ "$actual" != "$expected" ]; then
        echo "[ERROR] $label checkpoint expected=$expected actual=${actual:-missing}: $path" >&2
        exit 1
    fi
}

probe_prefix_for_task() {
    case "$1" in
        wiki) echo "${WIKI_PROBE_PREFIX:-$R/data/wiki/test/test_text_document}" ;;
        code) echo "${CODE_PROBE_PREFIX:-$R/data/code/test/test_text_document}" ;;
        *) echo "[ERROR] unsupported probe task: $1" >&2; return 1 ;;
    esac
}

run_dump() {
    local task="$1"
    local label="$2"
    local load_dir="$3"
    local num_experts="$4"
    local port="$5"
    local probe_prefix="$6"
    local task_out="$OUT_ROOT/$task"
    local out_npz="$task_out/hidden/${label}.npz"
    local out_log="$task_out/logs/${label}.log"
    local resume_args=()

    if [ -s "$out_npz" ] && [ "${FORCE_DUMP:-0}" != "1" ]; then
        echo "[SKIP] existing hidden dump: $out_npz"
        return
    fi
    if [ "$num_experts" -gt 8 ]; then
        resume_args=(--moe-resume-from-num-experts 8)
    fi

    export NUM_EXPERTS="$num_experts"
    # shellcheck source=/dev/null
    source "$MODEL_CONFIG_SCRIPT"

    echo "[RUN] task=$task label=$label experts=$num_experts port=$port"
    torchrun \
        --nproc_per_node "$NPROC_PER_NODE" \
        --master_addr "$MASTER_ADDR" \
        --master_port "$port" \
        Megatron-LM/pretrain_gpt.py \
        "${MODEL_ARGS[@]}" \
        --transformer-impl local \
        --pipeline-model-parallel-size 1 \
        --expert-model-parallel-size 1 \
        --distributed-timeout-minutes 30 \
        --no-persist-layer-norm \
        --bf16 \
        --micro-batch-size "$MICRO_BATCH_SIZE" \
        --global-batch-size "$GLOBAL_BATCH_SIZE" \
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
        --probe-name "${task}_probe" \
        --probe-eval-iters "$PROBE_EVAL_ITERS" \
        --probe-eval-interval 1 \
        --probe-data-path 1.0 "$probe_prefix" \
        --run-initial-probe-eval \
        --hidden-space-dump-path "$out_npz" \
        --hidden-space-dump-label "$label" \
        --hidden-space-dump-max-tokens "$HIDDEN_SPACE_MAX_TOKENS" \
        --hidden-space-dump-layers "$HIDDEN_SPACE_LAYERS" \
        > "$out_log" 2>&1
    echo "[DONE] $out_npz"
}

require_checkpoint_step "$WIKI_DIR" 1800 wiki_only
require_checkpoint_step "$KD_INIT_DIR" 1800 kd_init
require_checkpoint_step "$CODE_1TO1_DIR" 1800 code_wiki_1to1
if [ ! -f "$STAGE1_LOG" ]; then
    echo "[ERROR] missing Code phase-1 log: $STAGE1_LOG" >&2
    exit 1
fi

echo "[CONFIG] Wiki-only -> KD-init -> Code:Wiki 1:1 hidden-space analysis"
echo "[CONFIG] wiki_only=$WIKI_DIR"
echo "[CONFIG] kd_init=$KD_INIT_DIR"
echo "[CONFIG] code_wiki_1to1=$CODE_1TO1_DIR"
echo "[CONFIG] probe_tasks=$PROBE_TASKS out_root=$OUT_ROOT"
echo "[CONFIG] max_tokens=$HIDDEN_SPACE_MAX_TOKENS layers=$HIDDEN_SPACE_LAYERS"

if [ "${PLAN_ONLY:-0}" = "1" ]; then
    exit 0
fi

mkdir -p "$OUT_ROOT"
port="$MASTER_PORT_BASE"
for task in $PROBE_TASKS; do
    probe_prefix="$(probe_prefix_for_task "$task")"
    task_out="$OUT_ROOT/$task"
    mkdir -p "$task_out/hidden" "$task_out/logs" "$task_out/plots"

    run_dump "$task" wiki_only "$WIKI_DIR" 8 "$port" "$probe_prefix"
    port=$((port + 1))
    run_dump "$task" kd_init "$KD_INIT_DIR" 16 "$port" "$probe_prefix"
    port=$((port + 1))
    run_dump "$task" code_wiki_1to1 "$CODE_1TO1_DIR" 16 "$port" "$probe_prefix"
    port=$((port + 1))

    python scripts/analysis/plot_hidden_space_kd_1to1.py \
        --wiki-only "$task_out/hidden/wiki_only.npz" \
        --kd-init "$task_out/hidden/kd_init.npz" \
        --code-wiki-1to1 "$task_out/hidden/code_wiki_1to1.npz" \
        --out-dir "$task_out/plots" \
        --probe-task "$task" \
        --stage1-log "$STAGE1_LOG" \
        --code-phase-steps 1800 \
        --method "${HIDDEN_SPACE_PLOT_METHOD:-pca}" \
        --max-points-per-stage "${HIDDEN_SPACE_PLOT_MAX_POINTS_PER_STAGE:-1200}"
done

echo "[DONE] KD + 1:1 Wiki/Code hidden-space analysis"
echo "[OUTPUT] $OUT_ROOT"
