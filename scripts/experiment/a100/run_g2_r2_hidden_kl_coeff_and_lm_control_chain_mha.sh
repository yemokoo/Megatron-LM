#!/bin/bash
set -euo pipefail

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/../../.." && pwd)"

export LOCAL_BASE="${LOCAL_BASE:-/data3/seonghyeonnoh/LLM-continual-learning-runs/local}"
export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/data3/seonghyeonnoh/LLM-continual-learning-staging/flame-moe}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-/data3/seonghyeonnoh/LLM-continual-learning-models/pythia-12b-tokenizer}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export WANDB_MODE="${WANDB_MODE:-offline}"

STEPS=300
MB=36
GBS=2304
G2_ROOT="$LOCAL_BASE/weights/a100/mha/g2-checkpoints"
WIKI_SOURCE="/data3/seonghyeonnoh/LLM-continual-learning-runs/wiki/g2-wiki-e8-ffn352-top4-h100-mb72-1800"
R1_SOURCE="$G2_ROOT/code/expansion_distill_init/r1-code-expand-wiki-kd-e8to16-mb36-600"
DIAG_ROOT="$G2_ROOT/code/joint_old_data_hidden_kl/normalization_fix_diagnostic"
C1_ID="r2-diagnostic-hiddenkl-normfix-v3-c1.0-mb36-300"
C1_OUT="$DIAG_ROOT/$C1_ID"
LM_ID="r2-control-wiki-lm-replay-mb36-300"
LM_OUT="$G2_ROOT/code/joint_lm_replay/hidden_kl_coeff_control/$LM_ID"
CHAIN_LOG_DIR="$LOCAL_BASE/logs/r2_hidden_kl_coeff_and_lm_control"

checkpoint_at() {
    [ -f "$1/latest_checkpointed_iteration.txt" ] &&
        [ "$(tr -d '[:space:]' < "$1/latest_checkpointed_iteration.txt")" = "$2" ]
}

hidden_out() {
    local coeff="$1"
    echo "$DIAG_ROOT/r2-diagnostic-hiddenkl-normfix-c${coeff}-mb36-300"
}

print_plan() {
    echo "[COMMON] source=$R1_SOURCE teacher=$WIKI_SOURCE steps=$STEPS MB=$MB GBS=$GBS probes=100"
    echo "[1] hidden KL coefficient 1: $C1_OUT (reuse running/completed v3)"
    for coeff in 10 100 500; do
        echo "[$coeff] hidden KL coefficient $coeff: $(hidden_out "$coeff")"
    done
    echo "[LM] Wiki LM replay control: $LM_OUT"
}

print_plan
[ "${PLAN_ONLY:-0}" != "1" ] || exit 0
mkdir -p "$CHAIN_LOG_DIR"

echo "[WAIT] coefficient 1 run final checkpoint"
while ! checkpoint_at "$C1_OUT" "$STEPS"; do
    if ! tmux has-session -t r2_hiddenkl_diag 2>/dev/null; then
        echo "[ERROR] coefficient 1 tmux ended without step-$STEPS checkpoint: $C1_OUT" >&2
        exit 1
    fi
    sleep 30
done
echo "[DONE] coefficient 1"

port=29883
for coeff in 10 100 500; do
    out="$(hidden_out "$coeff")"
    if checkpoint_at "$out" "$STEPS"; then
        echo "[SKIP] coefficient $coeff complete: $out"
        port=$((port + 1))
        continue
    fi
    echo "[START] hidden KL coefficient $coeff -> $out"
    OLD_HIDDEN_KL_COEFF="$coeff" \
    RUN_ID="$(basename "$out")" \
    TRAIN_ITERS="$STEPS" MICRO_BATCH_SIZE="$MB" \
    PROBE_EVAL_INTERVAL=100 SECONDARY_PROBE_EVAL_INTERVAL=100 \
    MASTER_PORT="$port" \
    bash "$D/run_g2_r2_hidden_kl_normalization_fix_diagnostic_mha.sh" \
        2>&1 | tee "$CHAIN_LOG_DIR/hidden_kl_c${coeff}.log"
    checkpoint_at "$out" "$STEPS" || {
        echo "[ERROR] coefficient $coeff missing final checkpoint: $out" >&2
        exit 1
    }
    echo "[DONE] coefficient $coeff"
    port=$((port + 1))
done

if checkpoint_at "$LM_OUT" "$STEPS"; then
    echo "[SKIP] Wiki LM control complete: $LM_OUT"
else
    echo "[START] Wiki LM replay control -> $LM_OUT"
    SOURCE_WEIGHTS_DIR="$R1_SOURCE" SOURCE_REQUIRED_ITERS=600 \
    TRAIN_ITERS="$STEPS" MICRO_BATCH_SIZE="$MB" GLOBAL_BATCH_SIZE="$GBS" \
    SAVE_INTERVAL="$STEPS" EVAL_INTERVAL=100 LOG_INTERVAL=10 \
    PROBE_EVAL_INTERVAL=100 SECONDARY_PROBE_EVAL_INTERVAL=100 \
    MOE_NEW_EXPERT_LR_RAMP_STEPS=900 \
    RUN_ID="$LM_ID" TRAIN_WEIGHTS="$LM_OUT" MASTER_PORT=29886 \
    bash "$D/run_g2_ffn_only_code_wiki_joint_lm_allrouter_mha.sh" \
        2>&1 | tee "$CHAIN_LOG_DIR/wiki_lm_control.log"
    checkpoint_at "$LM_OUT" "$STEPS" || {
        echo "[ERROR] Wiki LM control missing final checkpoint: $LM_OUT" >&2
        exit 1
    }
    echo "[DONE] Wiki LM replay control"
fi

echo "[ALL DONE] coefficient 1/10/100/500 + Wiki LM control"
