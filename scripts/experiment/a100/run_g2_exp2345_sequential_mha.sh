#!/bin/bash
# Run G2 experiments 2 -> 3 -> 4 -> 5 back-to-back (one training job at a time).
# See EXPERIMENT_PLAN_G2_5EXP.md. Exp 1 (dense-active sequential) is already done.
#
#   exp2  fixed24 MoE sequential  : wiki -> code -> conversation, from scratch
#   exp3  expand-MoE attn-unfreeze: conversation only, continues from the existing
#                                   wiki->code router-retuned code checkpoint (iter 3600)
#   exp4  dense(active) mixed     : single 5400-step joint 1:1:1 upper bound
#   exp5  fixed24 MoE mixed       : single 5400-step joint 1:1:1 upper bound
#   exp6  dense(active) cumul-mix : 3-stage replay chain wiki -> wiki+code ->
#                                   wiki+code+conv, 1800/3600/5400 (=10800), CL ceiling
#   exp7  fixed24 MoE  cumul-mix  : same 3-stage replay chain (=10800), CL ceiling
#
# --- micro-batch policy (per request) ---
# KL/KD stages get their micro-batch dropped one notch (they carry a teacher model,
# so they are the OOM-prone ones), clamped to the 32..48 floor:
#   * exp2 code + conversation (OLD_MODEL_KL_COEFF=1.0) : 48 -> 36
#     (40 is invalid: global_batch 2304 must be divisible by micro*dp; 36 and 32
#      are the valid steps below 48. 36 keeps us safely inside the 32..48 floor.)
#   * exp2 wiki (no teacher, not a KL stage)            : stays at 48
#   * exp3 conversation                                 : 96 -> 72
#     (exp3 actually runs with KL disabled, but attention-unfreeze is memory-heavy
#      and prior phase4 conversation runs used mb48, so we drop the mb96 default a
#      notch per request. 80 is invalid; 72 is the valid step below 96.)
#   * exp4 mixed (no KD)                                : stays at 72
#   * exp5 mixed (no KD)                                : stays at 48
#   * exp6/exp7 cumulative-mixed (pure replay, no KD)   : stay at 96 / 72
#
# Env prerequisites: grouped_gemm must be importable for the 24-expert MoE runs
# (exp2/3/5/7). exp4 & exp6 (dense, num_experts=1) set MOE_GROUPED_GEMM=0 and do
# not need it.
#
# Overridable env:
#   EXPERIMENTS="2 3 4 5 6 7"   which experiments to run, in order
#   STOP_ON_ERROR=0         1 = abort the whole batch if one experiment fails
#   CUDA_VISIBLE_DEVICES / NPROC_PER_NODE   GPU fan-out (default 4 GPUs, like exp1)
#   WANDB_MODE=offline      offline|online|disabled
#   PAUSE_SECONDS=180       cool-down between stages/experiments

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

export EXPERIMENTS="${EXPERIMENTS:-2 3 4 5 6 7}"
export STOP_ON_ERROR="${STOP_ON_ERROR:-0}"

# Match exp1's fan-out (it ran on 4 GPUs). All chosen micro-batches are valid for
# both dp=2 and dp=4 against global_batch 2304, so overriding to "0,1"/NPROC=2 is
# also safe if you want to leave GPUs free.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export PAUSE_SECONDS="${PAUSE_SECONDS:-180}"

# The pythia-12b tokenizer is already cached under ~/.cache/huggingface; force
# offline so a run never blocks/fails trying to reach huggingface.co when the box
# has no outbound network (verified failure mode). Override to 0 if you truly need
# to fetch something.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

BATCH_LOG_DIR="${BATCH_LOG_DIR:-$PROJECT_ROOT/logs}"
mkdir -p "$BATCH_LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
BATCH_LOG="$BATCH_LOG_DIR/g2_exp2345_sequential_${STAMP}.log"
# Mirror everything to a batch-level log as well as the console.
exec > >(tee -a "$BATCH_LOG") 2>&1

echo "=========================================================================="
echo "[BATCH] G2 exp 2->3->4->5->6->7   $(date)"
echo "[BATCH] experiments=$EXPERIMENTS  stop_on_error=$STOP_ON_ERROR"
echo "[BATCH] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES nproc=$NPROC_PER_NODE wandb=$WANDB_MODE"
echo "[BATCH] log=$BATCH_LOG"
echo "=========================================================================="

# exp3's wrapper `exec`s the phase4 chain and run_guarded_training.sh directly
# (not via `bash`), so they must be executable. Ensure it defensively.
chmod +x "$SCRIPT_DIR/run_g2_phase4_conversation_after_router_finetune_offline_chain_mha.sh" \
         "$SCRIPT_DIR/run_guarded_training.sh" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Preflight: grouped_gemm (needed by exp2/3/5) + exp3 source checkpoints.
# ---------------------------------------------------------------------------
GEMM_OK=0
# Run the env-activation + import inside a subshell: activate_kt_env.sh reassigns
# SCRIPT_DIR to its own dir (scripts/miscellaneous), which would break the
# "$SCRIPT_DIR/run_g2_*" wrapper lookups below if it leaked into this shell.
if ( source "$PROJECT_ROOT/scripts/miscellaneous/activate_kt_env.sh" >/dev/null 2>&1 \
     && python -c "import grouped_gemm" >/dev/null 2>&1 ); then
    GEMM_OK=1
    echo "[PREFLIGHT] grouped_gemm import OK"
else
    echo "[PREFLIGHT][WARN] grouped_gemm NOT importable yet."
    echo "                 exp2/exp3/exp5 (24-expert MoE) will fail without it;"
    echo "                 exp4 (dense) does not need it. Install then rerun the"
    echo "                 failed experiments (completed stages are skipped)."
fi

G2_ROOT="$PROJECT_ROOT/.local/weights/a100/mha/g2-checkpoints"
EXP3_FFN_SOURCE="$G2_ROOT/code/phase3/g2matched-attn-freeze-phase3-router-only-retune-wikicode-no-reinit-mb96-1800"
for src in \
    "$EXP3_FFN_SOURCE" \
    "$G2_ROOT/code/phase3/g2-exp2-phase3-router-only-retune-wikicode-from-new-experts-all-router-no-reinit-mb72-1800" \
    "$G2_ROOT/code/phase3/g2-exp1-phase3-router-only-retune-wikicode-from-all-experts-router-no-reinit-mb72-1800"; do
    t="$src/latest_checkpointed_iteration.txt"
    if [ -f "$t" ] && [ "$(tr -d '[:space:]' < "$t")" = "3600" ]; then
        echo "[PREFLIGHT] exp3 source OK (iter 3600): $(basename "$src")"
    else
        echo "[PREFLIGHT][WARN] exp3 source missing/incomplete: $src"
    fi
done
echo ""

# ---------------------------------------------------------------------------
# Per-experiment runner: never let one failure kill the batch (unless asked).
# ---------------------------------------------------------------------------
declare -a RESULTS=()
run_experiment() {
    local tag="$1"; shift
    echo "--------------------------------------------------------------------------"
    echo "[EXP $tag] START $(date)"
    echo "--------------------------------------------------------------------------"
    local start end rc
    start="$(date +%s)"
    "$@"
    rc=$?
    end="$(date +%s)"
    local mins=$(( (end - start) / 60 ))
    if [ "$rc" -eq 0 ]; then
        echo "[EXP $tag] DONE ok (${mins} min) $(date)"
        RESULTS+=("exp$tag: OK (${mins}m)")
    else
        echo "[EXP $tag] FAILED rc=$rc (${mins} min) $(date)"
        RESULTS+=("exp$tag: FAILED rc=$rc (${mins}m)")
        if [ "$STOP_ON_ERROR" = "1" ]; then
            echo "[BATCH] STOP_ON_ERROR=1 -> aborting remaining experiments."
            print_summary
            exit "$rc"
        fi
    fi
    echo "[BATCH] cool-down ${PAUSE_SECONDS}s"; sleep "$PAUSE_SECONDS"
}

# --- exp2: fixed24 sequential wiki->code->conv (wiki mb48, code/conv KL mb36) ---
exp2() {
    env \
        WANDB_MODE="$WANDB_MODE" \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        NPROC_PER_NODE="$NPROC_PER_NODE" \
        PAUSE_SECONDS="$PAUSE_SECONDS" \
        MICRO_BATCH_SIZE=36 \
        WIKI_MICRO_BATCH_SIZE=48 \
        OLD_MODEL_KL_COEFF="${EXP2_OLD_MODEL_KL_COEFF:-1.0}" \
        bash "$SCRIPT_DIR/run_g2_fixed24_wiki_code_conversation_mha.sh"
}

# --- exp3: expand-MoE attn-unfreeze conversation (mb 96->72), from code 3600 ---
exp3() {
    env \
        WANDB_MODE="$WANDB_MODE" \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        NPROC_PER_NODE="$NPROC_PER_NODE" \
        PAUSE_SECONDS="$PAUSE_SECONDS" \
        FFN_MICRO_BATCH_SIZE=72 \
        bash "$SCRIPT_DIR/run_g2_exp3_ffn_only_attn_unfreeze_conversation_mha.sh"
}

# --- exp4: dense(active) mixed 5400-step upper bound (mb 72, no KD) ---
exp4() {
    env \
        WANDB_MODE="$WANDB_MODE" \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        NPROC_PER_NODE="$NPROC_PER_NODE" \
        MICRO_BATCH_SIZE=72 \
        bash "$SCRIPT_DIR/run_g2_exp4_dense_active_mixed_wiki_code_conv_mha.sh"
}

# --- exp5: fixed24 mixed 5400-step upper bound (mb 48, no KD) ---
exp5() {
    env \
        WANDB_MODE="$WANDB_MODE" \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        NPROC_PER_NODE="$NPROC_PER_NODE" \
        MICRO_BATCH_SIZE=48 \
        bash "$SCRIPT_DIR/run_g2_exp5_fixed24_mixed_wiki_code_conv_mha.sh"
}

# --- exp6: dense(active) cumulative-mixed replay chain (1800/3600/5400, mb96) ---
exp6() {
    env \
        WANDB_MODE="$WANDB_MODE" \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        NPROC_PER_NODE="$NPROC_PER_NODE" \
        PAUSE_SECONDS="$PAUSE_SECONDS" \
        bash "$SCRIPT_DIR/run_g2_exp6_dense_active_cumulative_mixed_wiki_code_conv_mha.sh"
}

# --- exp7: fixed24 MoE cumulative-mixed replay chain (1800/3600/5400, mb72) ---
exp7() {
    env \
        WANDB_MODE="$WANDB_MODE" \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        NPROC_PER_NODE="$NPROC_PER_NODE" \
        PAUSE_SECONDS="$PAUSE_SECONDS" \
        bash "$SCRIPT_DIR/run_g2_exp7_fixed24_cumulative_mixed_wiki_code_conv_mha.sh"
}

print_summary() {
    echo ""
    echo "=========================================================================="
    echo "[BATCH] SUMMARY $(date)"
    for r in "${RESULTS[@]}"; do echo "   - $r"; done
    echo "=========================================================================="
}

# Defensive: make sure nothing above left SCRIPT_DIR pointing elsewhere before we
# dispatch the "$SCRIPT_DIR/run_g2_*" wrappers.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for e in $EXPERIMENTS; do
    case "$e" in
        2) run_experiment 2 exp2 ;;
        3) run_experiment 3 exp3 ;;
        4) run_experiment 4 exp4 ;;
        5) run_experiment 5 exp5 ;;
        6) run_experiment 6 exp6 ;;
        7) run_experiment 7 exp7 ;;
        *) echo "[BATCH][WARN] unknown experiment '$e' (expected 2..7), skipping" ;;
    esac
done

print_summary
