#!/bin/bash
# Evaluate every stage checkpoint under one identical protocol and print the
# lower-triangular accuracy matrix inputs for AA / FM.
#
# The training-time probes cannot be compared across methods: each probe used
# that method's own micro-batch, so the evaluated sample count ranged from
# 1,200 to 3,600 sequences.  Here every method is scored with the same
# micro-batch and the same number of probe iterations.
#
# Each checkpoint is exposed through a scratch directory of symlinks so the
# loader takes the resume path.  Loading via --finetune would reset iteration
# to 0, and the O-LoRA adapter setup re-initialises the current slot at
# iteration 0 -- that would wipe the very slot being evaluated.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BASELINES6_EVAL_ONLY=1
export PROBE_ITERS="${EVAL_PROBE_ITERS:-100}"
# run_probe_evaluation only fires when iteration % interval == 0, and the
# checkpoints sit at iteration 1800; an interval of 1 always divides it.  With
# --skip-train there is no training loop, so nothing else can trigger it.
export PROBE_INTERVAL=1
EVAL_MB="${EVAL_MICRO_BATCH:-48}"
source "$DIR/train_stage.sh"

ROOT="$BASELINES6_OUTPUT_ROOT"
OUT="${EVAL_OUTPUT_ROOT:-$ROOT/eval_matrix_mb${EVAL_MB}}"
mkdir -p "$OUT"

stage() {   # name method task real_stage_dir port [method args...]
    local name="$1" method="$2" task="$3" real="$4" port="$5"; shift 5
    local work="$OUT/work/$name"
    if [ ! -f "$real/latest_checkpointed_iteration.txt" ]; then
        echo "[MISS] $name ($real)"; return 0
    fi
    if [ -s "$OUT/$name.probe" ]; then
        echo "[HAVE] $name"; return 0
    fi
    rm -rf "$work"; mkdir -p "$work"
    cp "$real/latest_checkpointed_iteration.txt" "$work/"
    local it; it="$(tr -d '[:space:]' < "$real/latest_checkpointed_iteration.txt")"
    ln -sfn "$real/$(printf 'iter_%07d' "$it")" "$work/$(printf 'iter_%07d' "$it")"
    echo "=== $name ($method/$task, iter $it) ==="
    run_baselines6_stage "$method" "$task" "$work" "" "" "$EVAL_MB" "$port" "$@" || true
    { grep '^probe ' "$work/logs/train.log" 2>/dev/null | tail -3 > "$OUT/$name.probe"; } || true
    sed 's/^/    /' "$OUT/$name.probe"
}

# SLoRA merges its adapter into the base weights, so those checkpoints are
# plain dense models; evaluating them with the dense spec is both correct and
# simpler. O-LoRA keeps its slots unmerged and Fixed MoE has a different FFN,
# so those two need their own architecture.
OL=(--continual-olora-rank 352 --continual-olora-alpha 352 --continual-olora-dropout 0.1 --continual-olora-orth-lambda 0.5)
D=sequential_dense

stage common_wiki   $D       wiki         "$ROOT/common_dense/wiki"              29901
stage ewc_code      $D       code         "$ROOT/ewc/code"                       29902
stage ewc_conv      $D       conversation "$ROOT/ewc/conversation"                29903
stage trace_code    $D       code         "$ROOT/trace_gem/code"                 29904
stage trace_conv    $D       conversation "$ROOT/trace_gem/conversation"          29905
stage seq_code      $D       code         "$ROOT/sequential_dense/code"          29906
stage seq_conv      $D       conversation "$ROOT/sequential_dense/conversation"   29907
stage slora64_code  $D       code         "$ROOT/slora_pre/rank64/code"          29908
stage slora64_conv  $D       conversation "$ROOT/slora_pre/rank64/conversation"   29909
stage slora16_code  $D       code         "$ROOT/slora_pre/rank16/code"          29910
stage slora16_conv  $D       conversation "$ROOT/slora_pre/rank16/conversation"   29911
stage slora32_code  $D       code         "$ROOT/slora_pre/rank32/code"          29912
stage olora_wiki    olora    wiki         "$ROOT/olora/wiki"                     29913 "${OL[@]}"
stage olora_code    olora    code         "$ROOT/olora/code"                     29914 "${OL[@]}"
stage olora_conv    olora    conversation "$ROOT/olora/conversation"              29915 "${OL[@]}"
stage moe_wiki      fixed_moe wiki        "$ROOT/fixed_moe/wiki"                 29916
stage moe_code      fixed_moe code        "$ROOT/fixed_moe/code"                 29917
stage moe_conv      fixed_moe conversation "$ROOT/fixed_moe/conversation"         29918

echo; echo "=== 결과 ($OUT) ==="
for f in "$OUT"/*.probe; do [ -s "$f" ] && { echo "## $(basename "$f" .probe)"; sed 's/^/   /' "$f"; }; done
