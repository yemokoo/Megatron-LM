#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning"
TRACE_ROOT="$ROOT/trace"
RUN_ROOT="$TRACE_ROOT/results/full_runs/llama31"
V2_RUN="$RUN_ROOT/ours_lora_moe_v2_strict_1phase_ratio5"
V3_RUN="$RUN_ROOT/ours_lora_moe_v3_strict_1phase"
V2_RESUME="$V2_RUN/6"
V3_RESUME="$V3_RUN/4"
LOG_ROOT="/data3/seonghyeonnoh/LLM-continual-learning-runs/local/logs"
G2_TARGET="$ROOT/scripts/experiment/a100/run_g2_7run_hidden_kl_c100to10_s600_postfirst_h100_chain_mha.sh"
G2_LOG="$LOG_ROOT/g2_7run_hidden_kl_c100to10_s600_launcher.log"
G2_SENTINEL="$LOG_ROOT/g2_7run_hidden_kl_c100to10_s600_launched.sentinel"

timestamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }

require_checkpoint() {
    local checkpoint="$1"
    test -s "$checkpoint/pytorch_model.bin"
    test -s "$checkpoint/lora_moe_meta.json"
}

run_v3() {
    export PYTHONNOUSERSITE=1
    export WANDB_MODE=offline
    export OURS_LORAMOE_OUTPUT_ROOT="$V3_RUN"
    export OURS_LORAMOE_GPUS="0,1,2,3"
    export OURS_LORAMOE_MICRO_BATCH=16
    export OURS_LORAMOE_GRAD_ACCUM=1
    export OURS_LORAMOE_PORT=29641
    export OURS_LORAMOE_RESUME_CHECKPOINT="$V3_RESUME"
    export OURS_V2_KD_MEMORY_BATCH_SIZE=8
    export OURS_V2_JOINT_NEW_TO_REPLAY_RATIO=5
    "$TRACE_ROOT/scripts/baselines/_run_ours_lora_moe.sh" train llama31 v3
    "$TRACE_ROOT/scripts/baselines/run_ours_sparse15_parallel.sh" \
        llama31 v3 "$V3_RUN" "0,1,2,3"
}

run_v2() {
    export PYTHONNOUSERSITE=1
    export WANDB_MODE=offline
    export OURS_LORAMOE_OUTPUT_ROOT="$V2_RUN"
    export OURS_LORAMOE_GPUS="4,5,6,7"
    export OURS_LORAMOE_MICRO_BATCH=16
    export OURS_LORAMOE_GRAD_ACCUM=1
    export OURS_LORAMOE_PORT=29642
    export OURS_LORAMOE_RESUME_CHECKPOINT="$V2_RESUME"
    export OURS_V2_KD_MEMORY_BATCH_SIZE=8
    export OURS_V2_JOINT_NEW_TO_REPLAY_RATIO=5
    "$TRACE_ROOT/scripts/baselines/_run_ours_lora_moe.sh" train llama31 v2
    "$TRACE_ROOT/scripts/baselines/run_ours_sparse15_parallel.sh" \
        llama31 v2 "$V2_RUN" "4,5,6,7"
}

validate_summary() {
    python3 - "$1" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
if payload.get("evaluation_mode") != "sparse_15":
    raise SystemExit(f"not sparse_15: {path}")
if len(payload.get("diagonal_scores_rounds_1_to_7", [])) != 7:
    raise SystemExit(f"missing diagonal scores: {path}")
if len(payload.get("final_scores_round_8", [])) != 8:
    raise SystemExit(f"missing final scores: {path}")
PY
}

gpus_busy() {
    local compute
    compute="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        2>/dev/null | sed '/^[[:space:]]*$/d')"
    test -n "$compute"
}

mkdir -p "$LOG_ROOT"
require_checkpoint "$V2_RESUME"
require_checkpoint "$V3_RESUME"
echo "[$(timestamp)] recovery chain started"

run_v3 > "$V3_RUN/resume_from_4.launcher.log" 2>&1 &
v3_pid=$!
run_v2 > "$V2_RUN/resume_from_6.launcher.log" 2>&1 &
v2_pid=$!

failed=0
wait "$v3_pid" || failed=1
wait "$v2_pid" || failed=1
if test "$failed" -ne 0; then
    echo "[$(timestamp)] TRACE recovery/evaluation failed; G2 will not launch" >&2
    exit 1
fi

validate_summary "$V2_RUN/sparse15_summary.json"
validate_summary "$V3_RUN/sparse15_summary.json"
echo "[$(timestamp)] both sparse-15 summaries validated"

while gpus_busy; do
    echo "[$(timestamp)] waiting for all GPUs to become idle"
    sleep 60
done

if pgrep -f '[r]un_g2_7run_hidden_kl_c100to10_s600_postfirst_h100_chain_mha.sh' >/dev/null; then
    echo "[$(timestamp)] G2 chain is already running; exiting"
    exit 0
fi
if ! (set -o noclobber; printf '%s\n' "$(timestamp)" > "$G2_SENTINEL") 2>/dev/null; then
    echo "[$(timestamp)] G2 launch sentinel already exists: $G2_SENTINEL" >&2
    exit 1
fi

cd "$ROOT"
echo "[$(timestamp)] launching G2 R2-R7 chain on all 8 GPUs"
exec bash "$G2_TARGET" >> "$G2_LOG" 2>&1
