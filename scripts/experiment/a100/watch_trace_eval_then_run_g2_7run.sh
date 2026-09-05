#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning"
TRACE_ROOT="$ROOT/trace/results/full_runs/llama31"
V2_SUMMARY="$TRACE_ROOT/ours_lora_moe_v2_strict_1phase_ratio5/sparse15_summary.json"
V3_SUMMARY="$TRACE_ROOT/ours_lora_moe_v3_strict_1phase/sparse15_summary.json"
TARGET="scripts/experiment/a100/run_g2_7run_hidden_kl_c100to10_s600_postfirst_h100_chain_mha.sh"
LOG_ROOT="/data3/seonghyeonnoh/LLM-continual-learning-runs/local/logs"
TARGET_LOG="$LOG_ROOT/g2_7run_hidden_kl_c100to10_s600_launcher.log"
SENTINEL="$LOG_ROOT/g2_7run_hidden_kl_c100to10_s600_launched.sentinel"
POLL_SECONDS="${POLL_SECONDS:-60}"

timestamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }

summaries_valid() {
    [ -f "$V2_SUMMARY" ] && [ -f "$V3_SUMMARY" ] || return 1
    python3 - "$V2_SUMMARY" "$V3_SUMMARY" <<'PY'
import json
import sys
for path in sys.argv[1:]:
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

trace_processes_running() {
    pgrep -f '[t]raining/main_Ours_LoRA_MoE.py' >/dev/null ||
    pgrep -f '[e]valuate_Ours_LoRA_MoE.py' >/dev/null ||
    pgrep -f '[r]un_ours_sparse15_parallel.sh' >/dev/null
}

gpus_busy() {
    local compute
    compute="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
    [ -n "$compute" ]
}

mkdir -p "$LOG_ROOT"
echo "[$(timestamp)] watcher started"
echo "[$(timestamp)] waiting for $V2_SUMMARY"
echo "[$(timestamp)] waiting for $V3_SUMMARY"

while ! summaries_valid; do
    sleep "$POLL_SECONDS"
done
echo "[$(timestamp)] both sparse-15 summaries validated"

while trace_processes_running || gpus_busy; do
    echo "[$(timestamp)] summaries ready; waiting for TRACE processes and GPUs to become idle"
    sleep "$POLL_SECONDS"
done

if pgrep -f '[r]un_g2_7run_hidden_kl_c100to10_s600_postfirst_h100_chain_mha.sh' >/dev/null; then
    echo "[$(timestamp)] G2 target already running; watcher exits without launching a duplicate"
    exit 0
fi

if ! (set -o noclobber; printf '%s\n' "$(timestamp)" > "$SENTINEL") 2>/dev/null; then
    echo "[$(timestamp)] launch sentinel already exists: $SENTINEL"
    exit 0
fi

cd "$ROOT"
echo "[$(timestamp)] launching G2 chain in-place; log=$TARGET_LOG"
exec bash "$TARGET" >> "$TARGET_LOG" 2>&1
