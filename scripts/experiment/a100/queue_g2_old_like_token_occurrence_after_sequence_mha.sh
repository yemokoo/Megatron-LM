#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEQUENCE_SESSION="${SEQUENCE_SESSION:-old_like_subset_router_20260812}"
SEQUENCE_ROOT="${SEQUENCE_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/code_old_like_gt_subset_router_replay_20260812}"
OCCURRENCE_ROOT="${OCCURRENCE_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/code_old_like_gt_token_occurrence_router_replay_20260812}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
QUEUE_LOG="$OCCURRENCE_ROOT/logs/queue_after_sequence.log"

SEQUENCE_LM="$SEQUENCE_ROOT/checkpoints/01_old_like_router_lm/g2-code-oldlike-subset-all8-top1-router-lm-mb48-gbs2304-s1800"
SEQUENCE_MSE="$SEQUENCE_ROOT/checkpoints/02_old_like_router_hidden_mse/g2-code-oldlike-subset-all8-top1-router-hiddenmse-c10-l2to9-mb48-gbs2304-s1800"

mkdir -p "$(dirname "$QUEUE_LOG")"

record() {
    printf '%s\t%s\n' "$(date -Is)" "$*" | tee -a "$QUEUE_LOG"
}

stage_complete() {
    local root="$1" objective="$2"
    [[ -s "$root/latest_checkpointed_iteration.txt" ]] || return 1
    [[ "$(tr -d '[:space:]' < "$root/latest_checkpointed_iteration.txt")" == "1800" ]] || return 1
    [[ -s "$root/stage_completion_manifest.json" ]] || return 1
    /data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin/python - \
        "$root/stage_completion_manifest.json" "$objective" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not (
    value.get("complete") is True
    and value.get("objective") == sys.argv[2]
    and value.get("gt_replay_unit", "positive_sequence") == "positive_sequence"
    and value.get("gt_replay_subset_count") == 798138
):
    raise SystemExit(1)
PY
}

record "waiting for verified positive-sequence LM and hidden-MSE stages"
wait_count=0
until stage_complete "$SEQUENCE_LM" old_like_router_lm && \
      stage_complete "$SEQUENCE_MSE" old_like_router_hidden_mse; do
    if ! tmux has-session -t "$SEQUENCE_SESSION" 2>/dev/null; then
        record "ERROR sequence chain ended without both verified step-1800 stages"
        exit 1
    fi
    wait_count=$((wait_count + 1))
    if (( wait_count % 10 == 0 )); then
        record "still waiting for sequence chain"
    fi
    sleep "$WAIT_SECONDS"
done

record "sequence chain verified; starting token-occurrence LM then hidden-MSE"
export MOE_JOINT_REPLAY_OLD_LIKE_UNIT=token_occurrence
export STUDY_ROOT="$OCCURRENCE_ROOT"
exec bash "$SCRIPT_DIR/run_g2_old_like_gt_lm_then_hidden_mse_4gpu_chain_mha.sh"
