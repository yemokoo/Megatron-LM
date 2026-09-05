#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PY_ENV="${PY_ENV:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100}"
PYTHON_BIN="${PYTHON_BIN:-$PY_ENV/bin/python}"
OUT_ROOT="${OUT_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/code_token_representation_gt_20260811}"
EXPECTED_WORKERS="${EXPECTED_WORKERS:-8}"
POLL_SECONDS="${POLL_SECONDS:-60}"
WATCH_LOG="$OUT_ROOT/logs/watcher.log"

mkdir -p "$OUT_ROOT/logs"
echo "[$(date --iso-8601=seconds)] watcher started" >> "$WATCH_LOG"
inactive_checks=0
while true; do
    "$PYTHON_BIN" scripts/analysis/analyze_code_token_hidden_pair.py \
        "$OUT_ROOT" --expected-workers "$EXPECTED_WORKERS" --write \
        >/dev/null 2>>"$WATCH_LOG" || true

    completed=0
    for ((worker=0; worker<EXPECTED_WORKERS; worker++)); do
        metadata="$(printf '%s/rank_%03d/metadata.json' "$OUT_ROOT" "$worker")"
        [[ -f "$metadata" ]] && completed=$((completed + 1))
    done
    echo "[$(date --iso-8601=seconds)] completed_workers=$completed/$EXPECTED_WORKERS" \
        >> "$WATCH_LOG"
    if (( completed == EXPECTED_WORKERS )); then
        "$PYTHON_BIN" scripts/analysis/validate_code_token_hidden_pair.py \
            --root "$OUT_ROOT" --deep --output "$OUT_ROOT/validation.json" \
            >> "$WATCH_LOG" 2>&1
        "$PYTHON_BIN" scripts/analysis/analyze_code_token_hidden_pair.py \
            "$OUT_ROOT" --expected-workers "$EXPECTED_WORKERS" --write \
            >/dev/null 2>>"$WATCH_LOG"
        echo "[$(date --iso-8601=seconds)] deep validation complete" >> "$WATCH_LOG"
        exit 0
    fi

    active="$({ pgrep -af 'pretrain_gpt.py.*--code-token-hidden-pair-path' || true; } \
        | grep -F -- "$OUT_ROOT/rank_" | wc -l)"
    if (( active == 0 )); then
        inactive_checks=$((inactive_checks + 1))
    else
        inactive_checks=0
    fi
    if (( inactive_checks >= 3 )); then
        echo "[$(date --iso-8601=seconds)] ERROR: extraction stopped before all workers completed" \
            >> "$WATCH_LOG"
        exit 1
    fi
    sleep "$POLL_SECONDS"
done
