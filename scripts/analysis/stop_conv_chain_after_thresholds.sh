#!/usr/bin/env bash
set -uo pipefail

# The 0,1 chain ends with a Conversation random arm that duplicates step 3 of the
# 2,3 random chain, which is the one that belongs to the random lineage.  The
# chain is already running, so rather than editing a live script this waits for
# the extraction to finish and stops the chain before that arm starts.

C=/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_conv_census_20260817
STATUS="$C/logs/overnight_status.tsv"
CHAIN_PID="${CHAIN_PID:?set CHAIN_PID to the overnight chain pid}"

while kill -0 "$CHAIN_PID" 2>/dev/null; do
    if grep -q "THRESHOLDS-DONE" "$STATUS" 2>/dev/null; then
        # Stop the chain and any census/training it spawned, leaving every
        # extraction artifact in place.
        pkill -TERM -P "$CHAIN_PID" 2>/dev/null
        kill -TERM "$CHAIN_PID" 2>/dev/null
        sleep 10
        pkill -KILL -P "$CHAIN_PID" 2>/dev/null
        kill -KILL "$CHAIN_PID" 2>/dev/null
        echo "$(date -Is) CHAIN-STOPPED-AFTER-THRESHOLDS (conv random left to the 2,3 chain)" \
            | tee -a "$STATUS"
        exit 0
    fi
    if grep -qE "CONV-RANDOM-START" "$STATUS" 2>/dev/null; then
        echo "$(date -Is) WARN conv-random already started on 0,1; not interfering" \
            | tee -a "$STATUS"
        exit 1
    fi
    sleep 60
done
echo "$(date -Is) CHAIN-ALREADY-EXITED pid=$CHAIN_PID" | tee -a "$STATUS"
