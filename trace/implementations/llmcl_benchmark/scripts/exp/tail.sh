#!/bin/bash
# Live-follow an experiment log, filtered to the useful lines (raw logs are full
# of \r progress-bar spam -- this cleans it up so it's actually readable).
# Meant to be run interactively in your own terminal; Ctrl-C to stop.
#
# Usage: bash scripts/exp/tail.sh [name]
#   name  defaults to "latest" (whatever exp/run.sh launched most recently)
cd "$(dirname "$0")/../.."

NAME="${1:-latest}"
LOG="logs/${NAME}.log"
if [ ! -f "$LOG" ]; then
  echo "no log at $LOG"
  exit 1
fi

tail -f -n 200 "$LOG" | tr '\r' '\n' | grep --line-buffered -E \
  "phase1|phase2|step .*loss|Beginning|saving model|Sucessful|Running .* continual|Error|Traceback|CUDA out of memory|does not require grad"
