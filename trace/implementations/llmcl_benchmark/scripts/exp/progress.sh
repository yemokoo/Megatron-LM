
#!/bin/bash
# Show current training progress: task/phase, step X/Y, %, elapsed<ETA, it/s.
# tqdm's own progress line already carries step count + ETA -- it's just written
# with \r (overwrite) instead of newlines, so a raw `tail` shows a messy wall of
# concatenated history. This grabs only the LATEST \r-delimited update.
#
# Usage: bash scripts/exp/progress.sh [name]
#   name  defaults to "latest" (whatever exp/run.sh launched most recently)
#
# Tip: for a live-refreshing view, run:
#   watch -n 5 bash scripts/exp/progress.sh [name]
cd "$(dirname "$0")/../.."

NAME="${1:-latest}"
LOG="logs/${NAME}.log"
if [ ! -f "$LOG" ]; then
  echo "no log at $LOG"
  exit 1
fi

echo "================ current step (log: $LOG) ================"
CUR=$(tail -c 4000 "$LOG" | tr '\r' '\n' | grep -E '[0-9]+%\|.*(it/s|s/it)' | tail -1)
if [ -n "$CUR" ]; then
  echo "$CUR"
else
  echo "(no progress line yet -- still loading model/data)"
fi

echo
echo "================ recent task/phase transitions ================"
tr '\r' '\n' < "$LOG" \
  | grep -E "Running .* continual|: epoch 1/|saving model|Sucessful" \
  | tail -6
