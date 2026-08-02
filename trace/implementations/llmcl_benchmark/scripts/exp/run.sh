#!/bin/bash
# Launch any training/eval command in the background with clean, timestamped
# logging and PID tracking, so status/kill/tail scripts can find it reliably.
#
# Usage: bash scripts/exp/run.sh <name> <command...>
#   e.g. bash scripts/exp/run.sh track1_8b \
#          env EPOCHS=2,2,2,2,2,2,2,2 bash scripts/train_Ours_LoRA_MoE.sh 1 \
#              /home/work/Agent_HJ/00_models/Qwen3-8B
#
# Writes:
#   logs/<name>_<timestamp>.log   full unbuffered output
#   logs/<name>.pid               wrapper PID
#   logs/<name>.log -> ...        stable per-name symlink
#   logs/latest.log -> ...        most-recent-run symlink (used by status/kill/tail
#                                  when no name is given)
set -e
cd "$(dirname "$0")/../.."   # repo root
mkdir -p logs

NAME="$1"; shift || true
if [ -z "$NAME" ] || [ "$#" -eq 0 ]; then
  echo "Usage: $0 <name> <command...>"
  exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/${NAME}_${TS}.log"
PIDFILE="logs/${NAME}.pid"

export PATH=/home/work/Agent_HJ/30_flame_agent/envs/train_env/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

# Session-preservation daemon: if training dies (esp. CUDA OOM), the GPUs go idle
# and the platform may reclaim a low-utilization session. On such a death we launch
# agent_data_make.py to keep the GPUs warm. Set KEEPALIVE=0 to disable.
#   KEEPALIVE=1              start it immediately on a DEATH (OOM / nonzero exit).
#   KEEPALIVE_ON_FINISH=1    ALSO start it after a CLEAN finish, delayed by
#                            KEEPALIVE_FINISH_DELAY seconds (default 1200 = 20 min),
#                            to keep the session alive once the run ends.
KEEPALIVE=${KEEPALIVE:-1}
KEEPALIVE_ON_FINISH=${KEEPALIVE_ON_FINISH:-0}
KEEPALIVE_FINISH_DELAY=${KEEPALIVE_FINISH_DELAY:-1200}
KEEPALIVE_SCRIPT=${KEEPALIVE_SCRIPT:-/home/work/Agent_HJ/30_flame_agent/agent_data_make.py}
rm -f logs/.killed   # clear any stale intentional-kill sentinel from a previous run

# Helper (inherited by the ( ) subshell below): launch the keepalive daemon once.
start_keepalive () {
  nohup python "$KEEPALIVE_SCRIPT" >> logs/keepalive.log 2>&1 &
  echo "$!" > logs/keepalive.pid
}

# Supervisor: run the job in the background, and when it EXITS, decide whether/when
# to start the keepalive daemon. Runs in its own background subshell so run.sh still
# returns immediately.
(
  "$@" > "$LOG" 2>&1
  rc=$?
  if [ -f logs/.killed ]; then
    mode="killed"                       # intentional kill via kill.sh -> never keepalive
  elif grep -qiE "out of memory|OutOfMemoryError" "$LOG" 2>/dev/null; then
    mode="death"; reason="CUDA OOM"
  elif [ "$rc" -ne 0 ]; then
    mode="death"; reason="nonzero exit ($rc)"
  else
    mode="clean"                        # all tasks finished successfully
  fi

  if [ "$mode" = "death" ] && [ "$KEEPALIVE" = "1" ]; then
    echo "[run.sh] job '$NAME' ended: $reason -> starting session-keepalive daemon now" >> "$LOG"
    start_keepalive
  elif [ "$mode" = "clean" ] && [ "$KEEPALIVE_ON_FINISH" = "1" ]; then
    echo "[run.sh] job '$NAME' finished cleanly -> keepalive in ${KEEPALIVE_FINISH_DELAY}s" >> "$LOG"
    sleep "$KEEPALIVE_FINISH_DELAY"
    if [ -f logs/.killed ]; then
      echo "[run.sh] kill sentinel set during the delay -> keepalive NOT started" >> "$LOG"
    else
      echo "[run.sh] delay elapsed -> starting session-keepalive daemon" >> "$LOG"
      start_keepalive
    fi
  else
    echo "[run.sh] job '$NAME' exited rc=$rc (mode=$mode); keepalive not started" >> "$LOG"
  fi
) &
PID=$!
echo "$PID" > "$PIDFILE"
ln -sfn "$(basename "$LOG")" "logs/${NAME}.log"
ln -sfn "${NAME}_${TS}.log" "logs/latest.log"
echo "$NAME" > logs/latest.name

echo "launched '$NAME' (supervisor PID $PID)"
echo "log:    $LOG"
echo "        (symlinks: logs/${NAME}.log, logs/latest.log)"
echo "keepalive on death: ${KEEPALIVE} (agent_data_make.py; logs/keepalive.log)"
echo "status: bash scripts/exp/status.sh $NAME"
echo "tail:   bash scripts/exp/tail.sh $NAME     # live follow, Ctrl-C to stop"
echo "kill:   bash scripts/exp/kill.sh"
