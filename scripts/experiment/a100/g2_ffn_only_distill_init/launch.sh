#!/bin/bash
# =============================================================================
# One-command launcher for the G2 FFN-only distill-init 9-run experiment.
# Starts run_all9.sh detached with nohup so it survives terminal disconnect.
#
#   bash launch.sh
#
# Re-running is safe: completed stages are auto-skipped (resume).
# =============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DIR/../../../.." && pwd)"
LOGDIR="$PROJECT_ROOT/.local/logs/g2_ffn_only_distill_init"
mkdir -p "$LOGDIR"

# ---- preflight: source 8-expert wiki checkpoint must exist ------------------
SRC_REG="$PROJECT_ROOT/.local/weights/a100/mha/g2-checkpoints/wiki/g2matched-top4-e8-ffn352-wiki-ffn-moe-mha-a100-bf16-mb128-1800"
SRC_ORIG="$PROJECT_ROOT/.local/weights/a100/mha/wiki-a-moe-g2matched-bf16/g2matched-top4-e8-ffn352-wiki-ffn-moe-mha-a100-bf16-mb128-1800"
if [ ! -f "$SRC_REG/latest_checkpointed_iteration.txt" ] && [ ! -f "$SRC_ORIG/latest_checkpointed_iteration.txt" ]; then
    echo "[ERROR] 8-expert wiki source checkpoint not found:" >&2
    echo "        $SRC_REG" >&2
    echo "        $SRC_ORIG" >&2
    exit 1
fi

# ---- already running? -------------------------------------------------------
PIDFILE="$LOGDIR/run_all9.pid"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "[ERROR] a run_all9 driver is already running (PID $(cat "$PIDFILE"))." >&2
    echo "        stop it first:  kill $(cat "$PIDFILE")" >&2
    exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER="$LOGDIR/run_all9_${STAMP}.log"

nohup bash "$DIR/run_all9.sh" > "$MASTER" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"

cat <<EOF

  launched G2 FFN-only distill-init 9-run driver
  ----------------------------------------------
  PID          : $PID   (saved to $PIDFILE)
  master log   : $MASTER
  per-run logs : $LOGDIR/{A,B,C}_<mode>.log
  wandb        : offline  (sync later with: wandb sync <run-dir>)

  monitor      : tail -f $MASTER
  gpu watch    : watch -n5 nvidia-smi
  stop         : kill $PID && pkill -f torch.distributed.run

EOF
