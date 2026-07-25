#!/bin/bash
# =============================================================================
# One-command launcher for the G2 shared-router (FFN + QKVO attention experts)
# pre-Code expert-init distillation 9-run experiment.
#
# Wiki 8-expert shared-router teacher -> 16-expert student. Full A -> B -> C
# pipeline across all three distill modes (9 runs), via run_all9.sh:
#   Stage A (x3): expand 8->16 + KL distill on WIKI      (mb48)
#   Stage B (x3): code training from the distill-init    (mb72)
#   Stage C (x3): router-only retune on WIKI+CODE mix    (mb72)
#
#   bash launch.sh
#
# Starts detached with nohup so it survives terminal disconnect.
# Re-running is safe: completed stages are auto-skipped via their checkpoint
# tracker, so a re-launch resumes after a failure.
# =============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DIR/../../../.." && pwd)"
DRIVER="$DIR/run_all9.sh"
LOGDIR="$PROJECT_ROOT/.local/logs/g2_shared_router_distill_init"
mkdir -p "$LOGDIR"

# ---- preflight: the 9-run driver + source checkpoint must exist -------------
if [ ! -f "$DRIVER" ]; then
    echo "[ERROR] 9-run driver not found: $DRIVER" >&2
    exit 1
fi

SRC="$PROJECT_ROOT/.local/weights/a100/mha/g2-checkpoints/wiki/g2-top4-e8-ffn352-r256-wiki-shared-router-qkvo-mha-a100-bf16-mb72-1800"
if [ ! -f "$SRC/latest_checkpointed_iteration.txt" ]; then
    echo "[ERROR] Wiki 8-expert shared-router source checkpoint not found:" >&2
    echo "        $SRC" >&2
    exit 1
fi

# ---- already running? -------------------------------------------------------
PIDFILE="$LOGDIR/run_all9.pid"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "[ERROR] a shared-router distill driver is already running (PID $(cat "$PIDFILE"))." >&2
    echo "        stop it first:  kill $(cat "$PIDFILE")" >&2
    exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER="$LOGDIR/run_all9_${STAMP}.log"

nohup bash "$DRIVER" > "$MASTER" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"

cat <<EOF

  launched G2 shared-router (FFN + attention) distill-init 9-run driver
  ---------------------------------------------------------------------
  PID          : $PID   (saved to $PIDFILE)
  modes        : logits -> logits_hidden -> logits_hidden_router
  order        : all A -> all B -> all C  (stage-major, sequential)
  source ckpt  : $SRC
  target root  : .local/weights/a100/mha/g2-checkpoints/code/shared_router_{expansion_distill_init,from_distill_init,from_distill_init_phase3}/
  master log   : $MASTER
  per-run logs : $LOGDIR/{A,B,C}_<mode>.log
  wandb        : offline  (sync later with: wandb sync <run-dir>)

  monitor      : tail -f $MASTER
  gpu watch    : watch -n5 nvidia-smi
  stop         : kill $PID && pkill -f torch.distributed.run

EOF
