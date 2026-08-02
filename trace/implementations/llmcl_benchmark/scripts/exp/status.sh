#!/bin/bash
# One-shot snapshot: GPU usage, running training processes, and the tail of the
# relevant log lines (filtered -- raw logs are full of \r progress-bar noise).
#
# Usage: bash scripts/exp/status.sh [name] [n_lines]
#   name     defaults to "latest" (whatever exp/run.sh launched most recently)
#   n_lines  defaults to 20
cd "$(dirname "$0")/../.."

NAME="${1:-latest}"
N="${2:-20}"
LOG="logs/${NAME}.log"

echo "================ GPU ================"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo
echo "================ PROCESSES ================"
pgrep -af "main_Ours_LoRA_MoE|main_Ours_MoE_FFN|main_baseline|torchrun|torch.distributed.run" | grep -v grep || echo "(none running)"

echo
echo "================ LOG: $LOG (last $N relevant lines) ================"
if [ -f "$LOG" ]; then
  tr '\r' '\n' < "$LOG" \
    | grep -E "phase1|phase2|step .*loss|Beginning|saving model|Sucessful|Running .* continual|Error|Traceback|CUDA out of memory|does not require grad" \
    | tail -n "$N"
else
  echo "(no log found at $LOG -- pass a name matching logs/<name>.log, or run exp/run.sh first)"
fi
