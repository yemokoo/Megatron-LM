#!/usr/bin/env bash
# Wait for orphaned eval shards from the crashed run to drain, then resume the
# sparse-15 queue with a Py150 batch that fits an 80 GB card (16 OOMed).
set -uo pipefail
TRACE=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace
RUN=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/selfgen_cl_20260831
echo "[RESUME $(date '+%F %T')] draining orphaned eval shards"
while pgrep -f "evaluate_Ours_LoRA_MoE.py" > /dev/null; do sleep 60; done
echo "[RESUME $(date '+%F %T')] drained; restarting sparse-15 with PY150_BATCH=8"
export SPARSE15_PY150_BATCH=8
exec bash "$TRACE/scripts/selfgen/eval_selfgen_sparse15.sh"
