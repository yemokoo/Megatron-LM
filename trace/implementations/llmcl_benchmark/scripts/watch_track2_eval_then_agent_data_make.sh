#!/usr/bin/env bash
set -u

EVAL_PID="${1:?usage: $0 EVAL_PID}"
BENCH=/home/work/Agent_HJ/30_flame_agent/llmcl_benchmark
EVAL_ROOT="$BENCH/eval_out/track2_OLMoE_ept1_force_upper_5k_seed1234_sparse"
DRIVER_LOG="$BENCH/eval_out/track2_OLMoE_ept1_force_upper_5k_seed1234_sparse_driver_batch16.log"
AGENT=/home/work/Agent_HJ/30_flame_agent/agent_data_make.py
AGENT_LOG="$BENCH/eval_out/track2_OLMoE_agent_data_make.log"

echo "watching evaluation pid=$EVAL_PID"
while kill -0 "$EVAL_PID" 2>/dev/null; do
  sleep 10
done

reason=""
if [[ -s "$EVAL_ROOT/cl_summary_sparse.json" ]]; then
  reason="evaluation completed"
elif grep -RqiE 'CUDA out of memory|OutOfMemoryError' \
    "$EVAL_ROOT/logs" "$DRIVER_LOG" 2>/dev/null; then
  reason="evaluation OOM"
fi

if [[ -z "$reason" ]]; then
  echo "evaluation ended without completion marker or OOM; agent_data_make not started"
  exit 1
fi

if pgrep -f "python .*agent_data_make.py" >/dev/null; then
  echo "$reason; agent_data_make.py already running"
  exit 0
fi

echo "$reason; starting agent_data_make.py"
cd /home/work/Agent_HJ/30_flame_agent
setsid python "$AGENT" >>"$AGENT_LOG" 2>&1 < /dev/null &
echo "agent_data_make pid=$!"
