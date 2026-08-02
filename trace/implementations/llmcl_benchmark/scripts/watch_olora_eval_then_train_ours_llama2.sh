#!/usr/bin/env bash
set -uo pipefail
EVAL_PID="${1:-2347667}"
BENCH_ROOT="/home/work/Agent_HJ/30_flame_agent/llmcl_benchmark"
FINAL_RESULT="/home/work/Agent_HJ/30_flame_agent/TreeLoRA/outputs_LLM-CL/reproduction/olora_llama2_7b_trace500_mb2_gb32_seed1234/predictions/final_results_sparse.txt"
WATCH_LOG="$BENCH_ROOT/logs/watch_olora_eval_then_train_ours_llama2.log"
TRAIN_QUEUE_LOG="$BENCH_ROOT/logs/ours_lora_moe_llama2_trace500_launcher.log"
TRAIN_SCRIPT="$BENCH_ROOT/scripts/train_Ours_LoRA_MoE_llama2_trace500.sh"
AGENT_PYTHON="/home/work/Agent_HJ/30_flame_agent/envs/train_env/bin/python"
AGENT_SCRIPT="/home/work/Agent_HJ/30_flame_agent/agent_data_make.py"
AGENT_LOG="$BENCH_ROOT/logs/agent_data_make_after_llama2_ours.log"
start_agent_data_make() {
  local reason="$1"
  if pgrep -f '[a]gent_data_make.py' >/dev/null 2>&1; then
    echo "[$(date --iso-8601=seconds)] agent_data_make already running; skip ($reason)" >> "$WATCH_LOG"
    return
  fi
  setsid env -u PYTHONPATH -u BNB_CUDA_VERSION "$AGENT_PYTHON" "$AGENT_SCRIPT" >> "$AGENT_LOG" 2>&1 < /dev/null &
  echo "[$(date --iso-8601=seconds)] launched agent_data_make PID=$! ($reason)" >> "$WATCH_LOG"
}
mkdir -p "$BENCH_ROOT/logs"
echo "[$(date --iso-8601=seconds)] watching O-LoRA eval PID=$EVAL_PID" >> "$WATCH_LOG"
while kill -0 "$EVAL_PID" 2>/dev/null; do sleep 15; done
sleep 10
if [[ ! -s "$FINAL_RESULT" ]]; then
  echo "[$(date --iso-8601=seconds)] eval ended without final result; training not started" >> "$WATCH_LOG"
  start_agent_data_make "evaluation failed, so training was not started"
  exit 1
fi
if pgrep -af 'training/main_Ours_LoRA_MoE.py.*Llama-2-7b-chat-hf' >/dev/null 2>&1; then
  echo "[$(date --iso-8601=seconds)] matching training already running; skip duplicate" >> "$WATCH_LOG"
  exit 0
fi
cd "$BENCH_ROOT"
setsid bash "$TRAIN_SCRIPT" >> "$TRAIN_QUEUE_LOG" 2>&1 < /dev/null &
train_pid=$!
echo "[$(date --iso-8601=seconds)] eval complete; launched training PID=$train_pid log=$TRAIN_QUEUE_LOG" >> "$WATCH_LOG"

sleep 15
if ! kill -0 "$train_pid" 2>/dev/null && ! pgrep -f '[a]gent_data_make.py' >/dev/null 2>&1; then
  start_agent_data_make "training failed to remain running after launch"
fi
