#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace"
RUN="/data3/seonghyeonnoh/LLM-continual-learning-runs/local/trace/results/full_runs/llama31/ours_lora_moe_v3_st_top1_fixed1000_epochprobe64"
CHECKPOINT="${RUN}/2"
OLD_TORCHRUN_PID=3648444
WATCH_LOG="${RUN}/router_gradient_resume_watch.log"

exec >>"${WATCH_LOG}" 2>&1
echo "[$(date --iso-8601=seconds)] waiting for MeetingBank checkpoint 2"

while [[ ! -s "${CHECKPOINT}/pytorch_model.bin" || ! -s "${CHECKPOINT}/lora_moe_meta.json" ]]; do
  if ! kill -0 "${OLD_TORCHRUN_PID}" 2>/dev/null; then
    echo "[$(date --iso-8601=seconds)] old torchrun exited before checkpoint 2"
    exit 1
  fi
  sleep 2
done

"${ROOT}/.venv-runtime/bin/python" -c \
  'import json,sys; json.load(open(sys.argv[1], encoding="utf-8"))' \
  "${CHECKPOINT}/lora_moe_meta.json"
echo "[$(date --iso-8601=seconds)] checkpoint 2 complete; stopping old process"
kill -TERM "${OLD_TORCHRUN_PID}"

for _ in $(seq 1 60); do
  kill -0 "${OLD_TORCHRUN_PID}" 2>/dev/null || break
  sleep 2
done
if kill -0 "${OLD_TORCHRUN_PID}" 2>/dev/null; then
  echo "[$(date --iso-8601=seconds)] old torchrun did not stop cleanly"
  exit 1
fi

sleep 3
echo "[$(date --iso-8601=seconds)] resuming at Py150 with router-gradient memory"
cd "${ROOT}"
export OURS_LORAMOE_RESUME_CHECKPOINT="${CHECKPOINT}"
export OURS_REPLAY_SELECTION_MODE=router_gradient
nohup bash "${ROOT}/scripts/run_v3_st_top1_8gpu.sh" \
  >>"${RUN}/launcher.log" 2>&1 &
echo "[$(date --iso-8601=seconds)] resume launcher pid=$!"
