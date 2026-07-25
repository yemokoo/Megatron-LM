#!/bin/bash
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/../../.." && pwd)"
cd "$R"
TARGET=g2-kdinit-conv-stepwise-continue-v8-5501-5600-mb96
DELAY_SECONDS="${DELAY_SECONDS:-1800}"
echo "[$(date -u '+%F %T UTC')] scheduled: wait ${DELAY_SECONDS}s"
sleep "$DELAY_SECONDS"
mapfile -t pids < <(pgrep -f "pretrain_gpt.py.*${TARGET}|torch.distributed.run.*${TARGET}" || true)
if [ "${#pids[@]}" -gt 0 ]; then
  echo "[$(date -u '+%F %T UTC')] TERM v8 PIDs: ${pids[*]}"
  kill -TERM "${pids[@]}" 2>/dev/null || true
fi
for _ in $(seq 1 60); do
  pgrep -f "pretrain_gpt.py.*${TARGET}|torch.distributed.run.*${TARGET}" >/dev/null || break
  sleep 2
done
mapfile -t remaining < <(pgrep -f "pretrain_gpt.py.*${TARGET}|torch.distributed.run.*${TARGET}" || true)
if [ "${#remaining[@]}" -gt 0 ]; then
  echo "[$(date -u '+%F %T UTC')] KILL remaining v8 PIDs: ${remaining[*]}"
  kill -KILL "${remaining[@]}" 2>/dev/null || true
fi
for _ in $(seq 1 60); do
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]' || break
  sleep 2
done
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo 'ERROR: GPU compute processes remain; refusing to start chain.' >&2
  nvidia-smi
  exit 1
fi
echo "[$(date -u '+%F %T UTC')] GPUs free; starting no-KD chain"
exec bash "$D/run_g2_ffn_only_no_kd_joint_lm_code_conversation_chain_mha.sh"
