#!/bin/bash
# Launch the SLoRA joint-replay control on 4,5,6,7 once kd200 vacates them.
#
# The ablation lane process is the right thing to wait on: it owns kd200's
# train *and* eval, and exits on its own once the queue drains.  Polling
# nvidia-smi alone would race the gap between the two phases.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LANE_PID="${LANE_PID:?export LANE_PID (the ablation lane owning 4567)}"
GPUS="${SLORA_GPUS:-4,5,6,7}"
LOG="${CHAIN_LOG:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/slora_pre_replay/chain.log}"
mkdir -p "$(dirname "${LOG}")"

say() { printf '[CHAIN %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG}"; }

say "waiting for lane pid ${LANE_PID} (kd200 train+eval) to finish"
while kill -0 "${LANE_PID}" 2>/dev/null; do sleep 60; done
say "lane pid ${LANE_PID} gone"

# The lane exits when its last child returns, but freeing device memory can
# lag; starting into a still-occupied card is how a run OOMs at step 0.
for attempt in $(seq 1 60); do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', ' -v g="${GPUS}" 'index(","g",", ","$1",")>0 && $2>2000' | wc -l)
  [[ "${busy}" -eq 0 ]] && break
  say "waiting for ${busy} of ${GPUS} to release memory (attempt ${attempt})"
  sleep 30
done

export SLORA_GPUS="${GPUS}"
export WORLD_SIZE=4
export SLORA_LLAMA31_PATH="${SLORA_LLAMA31_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}"
export TRACE_DATA_ROOT="${TRACE_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

say "starting SLoRA joint-replay control on ${GPUS} (train + sparse-15 eval)"
bash "${ROOT}/scripts/run_slora_pre_replay_4gpu.sh" >> "${LOG}" 2>&1
say "finished exit=$?"
