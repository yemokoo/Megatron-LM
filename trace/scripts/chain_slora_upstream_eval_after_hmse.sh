#!/bin/bash
# Finish the SLoRA-Pre reproduction's sparse-15 once hmse_kd200 vacates 0-3.
#
# Six of fifteen cells landed before the earlier attempt died; the remaining
# nine are the seven diagonals plus order8's NumGLUE-ds and 20Minuten.  The
# infer step runs with --resume, so completed cells are skipped rather than
# recomputed.
#
# This matters for the meeting: the results table's slora_rel* column is the
# published reference, not our reproduction.  Per-task claims against SLoRA
# need our own model scored in our own environment.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAIT_PID="${WAIT_PID:?export WAIT_PID (the process holding 0-3)}"
GPUS="${SLORA_GPUS:-0,1,2,3}"
LOG="${CHAIN_LOG:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/slora_pre_upstream/eval_chain.log}"
mkdir -p "$(dirname "${LOG}")"

say() { printf '[SLORA-EVAL %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG}"; }

say "waiting for pid ${WAIT_PID} (hmse_kd200 train+eval) to finish"
while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 60; done
say "pid ${WAIT_PID} gone"

for attempt in $(seq 1 60); do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', ' -v g="${GPUS}" 'index(","g",", ","$1",")>0 && $2>2000' | wc -l)
  [[ "${busy}" -eq 0 ]] && break
  say "waiting for ${busy} of ${GPUS} to release memory (attempt ${attempt})"
  sleep 30
done

export SLORA_GPUS="${GPUS}"
export SLORA_SKIP_TRAIN=1
export SLORA_LLAMA31_PATH="${SLORA_LLAMA31_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}"
export TRACE_DATA_ROOT="${TRACE_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

say "starting sparse-15 eval on ${GPUS} (9 cells outstanding)"
bash "${ROOT}/scripts/run_slora_pre_upstream_4gpu.sh" >> "${LOG}" 2>&1
status=$?
cells=$(find /data2/seonghyeonnoh/LLM-continual-learning-runs/trace/slora_pre_upstream/llama31/pre/evaluation \
        -name infer.jsonl 2>/dev/null | wc -l)
say "finished exit=${status}; ${cells}/15 cells present"
