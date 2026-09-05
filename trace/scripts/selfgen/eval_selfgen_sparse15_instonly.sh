#!/usr/bin/env bash
# (fork of eval_selfgen_sparse15.sh: instonly trainer name + GPU list from env)
# Wait for the self-generated-replay CL chain to finish round 7, then run the
# sparse-15 evaluation with exactly the settings used for the `lm` baseline
# (method string ours_lora_moe_v3_new_replay1to1, OP 63.51).
set -uo pipefail
TRACE=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace
cd "$TRACE"
RUN=${RUN:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/selfgen_cl_20260831}
OUT=$RUN/model
log() { echo "[EVAL $(date '+%F %T')] $*"; }

log "waiting for $OUT/7"
while :; do
  if [[ -s "$OUT/7/pytorch_model.bin" && -s "$OUT/7/lora_moe_meta.json" ]] \
     && ! pgrep -f "${TRAIN_PROC_PATTERN:-train_selfgen(_instonly)?\.py}" > /dev/null; then
    break
  fi
  sleep 60
done
log "round 7 checkpoint complete; starting sparse-15 on GPUs ${SPARSE15_GPUS:-0-7}"

export PYTHONNOUSERSITE=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
export SLORA_LLAMA31_PATH="${SLORA_LLAMA31_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}"
export TRACE_DATA_ROOT="${TRACE_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace}"
export OURS_LORAMOE_OUTPUT_ROOT="$OUT"
export SPARSE15_METHODS=ours_lora_moe_v3_new_replay1to1
export SPARSE15_GPUS=${SPARSE15_GPUS:-0,1,2,3,4,5,6,7}
export SPARSE15_EVAL_BATCH=${SPARSE15_EVAL_BATCH:-32}
export SPARSE15_SCIENCEQA_BATCH=${SPARSE15_SCIENCEQA_BATCH:-192}
export SPARSE15_20MINUTEN_BATCH=${SPARSE15_20MINUTEN_BATCH:-32}
export SPARSE15_MEETINGBANK_BATCH=${SPARSE15_MEETINGBANK_BATCH:-2}
export SPARSE15_PY150_BATCH=${SPARSE15_PY150_BATCH:-16}
export SPARSE15_CPU_THREADS=${SPARSE15_CPU_THREADS:-4}
export SPARSE15_LOG_ROOT=${SPARSE15_LOG_ROOT:-$RUN/eval_queue_logs}

"$TRACE/.venv-runtime/bin/python" -u "$TRACE/scripts/run_ours_sparse15_efficient.py"
rc=$?
if [[ -s "$OUT/sparse15_summary.json" ]]; then
  log "DONE  OP=$(python3 -c "import json;print(round(json.load(open('$OUT/sparse15_summary.json'))['OP'],2))")"
else
  log "FAILED rc=$rc — no summary at $OUT/sparse15_summary.json"
fi
