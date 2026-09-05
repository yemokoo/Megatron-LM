#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${OURS_V3_RUN:-/data3/seonghyeonnoh/LLM-continual-learning-runs/local/trace/results/full_runs/llama31/ours_lora_moe_v3_st_top1_fixed1000_epochprobe64}"
RESUME_CHECKPOINT="${OURS_V3_RESUME_CHECKPOINT:-$RUN/2}"

[[ -s "$RESUME_CHECKPOINT/pytorch_model.bin" ]] || {
    echo "[ERROR] missing V3 resume weights: $RESUME_CHECKPOINT" >&2
    exit 2
}
[[ -s "$RESUME_CHECKPOINT/lora_moe_meta.json" ]] || {
    echo "[ERROR] missing V3 resume metadata: $RESUME_CHECKPOINT" >&2
    exit 2
}

export PYTHONNOUSERSITE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

if [[ ! -s "$RUN/7/pytorch_model.bin" || ! -s "$RUN/7/lora_moe_meta.json" ]]; then
    echo "[TRACE TRAIN] resuming V3 from $RESUME_CHECKPOINT"
    export OURS_LORAMOE_RESUME_CHECKPOINT="$RESUME_CHECKPOINT"
    export OURS_REPLAY_SELECTION_MODE=router_gradient
    bash "$ROOT/scripts/run_v3_st_top1_8gpu.sh"
else
    echo "[TRACE TRAIN] final checkpoint 7 already complete; skipping training"
fi

[[ -s "$RUN/7/pytorch_model.bin" && -s "$RUN/7/lora_moe_meta.json" ]] || {
    echo "[ERROR] V3 training returned without a complete checkpoint 7" >&2
    exit 3
}

export OURS_LORAMOE_OUTPUT_ROOT="$RUN"
export SPARSE15_METHODS=ours_lora_moe_v3
export SPARSE15_GPUS=0,1,2,3,4,5,6,7
export SPARSE15_EVAL_BATCH="${SPARSE15_EVAL_BATCH:-32}"
export SPARSE15_SCIENCEQA_BATCH="${SPARSE15_SCIENCEQA_BATCH:-192}"
export SPARSE15_20MINUTEN_BATCH="${SPARSE15_20MINUTEN_BATCH:-32}"
export SPARSE15_MEETINGBANK_BATCH="${SPARSE15_MEETINGBANK_BATCH:-2}"
export SPARSE15_PY150_BATCH="${SPARSE15_PY150_BATCH:-16}"
export SPARSE15_CPU_THREADS="${SPARSE15_CPU_THREADS:-4}"
export SPARSE15_LOG_ROOT="${SPARSE15_LOG_ROOT:-$RUN/eval_queue_logs}"

echo "[TRACE EVAL] starting sparse-15 evaluation on GPUs 0-7"
"$ROOT/.venv-runtime/bin/python" -u "$ROOT/scripts/run_ours_sparse15_efficient.py"
[[ -s "$RUN/sparse15_summary.json" ]] || {
    echo "[ERROR] sparse-15 evaluator returned without summary" >&2
    exit 4
}
echo "[TRACE DONE] V3 training and sparse-15 evaluation complete"
