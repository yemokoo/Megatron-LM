#!/usr/bin/env bash
set -euo pipefail
# V3-new, expert-per-task 1, top-1, with a doubled router-replay stream.
#
# v3_new pins active_memory_cap=1000 / persistent=500 / ratio 5:1, which makes
# the replay share 1000/(5000+1000) = 16.7% of per-epoch exposure from round 2
# on.  This profile doubles the stream (cap 2000, persistent 1000) so replay
# reaches 2000/(5000+2000) = 28.6%.  Round 1 has only one task stored, so the
# water-filling allocator naturally caps it at the 1000 available -- identical
# to v3_new's round 1.  Nothing else changes: architecture, KD, LM replay
# objective, LR schedule and expert count all match v3_new.
#
# Replay volume is the only variable here.  The early-boost quota schedule is
# deliberately left off: --v2_new_expert_quota_schedule is gated to v2_new in
# the trainer, and unpicking that gate would confound this measurement with a
# second change.  It gets its own run once the replay effect is known.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${OURS_ST_RUN_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/v3_replay40}"
OUTPUT="${RUN_ROOT}/v3_new_replay40_st_top1_cap2000"
if [[ -e "${OUTPUT}/0/lora_moe_meta.json" && -z "${OURS_LORAMOE_RESUME_CHECKPOINT:-}" ]]; then
  echo "[ERROR] refusing to overwrite existing run: ${OUTPUT}" >&2
  exit 2
fi
mkdir -p "${OUTPUT}"
export PYTHONNOUSERSITE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export OURS_LORAMOE_OUTPUT_ROOT="${OUTPUT}"
export OURS_LORAMOE_GPUS="${OURS_LORAMOE_GPUS:-4,5,6,7}"
export OURS_LORAMOE_PORT="${OURS_LORAMOE_PORT:-29741}"
# 4 GPUs at micro 8 x accum 2 keeps the published effective global batch of 64.
export OURS_LORAMOE_MICRO_BATCH="${OURS_LORAMOE_MICRO_BATCH:-8}"
export OURS_LORAMOE_GRAD_ACCUM="${OURS_LORAMOE_GRAD_ACCUM:-2}"
export OURS_LORAMOE_TOP_K=1
export OURS_LORAMOE_RANK="${OURS_LORAMOE_RANK:-64}"
export OURS_LORAMOE_ALPHA="${OURS_LORAMOE_ALPHA:-128}"
export OURS_LORAMOE_EXPERTS_PER_TASK=1
export OURS_LORAMOE_ROUTING_WEIGHT_MODE=straight_through_topk
export OURS_V2_JOINT_NEW_TO_REPLAY_RATIO=5
# The doubled replay stream and the pool that has to fill it.
export OURS_V2_NEW_ACTIVE_MEMORY_CAP=2000
export OURS_V2_NEW_PERSISTENT_SAMPLES_PER_TASK=1000
export OURS_REPLAY_SUBSET_RATIO=0.2
export OURS_ROUTER_REPLAY_EXPOSURE_SAMPLES=2000
export OURS_V2_KD_MEMORY_BATCH_SIZE=8
export OURS_V3_EPOCH_PROBE_SAMPLES=64
export OURS_REPLAY_SELECTION_MODE="${OURS_REPLAY_SELECTION_MODE:-random}"
echo "[TRAIN START] v3_new_replay40 GPUs=${OURS_LORAMOE_GPUS} cap=2000 persistent=1000 (replay-only change)"
exec bash "${ROOT}/scripts/baselines/_run_ours_lora_moe.sh" train llama31 v3_new_replay40
