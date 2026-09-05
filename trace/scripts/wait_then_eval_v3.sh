#!/bin/bash
# Evaluate one v3 variant, waiting first if it is still training.
#
# The series no longer holds every evaluation until the last model finishes:
# with two lanes and one long training left, that left a lane idle for hours.
# Instead each model's evaluation is its own queue unit, ordered by expected
# completion, and this waits on that model's round-7 checkpoint so a unit that
# is popped early blocks itself rather than failing.
#
#   wait_then_eval_v3.sh <slug> <version> <gpus>
set -uo pipefail
SLUG="${1:?usage: wait_then_eval_v3.sh <slug> <version> <gpus>}"
VERSION="${2:?usage: wait_then_eval_v3.sh <slug> <version> <gpus>}"
GPUS="${3:?usage: wait_then_eval_v3.sh <slug> <version> <gpus>}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT="/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/${SLUG}/${VERSION}_st_top1/7/lora_moe_meta.json"

if [[ ! -e "${CKPT}" ]]; then
  echo "[EVAL-WAIT] ${VERSION} is still training; waiting for round 7 $(date '+%F %T')"
  while [[ ! -e "${CKPT}" ]]; do sleep 120; done
  echo "[EVAL-WAIT] ${VERSION} finished training $(date '+%F %T')"
  # The trainer writes the checkpoint shards before its metadata, but give the
  # save a moment to settle before another process opens it.
  sleep 30
fi
exec bash "${DIR}/run_v3_job.sh" "${SLUG}" "${VERSION}" "${GPUS}" eval
