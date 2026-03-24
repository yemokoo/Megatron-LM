#!/bin/bash
set -euo pipefail

# Task B after Task A with a fixed 7-expert MoE and no freezing.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export NUM_EXPERTS="${NUM_EXPERTS:-7}"
export STAGE_A_REQUIRED_ITERS="${STAGE_A_REQUIRED_ITERS:-1}"
export RUN_ID="${RUN_ID:-stage-b-after-a-7experts-no-freeze-local-fp32-$(date -u +%Y%m%d-%H%M%S)}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$PROJECT_ROOT/.local/weights/continual-stage-B-after-A-7experts-no-freeze-local/$RUN_ID}"

exec bash "$PROJECT_ROOT/scripts/experiment/3090/stage_B_after_A_no_freeze_local_fp32.sh"
