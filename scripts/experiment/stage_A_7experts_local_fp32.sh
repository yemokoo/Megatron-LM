#!/bin/bash
set -euo pipefail

# Task A from scratch with a fixed 7-expert MoE.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export NUM_EXPERTS="${NUM_EXPERTS:-7}"
export RUN_ID="${RUN_ID:-stage-a-7experts-local-fp32-$(date -u +%Y%m%d-%H%M%S)}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$PROJECT_ROOT/.local/weights/continual-stage-A-7experts-local/$RUN_ID}"

exec bash "$PROJECT_ROOT/scripts/experiment/stage_A_local_fp32.sh"
