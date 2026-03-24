#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

source scripts/experiment/presets/local_3090_fp16.sh

export RUN_ID="${RUN_ID:-stage-a-local-3090-$(date -u +%Y%m%d-%H%M%S)}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-$RUN_ID}"
export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/tmp/flame-moe}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-300}"

exec bash scripts/experiment/3090/stage_A_local.sh
