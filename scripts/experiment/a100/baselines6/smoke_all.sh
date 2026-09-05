#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BASELINES6_SMOKE=1
export PAUSE_SECONDS=0
export SLORA_RANKS="${SLORA_SMOKE_RANKS:-16}"

bash "$DIR/run_all.sh"
