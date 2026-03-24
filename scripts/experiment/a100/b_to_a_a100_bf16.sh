#!/bin/bash
set -euo pipefail

export SOURCE_TASK=code
export TARGET_TASK=wiki
export FREEZE_SHARED=0
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_continual_moe_a100_bf16.sh"
