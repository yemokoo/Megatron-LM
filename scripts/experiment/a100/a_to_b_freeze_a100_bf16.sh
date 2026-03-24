#!/bin/bash
set -euo pipefail

export SOURCE_TASK=wiki
export TARGET_TASK=code
export FREEZE_SHARED=1
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_continual_moe_a100_bf16.sh"
