#!/bin/bash
set -euo pipefail

export TASK_NAME=wiki
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_base_moe_a100_bf16.sh"
