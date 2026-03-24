#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$PROJECT_ROOT/scripts/experiment/3090/stage_A_after_B_new_expert_router_only_local_fp32.sh"
