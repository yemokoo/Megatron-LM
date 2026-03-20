#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$PROJECT_ROOT/scripts/experiment/stage_B_first_local_fp32.sh"
