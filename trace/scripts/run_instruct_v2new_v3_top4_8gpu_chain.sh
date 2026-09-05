#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point retained for the previous three-run command.
# The canonical Instruct bundle now includes the original V2 reference.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${ROOT}/scripts/run_instruct_v2_v2new_top4_v3_8gpu_chain.sh" "$@"
