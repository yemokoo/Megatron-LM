#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point. The canonical order now prioritizes V3-new and
# V2-new-top4, evaluates them, then trains/evaluates V2 and V2-new.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${ROOT}/scripts/run_instruct_priority_fourway_train_eval_chain.sh" "$@"
