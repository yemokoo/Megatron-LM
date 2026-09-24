#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:?usage: tab1_dymoe.sh <validate|train>}"
exec "${BASELINES_DIR}/_run_tab1.sh" "${ACTION}" llama31 dymoe
