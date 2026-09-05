#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:?usage: ours_lora_moe_v2_new.sh <validate|train|eval|all>}"
exec "${BASELINES_DIR}/_run_model.sh" qwen25_7b "${ACTION}" ours_lora_moe_v2_new
