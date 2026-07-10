#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_MASTER_PORT="${BASE_MASTER_PORT:-29920}"

MODES=(logits logits_hidden logits_hidden_router)
for index in "${!MODES[@]}"; do
    mode="${MODES[$index]}"
    port="$((BASE_MASTER_PORT + index))"
    echo "############################################################"
    echo "[START] shared-router expansion distill mode=$mode port=$port"
    echo "############################################################"
    MASTER_PORT="$port" \
        bash "$SCRIPT_DIR/run_g2_shared_router_code_expert_distill_init_mha.sh" "$mode"
    echo "[END] shared-router expansion distill mode=$mode"
done

echo "[ALL DONE] shared-router logits/logits_hidden/logits_hidden_router initialization"
