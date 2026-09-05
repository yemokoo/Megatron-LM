#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Main sequence: the six baselines at their primary configuration.
bash "$DIR/train_dense_wiki.sh"
bash "$DIR/ewc.sh"
bash "$DIR/trace_gem.sh"
bash "$DIR/slora_pre.sh"
bash "$DIR/olora.sh"
bash "$DIR/sequential_dense.sh"
bash "$DIR/fixed_moe.sh"
echo "[MAIN DONE] six baselines at the primary configuration (15 stages)"

# Deferred last: the SLoRA rank sweep, which is an ablation rather than a
# baseline, so every method has a result before it starts.
bash "$DIR/slora_ablation.sh"
echo "[ALL DONE] six baselines and SLoRA rank ablations"
