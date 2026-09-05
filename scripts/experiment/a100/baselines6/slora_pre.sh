#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/train_stage.sh"
bash "$DIR/train_dense_wiki.sh"

WIKI="$BASELINES6_OUTPUT_ROOT/common_dense/wiki"
# Only the primary rank runs with the other baselines; slora_ablation.sh
# sweeps the remaining ranks after the main sequence finishes.
for RANK in ${SLORA_RANKS:-64}; do
    CODE="$BASELINES6_OUTPUT_ROOT/slora_pre/rank${RANK}/code"
    CONV="$BASELINES6_OUTPUT_ROOT/slora_pre/rank${RANK}/conversation"
    MB="$(method_micro_batch slora_pre)"
    PORT_OFFSET=$((RANK % 100))
    run_baselines6_stage slora_pre code "$CODE" "$WIKI" "" "$MB" "$((29730 + PORT_OFFSET))" \
        --continual-slora-rank "$RANK" --continual-slora-conversation-rank 64 \
        --continual-slora-max-rank 256 --continual-slora-alpha 128
    run_baselines6_stage slora_pre conversation "$CONV" "$CODE" "$(state_dir "$CODE" slora_pre)" "$MB" "$((29830 + PORT_OFFSET))" \
        --continual-slora-rank "$RANK" --continual-slora-conversation-rank 64 \
        --continual-slora-max-rank 256 --continual-slora-alpha 128
done
