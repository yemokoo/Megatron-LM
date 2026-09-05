#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/train_stage.sh"
bash "$DIR/train_dense_wiki.sh"

WIKI="$BASELINES6_OUTPUT_ROOT/common_dense/wiki"
CODE="$BASELINES6_OUTPUT_ROOT/sequential_dense/code"
CONV="$BASELINES6_OUTPUT_ROOT/sequential_dense/conversation"
MB="$(method_micro_batch sequential_dense)"
run_baselines6_stage sequential_dense code "$CODE" "$WIKI" "" "$MB" 29751
run_baselines6_stage sequential_dense conversation "$CONV" "$CODE" "" "$MB" 29752
