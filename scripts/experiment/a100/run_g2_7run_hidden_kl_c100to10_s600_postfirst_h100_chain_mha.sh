#!/bin/bash
set -euo pipefail

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Old-data hidden-KL schedule for the continual Code/Conversation phases:
# update 0: 100, update 300: 55, update >=600: 10.
export OLD_HIDDEN_KL_COEFF=10
export OLD_HIDDEN_KL_COEFF_START=100
export OLD_HIDDEN_KL_COEFF_DECAY_STEPS=600
export HIDDEN_KL_EXPERIMENT_TAG=c100to10_s600_normfix
export HIDDEN_KL_RUN_TAG=c100to10-s600-normfix
export CHAIN_LOG_DIR="${CHAIN_LOG_DIR:-/data3/seonghyeonnoh/LLM-continual-learning-runs/local/logs/g2_7run_hidden_kl_c100to10_s600_postfirst_h100}"

exec bash "$D/run_g2_7run_hidden_kl_c10_normfix_postfirst_h100_chain_mha.sh"
