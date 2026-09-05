#!/bin/bash
set -euo pipefail

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# H100 memory-calibrated settings from the completed runs:
#   code expansion mb36: ~61.3 GiB peak
#   code hidden-KL mb36: ~40.2 GiB peak -> mb48
#   conversation expansion mb24: ~48.2 GiB peak -> mb32
#   conversation hidden-KL mb24: ~34.2 GiB peak -> mb36
export OLD_HIDDEN_KL_COEFF="${OLD_HIDDEN_KL_COEFF:-10}"
export CODE_EXPAND_MB="${CODE_EXPAND_MB:-36}"
export CODE_PHASE_MB="${CODE_PHASE_MB:-48}"
export CONV_EXPAND_MB="${CONV_EXPAND_MB:-32}"
export CONV_PHASE_MB="${CONV_PHASE_MB:-36}"

export R2_RUN_ID="${R2_RUN_ID:-r2-code-hiddenkl-c10-normfix-teacher-pre8-mb48-1800}"
export R3_RUN_ID="${R3_RUN_ID:-r3-conv-expand-c10-normfix-wikicode-kd-prebranch-e16to24-mb32-600}"
export R4_RUN_ID="${R4_RUN_ID:-r4-conv-hiddenkl-c10-normfix-teacher-pre16-mb36-1800}"
export R5_RUN_ID="${R5_RUN_ID:-r5-code-hiddenkl-c10-normfix-teacher-post16-mb48-1800}"
export R6_RUN_ID="${R6_RUN_ID:-r6-conv-expand-c10-normfix-wikicode-kd-postbranch-e16to24-mb32-600}"
export R7_RUN_ID="${R7_RUN_ID:-r7-conv-hiddenkl-c10-normfix-teacher-post24-mb36-1800}"
export CHAIN_LOG_DIR="${CHAIN_LOG_DIR:-/data3/seonghyeonnoh/LLM-continual-learning-runs/local/logs/g2_7run_hidden_kl_c10_normfix_h100}"

exec bash "$D/run_g2_7run_hidden_kl_teacher_ablation_chain_mha.sh"
