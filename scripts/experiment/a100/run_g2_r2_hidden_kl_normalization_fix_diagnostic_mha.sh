#!/bin/bash
set -euo pipefail

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/data2/seonghyeonnoh/condatest/miniconda3}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate flame-megatron-h100

LOCAL_BASE="${LOCAL_BASE:-/data3/seonghyeonnoh/LLM-continual-learning-runs/local}"
export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/data3/seonghyeonnoh/LLM-continual-learning-staging/flame-moe}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-/data3/seonghyeonnoh/LLM-continual-learning-models/pythia-12b-tokenizer}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
G2_ROOT="$LOCAL_BASE/weights/a100/mha/g2-checkpoints"
WIKI_SOURCE="${WIKI_SOURCE:-/data3/seonghyeonnoh/LLM-continual-learning-runs/wiki/g2-wiki-e8-ffn352-top4-h100-mb72-1800}"
R1_SOURCE="${R1_SOURCE:-$G2_ROOT/code/expansion_distill_init/r1-code-expand-wiki-kd-e8to16-mb36-600}"

STEPS="${TRAIN_ITERS:-300}"
MB="${MICRO_BATCH_SIZE:-36}"
COEFF="${OLD_HIDDEN_KL_COEFF:-1.0}"
RUN_ID="${RUN_ID:-r2-diagnostic-hiddenkl-normfix-v3-c${COEFF}-mb${MB}-${STEPS}}"
OUT="$G2_ROOT/code/joint_old_data_hidden_kl/normalization_fix_diagnostic/$RUN_ID"

echo "[DIAGNOSTIC] env=$CONDA_DEFAULT_ENV python=$(command -v python)"
echo "[DIAGNOSTIC] source=$R1_SOURCE"
echo "[DIAGNOSTIC] teacher=$WIKI_SOURCE (8E)"
echo "[DIAGNOSTIC] hidden_kl_coeff=$COEFF steps=$STEPS eval_interval=50"
echo "[DIAGNOSTIC] output=$OUT"
echo "[DIAGNOSTIC] staging_root=$LOCAL_SSD_ROOT"
echo "[DIAGNOSTIC] tokenizer=$TOKENIZER_MODEL gpus=$CUDA_VISIBLE_DEVICES nproc=$NPROC_PER_NODE"

SOURCE_WEIGHTS_DIR="$R1_SOURCE" \
SOURCE_REQUIRED_ITERS=600 \
OLD_MODEL_KL_WEIGHTS_DIR="$WIKI_SOURCE" \
OLD_MODEL_KL_NUM_EXPERTS=8 \
OLD_HIDDEN_KL_COEFF="$COEFF" \
OLD_HIDDEN_KL_TEMPERATURE=1.0 \
OLD_HIDDEN_KL_LAYERS=all_but_last \
TRAIN_ITERS="$STEPS" \
MICRO_BATCH_SIZE="$MB" \
GLOBAL_BATCH_SIZE=2304 \
EVAL_INTERVAL=50 \
LOG_INTERVAL=10 \
SAVE_INTERVAL="$STEPS" \
MOE_NEW_EXPERT_LR_RAMP_STEPS=900 \
RUN_ID="$RUN_ID" \
TRAIN_WEIGHTS="$OUT" \
MASTER_PORT="${MASTER_PORT:-29882}" \
WANDB_MODE="${WANDB_MODE:-offline}" \
bash "$D/run_g2_ffn_only_code_wiki_joint_old_data_hidden_kl_allrouter_mha.sh"
