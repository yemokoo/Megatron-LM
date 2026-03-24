#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="${PROJECT_ROOT}/Megatron-LM:${PYTHONPATH:-}"
export WANDB_FINISH_TIMEOUT="${WANDB_FINISH_TIMEOUT:-30}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export TRAIN_ITERS="${TRAIN_ITERS:-10}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
export LOG_INTERVAL="${LOG_INTERVAL:-1}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-10}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-10}"
export SEQ_LENGTH="${SEQ_LENGTH:-512}"

export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-smoke}"
export OLD_MODEL_KL_COEFF="${OLD_MODEL_KL_COEFF:-1.0}"
export OLD_MODEL_KL_TEMPERATURE="${OLD_MODEL_KL_TEMPERATURE:-1.0}"

export WIKI_TRAIN_DATASET="${WIKI_TRAIN_DATASET:-$PROJECT_ROOT/data/wiki/train}"
export WIKI_TEST_DATASET="${WIKI_TEST_DATASET:-$PROJECT_ROOT/data/wiki/test}"
export CODE_TRAIN_DATASET="${CODE_TRAIN_DATASET:-$PROJECT_ROOT/data/code/train}"
export CODE_TEST_DATASET="${CODE_TEST_DATASET:-$PROJECT_ROOT/data/code/test}"

export LORA_WIKI_RUN_ID="${LORA_WIKI_RUN_ID:-smoke-1of4-lora-qv-wiki-pretrain}"
export LORA_CODE_RUN_ID="${LORA_CODE_RUN_ID:-smoke-2of4-lora-qv-code-continual}"
export DENSE_WIKI_RUN_ID="${DENSE_WIKI_RUN_ID:-smoke-3of4-dense-wiki-pretrain}"
export DENSE_CODE_RUN_ID="${DENSE_CODE_RUN_ID:-smoke-4of4-dense-code-continual}"

LORA_WIKI_CKPT="$PROJECT_ROOT/.local/weights/wiki-qv-lora-pretrain-local/$LORA_WIKI_RUN_ID"
DENSE_WIKI_CKPT="$PROJECT_ROOT/.local/weights/wiki-dense-pretrain-local/$DENSE_WIKI_RUN_ID"

echo "========== [1/4 smoke] LoRA-QV Wiki Pretrain =========="
env \
  RUN_ID="$LORA_WIKI_RUN_ID" \
  ATTN_LORA_NUM_EXPERTS="${ATTN_LORA_NUM_EXPERTS:-4}" \
  WANDB_EXP_NAME="${WANDB_EXP_NAME_LORA_WIKI:-[smoke-1-4]_LoRA-QV-Wiki-Pretrain}" \
  TRAIN_DATASET="$WIKI_TRAIN_DATASET" \
  PROBE_DATASET="$WIKI_TEST_DATASET" \
  SECONDARY_PROBE_DATASET="$CODE_TEST_DATASET" \
  bash "$PROJECT_ROOT/scripts/experiment/a100/pretrain_wiki_qv_lora_local_bf16.sh"

echo "========== [2/4 smoke] LoRA-QV Code Continual =========="
env \
  RUN_ID="$LORA_CODE_RUN_ID" \
  ATTN_LORA_SOURCE_NUM_EXPERTS="${ATTN_LORA_SOURCE_NUM_EXPERTS:-4}" \
  ATTN_LORA_NUM_EXPERTS="${ATTN_LORA_TARGET_NUM_EXPERTS:-7}" \
  OLD_MODEL_KL_COEFF="$OLD_MODEL_KL_COEFF" \
  OLD_MODEL_KL_TEMPERATURE="$OLD_MODEL_KL_TEMPERATURE" \
  WANDB_EXP_NAME="${WANDB_EXP_NAME_LORA_CODE:-[smoke-2-4]_LoRA-QV-Code-Continual}" \
  STAGE1_WEIGHTS_DIR="$LORA_WIKI_CKPT" \
  TRAIN_DATASET="$CODE_TRAIN_DATASET" \
  PROBE_DATASET="$CODE_TEST_DATASET" \
  SECONDARY_PROBE_DATASET="$WIKI_TEST_DATASET" \
  bash "$PROJECT_ROOT/scripts/experiment/continual_code_from_wiki_qv_lora_expand_local_bf16.sh"

echo "========== [3/4 smoke] Dense Wiki Pretrain =========="
env \
  RUN_ID="$DENSE_WIKI_RUN_ID" \
  WANDB_EXP_NAME="${WANDB_EXP_NAME_DENSE_WIKI:-[smoke-3-4]_Dense-Wiki-Pretrain}" \
  TRAIN_DATASET="$WIKI_TRAIN_DATASET" \
  PROBE_DATASET="$WIKI_TEST_DATASET" \
  SECONDARY_PROBE_DATASET="$CODE_TEST_DATASET" \
  bash "$PROJECT_ROOT/scripts/experiment/a100/pretrain_wiki_dense_local_bf16.sh"

echo "========== [4/4 smoke] Dense Code Continual =========="
env \
  RUN_ID="$DENSE_CODE_RUN_ID" \
  OLD_MODEL_KL_COEFF="$OLD_MODEL_KL_COEFF" \
  OLD_MODEL_KL_TEMPERATURE="$OLD_MODEL_KL_TEMPERATURE" \
  WANDB_EXP_NAME="${WANDB_EXP_NAME_DENSE_CODE:-[smoke-4-4]_Dense-Code-Continual}" \
  STAGE1_WEIGHTS_DIR="$DENSE_WIKI_CKPT" \
  TRAIN_DATASET="$CODE_TRAIN_DATASET" \
  PROBE_DATASET="$CODE_TEST_DATASET" \
  SECONDARY_PROBE_DATASET="$WIKI_TEST_DATASET" \
  bash "$PROJECT_ROOT/scripts/experiment/continual_code_from_wiki_dense_local_bf16.sh"

echo "========== smoke 4-stage complete =========="
