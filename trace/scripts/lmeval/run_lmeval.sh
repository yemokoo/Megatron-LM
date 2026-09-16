#!/usr/bin/env bash
# MMLU / GSM8K / PIQA with lm-evaluation-harness 0.4.8 (offline cache on /data2).
#   usage: GPU=0 MODEL=<HF model dir> OUT=<dir> [PEFT=<adapter dir>] [TASKS=mmlu,gsm8k,piqa] [LIMIT=N] [BS=auto] bash run_lmeval.sh
# defaults: mmlu 5-shot, gsm8k 5-shot (strict/flexible exact match), piqa 0-shot acc/acc_norm — lm-eval task defaults.
set -uo pipefail
R=/data2/seonghyeonnoh/LLM-continual-learning-runtime/lmeval-venv
export HF_HOME=/data2/seonghyeonnoh/huggingface
export HF_DATASETS_CACHE=/data2/seonghyeonnoh/LLM-continual-learning-data/lmeval_datasets
export HF_DATASETS_TRUST_REMOTE_CODE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
GPU=${GPU:?}; MODEL=${MODEL:?}; OUT=${OUT:?}
TASKS=${TASKS:-mmlu,gsm8k,piqa}; BS=${BS:-auto}
ARGS="pretrained=$MODEL,dtype=bfloat16"
[ -n "${PEFT:-}" ] && ARGS="$ARGS,peft=$PEFT"
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=$GPU $R/bin/lm_eval --model hf --model_args "$ARGS" \
  --tasks "$TASKS" --batch_size "$BS" --output_path "$OUT" \
  ${LIMIT:+--limit $LIMIT} ${EXTRA:-} 2>&1 | tee "$OUT/lmeval.log"
