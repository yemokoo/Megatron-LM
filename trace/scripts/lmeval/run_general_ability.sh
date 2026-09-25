#!/usr/bin/env bash
# General ability (MMLU 0-shot, GSM8K 5-shot, PIQA 0-shot; lm-eval task defaults) of one TRACE
# checkpoint, loaded exactly as the TRACE evaluators load it (run_lmeval_trace.py).
#   GPU=0 CKPT=<final checkpoint dir | base> OUT=<dir> bash scripts/lmeval/run_general_ability.sh
#   GUARD=1   header-guarded residual / mass-reservoir runs: --bos_guard --guard_header
#             --guard_decision none, as in their TRACE sparse15 eval (must match training)
#   env: LMEVAL_VENV, HF_DATASETS_CACHE (see setup_lmeval.sh), SLORA_LLAMA31_PATH (base model),
#        TASKS (default mmlu,gsm8k,piqa), BS (default 16), LIMIT
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV=${LMEVAL_VENV:-/data2/seonghyeonnoh/LLM-continual-learning-runtime/lmeval-venv}
export HF_HOME=${HF_HOME:-/data2/seonghyeonnoh/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/data2/seonghyeonnoh/LLM-continual-learning-data/lmeval_datasets}
export HF_DATASETS_TRUST_REMOTE_CODE=1 HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1} TOKENIZERS_PARALLELISM=false
GPU=${GPU:?}; CKPT=${CKPT:?}; OUT=${OUT:?}
TASKS=${TASKS:-mmlu,gsm8k,piqa}; BS=${BS:-16}
guard=(); [ "${GUARD:-0}" = 1 ] && guard=(--bos_guard --guard_header --guard_decision none)
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=$GPU "$VENV/bin/python" "$HERE/run_lmeval_trace.py" --ckpt "$CKPT" --out "$OUT" \
  --tasks "$TASKS" --batch_size "$BS" ${LIMIT:+--limit $LIMIT} "${guard[@]}" 2>&1 | tee "$OUT/lmeval.log"
