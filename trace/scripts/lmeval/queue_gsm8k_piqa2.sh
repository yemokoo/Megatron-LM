#!/usr/bin/env bash
# GSM8K (5-shot) + PIQA (0-shot): residual gen/rep (router-full) + Table-1 baselines.
# Waits until the old-label pre-review queue has dispatched everything, then takes
# GPUs 2,3,6,7 as they free up (owner file absent and <2GB used).
set -uo pipefail
OUT=/data2/seonghyeonnoh/LLM-continual-learning-runtime/lmeval/tab1_gsm8k_piqa; mkdir -p $OUT
OWN=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/tab3_20260913/.gpu_owner
PY=/data2/seonghyeonnoh/LLM-continual-learning-runtime/lmeval/../lmeval-venv/bin/python
export HF_HOME=/data2/seonghyeonnoh/huggingface HF_DATASETS_CACHE=/data2/seonghyeonnoh/LLM-continual-learning-data/lmeval_datasets HF_DATASETS_OFFLINE=1 HF_DATASETS_TRUST_REMOTE_CODE=1 TOKENIZERS_PARALLELISM=false
JOBS=("gen_res /data2/seonghyeonnoh/LLM-continual-learning-runs/trace/residual_router_20260915/gen_res" "rep_res /data2/seonghyeonnoh/LLM-continual-learning-runs/trace/residual_router_20260915/rep_res" "moe_lpr /data2/seonghyeonnoh/LLM-continual-learning-runs/trace/tab1_sweep_20260912/moelpr_g0.1_mb16/7" "slora_pre /data2/seonghyeonnoh/LLM-continual-learning-runs/trace/slora_pre_released_gb64_20260912/llama31/pre" "ewc /data2/seonghyeonnoh/LLM-continual-learning-runs/trace/tab1_fixed_20260913/ewc/7" "olora /data2/seonghyeonnoh/LLM-continual-learning-runs/trace/tab1_fixed_20260913/olora/7" "seq_lora /data2/seonghyeonnoh/LLM-continual-learning-runs/trace/tab1_fixed_20260913/seq_lora/7" "lifelong_moe /data2/seonghyeonnoh/LLM-continual-learning-runs/trace/tab3_20260913/tab3_lifelong_kd1.5/7" "zero_shot base")
say(){ printf '[G+P %s] %s\n' "$(date '+%F %T')" "$*" >> $OUT/queue.log; }
free_gpu(){ for g in 2 3 6 7; do [ -f $OWN/$g ] && continue; m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '); [ "$m" -lt 2000 ] && { echo $g; return; }; done; }
until grep -q "all dispatched" /data2/seonghyeonnoh/LLM-continual-learning-runs/trace/lpr_prereview_oldlabel_20260916/check.log 2>/dev/null; do sleep 60; done
sleep 120
for j in "${JOBS[@]}"; do
  set -- $j; name=$1; ckpt=$2
  until g=$(free_gpu) && [ -n "$g" ]; do sleep 30; done
  echo "gp$$" > $OWN/$g
  say "$name -> gpu$g"
  ( echo $BASHPID > $OWN/$g
    CUDA_VISIBLE_DEVICES=$g $PY /data2/seonghyeonnoh/LLM-continual-learning-runtime/lmeval/run_lmeval_trace.py --ckpt $ckpt --out $OUT/$name --tasks gsm8k,piqa --batch_size 16 > $OUT/$name.log 2>&1
    say "$name gpu$g exit=$? $(grep -E '^\|(gsm8k|piqa) ' -A1 $OUT/$name.log | tr -s ' ' | tr '\n' ' ' | cut -c1-300)"
    [ "$(cat $OWN/$g 2>/dev/null)" = "$BASHPID" ] && rm -f $OWN/$g ) &
  sleep 60
done
wait
say "all done"
