#!/usr/bin/env bash
# Full MMLU (57 subjects, 5-shot) on Table-1 final checkpoints. One job per GPU, FIFO.
set -uo pipefail
W=/data2/seonghyeonnoh/LLM-continual-learning-runtime/lmeval
PY=/data2/seonghyeonnoh/LLM-continual-learning-runtime/lmeval-venv/bin/python
T=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace
OWN=$T/tab3_20260913/.gpu_owner
export HF_HOME=/data2/seonghyeonnoh/huggingface HF_DATASETS_CACHE=/data2/seonghyeonnoh/LLM-continual-learning-data/lmeval_datasets
export HF_DATASETS_OFFLINE=1 HF_DATASETS_TRUST_REMOTE_CODE=1 TOKENIZERS_PARALLELISM=false
OUT=$W/tab1_mmlu; mkdir -p $OUT
JOBS=(
  "zero_shot base"
  "ours_rep $T/v3_replay1to1/v3_new_replay1to1_st_top1/7"
  "ours_gen $T/selfgen_cl_frozen_20260901/model/7"
  "moe_lpr $T/tab1_sweep_20260912/moelpr_g0.1_mb16/7"
  "lifelong_moe $T/tab3_20260913/tab3_lifelong_kd1.5/7"
  "seq_lora $T/tab1_fixed_20260913/seq_lora/7"
  "ewc $T/tab1_fixed_20260913/ewc/7"
  "olora $T/tab1_fixed_20260913/olora/7"
)
GPUS=(${GPUS:-2 3 4 5 6 7})
say(){ printf '[MMLU %s] %s\n' "$(date '+%F %T')" "$*" >> $OUT/queue.log; }
declare -A busy
next=0
while (( next < ${#JOBS[@]} )) || (( ${#busy[@]} > 0 )); do
  for g in "${GPUS[@]}"; do
    if [[ -n "${busy[$g]:-}" ]] && ! kill -0 "${busy[$g]}" 2>/dev/null; then unset "busy[$g]"; fi
    if [[ -z "${busy[$g]:-}" ]] && (( next < ${#JOBS[@]} )); then
      set -- ${JOBS[$next]}; name=$1; ckpt=$2; next=$((next+1))
      if [[ -f $OUT/$name/results_mmlu.json ]]; then say "$name already done"; continue; fi
      ( echo $BASHPID > $OWN/$g
        CUDA_VISIBLE_DEVICES=$g $PY $W/run_lmeval_trace.py --ckpt $ckpt --out $OUT/$name --tasks mmlu --batch_size 16 > $OUT/$name.log 2>&1
        rc=$?; say "$name gpu$g exit=$rc $(grep -m1 '^|mmlu ' $OUT/$name.log | tr -s ' ')"
        [[ "$(cat $OWN/$g 2>/dev/null)" == "$BASHPID" ]] && rm -f $OWN/$g ) &
      busy[$g]=$!
      say "$name -> gpu$g"
    fi
  done
  sleep 20
done
say "all done"
