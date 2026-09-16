#!/usr/bin/env bash
# Remaining 4 TRACE tasks (MeetingBank, Py150, ScienceQA, 20Minuten), final order8,
# for one router-tuned arm on two GPUs. Long tasks split into 2 sample shards, then merged.
#   usage: ARM=<arm> GPUS="2 3" bash run_trace_rest.sh
set -uo pipefail
ARM=${ARM:?}; read -r -a G <<< "${GPUS:?}"
R=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/residual_router_20260915
TRACE=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace
LLMCL=$TRACE/implementations/llmcl_benchmark
PY=$TRACE/.venv-runtime/bin/python
OWN=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/tab3_20260913/.gpu_owner
OUT=$R/$ARM/trace_order8
say(){ printf '[RES %s] %s\n' "$(date '+%F %T')" "$*" >> $R/progress.log; }
export SLORA_LLAMA31_PATH=/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# task shards batch
JOBS=("MeetingBank 0 2 2" "MeetingBank 1 2 2" "ScienceQA 0 2 16" "ScienceQA 1 2 16" "Py150 0 1 4" "20Minuten 0 1 8")
run_job(){  # gpu task shard nshards batch
  local g=$1 t=$2 k=$3 n=$4 b=$5 suf="" extra=()
  [ "$n" -gt 1 ] && { suf=".shard$k-of-$n"; extra=(--num_sample_shards $n --sample_shard_id $k --result_suffix $suf); }
  echo $BASHPID > $OWN/$g
  ( cd $LLMCL && CUDA_VISIBLE_DEVICES=$g $PY $TRACE/scripts/residual/eval_trace_router_tuned.py \
      --checkpoint_dir $R/$ARM --base_model_name_or_path $SLORA_LLAMA31_PATH \
      --data_path /data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace \
      --inference_tasks $t --inference_output_path $OUT --summary_filename $t$suf.summary.json \
      --max_prompt_len 0 --max_ans_len 1024 --no-task_generation_limits --slora_conv_mode llama3 \
      --per_device_eval_batch_size $b --temperature 0 "${extra[@]}" ) > $R/logs/trace_${ARM}_$t$suf.log 2>&1
  local rc=$?
  say "$ARM $t$suf gpu$g exit=$rc $(tr -d ' \n' < $OUT/$t$suf.summary.json 2>/dev/null)"
  [ "$(cat $OWN/$g 2>/dev/null)" = "$BASHPID" ] && rm -f $OWN/$g
}
say "$ARM TRACE rest start on GPUs ${G[*]}"
# two lanes, one per GPU, jobs alternate
lane(){ local g=$1; shift; for j in "$@"; do set -- $j; run_job $g "$@"; done; }
lane ${G[0]} "${JOBS[0]}" "${JOBS[2]}" "${JOBS[4]}" &
lane ${G[1]} "${JOBS[1]}" "${JOBS[3]}" "${JOBS[5]}" &
wait
for t in MeetingBank ScienceQA; do
  ( cd $LLMCL && $PY -u scripts/merge_trace_shards.py --input_dir $OUT --task $t --num_shards 2 \
      --summary_filename $t.summary.json ) > $R/logs/trace_${ARM}_${t}_merge.log 2>&1
  say "$ARM $t merged exit=$? $(tr -d ' \n' < $OUT/$t.summary.json 2>/dev/null)"
done
say "$ARM TRACE rest done"
