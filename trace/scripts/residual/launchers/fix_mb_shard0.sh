#!/usr/bin/env bash
# MeetingBank shard0-of-2 OOM'd at batch 2 (long meetings near the end). Re-run it at
# batch 1 on the arm's first free GPU once its lane script finishes, then merge.
set -uo pipefail
ARM=${ARM:?}; GPU=${GPU:?}
R=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/residual_router_20260915
TRACE=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace
LLMCL=$TRACE/implementations/llmcl_benchmark; PY=$TRACE/.venv-runtime/bin/python
OWN=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/tab3_20260913/.gpu_owner
OUT=$R/$ARM/trace_order8
say(){ printf '[RES %s] %s\n' "$(date '+%F %T')" "$*" >> $R/progress.log; }
until grep -q "$ARM TRACE rest done" $R/progress.log; do sleep 30; done
export SLORA_LLAMA31_PATH=/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo $$ > $OWN/$GPU
mv $R/logs/trace_${ARM}_MeetingBank.shard0-of-2.log $R/logs/trace_${ARM}_MeetingBank.shard0-of-2.oom_b2.log 2>/dev/null
say "$ARM MeetingBank.shard0-of-2 retry batch1 gpu$GPU"
( cd $LLMCL && CUDA_VISIBLE_DEVICES=$GPU $PY $TRACE/scripts/residual/eval_trace_router_tuned.py \
    --checkpoint_dir $R/$ARM --base_model_name_or_path $SLORA_LLAMA31_PATH \
    --data_path /data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace \
    --inference_tasks MeetingBank --inference_output_path $OUT --summary_filename MeetingBank.shard0-of-2.summary.json \
    --max_prompt_len 0 --max_ans_len 1024 --no-task_generation_limits --slora_conv_mode llama3 \
    --per_device_eval_batch_size 1 --temperature 0 --num_sample_shards 2 --sample_shard_id 0 --result_suffix .shard0-of-2 ) \
    > $R/logs/trace_${ARM}_MeetingBank.shard0-of-2.log 2>&1
say "$ARM MeetingBank.shard0-of-2 retry exit=$?"
( cd $LLMCL && $PY -u scripts/merge_trace_shards.py --input_dir $OUT --task MeetingBank --num_shards 2 --summary_filename MeetingBank.summary.json ) > $R/logs/trace_${ARM}_MeetingBank_merge.log 2>&1
say "$ARM MeetingBank merged exit=$? $(tr -d ' \n' < $OUT/MeetingBank.summary.json 2>/dev/null)"
[ "$(cat $OWN/$GPU 2>/dev/null)" = "$$" ] && rm -f $OWN/$GPU
