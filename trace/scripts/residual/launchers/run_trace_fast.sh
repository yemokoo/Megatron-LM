#!/usr/bin/env bash
# Fast TRACE subset (final order8 scores) on a router-tuned arm.
#   usage: ARM=<arm> GPU=<g> bash run_trace_fast.sh
set -uo pipefail
ARM=${ARM:?}; GPU=${GPU:?}
R=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/residual_router_20260915
TRACE=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace
PY=$TRACE/.venv-runtime/bin/python
OWN=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/tab3_20260913/.gpu_owner
TASKS=${TASKS:-C-STANCE,FOMC,NumGLUE-cm,NumGLUE-ds}
say(){ printf '[RES %s] %s\n' "$(date '+%F %T')" "$*" >> $R/progress.log; }
until grep -q "$ARM MMLU exit" $R/progress.log; do sleep 30; done
echo $$ > $OWN/$GPU
say "$ARM TRACE fast ($TASKS) start gpu$GPU"
export SLORA_LLAMA31_PATH=/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $TRACE/implementations/llmcl_benchmark && CUDA_VISIBLE_DEVICES=$GPU $PY $TRACE/scripts/residual/eval_trace_router_tuned.py \
  --checkpoint_dir $R/$ARM --base_model_name_or_path $SLORA_LLAMA31_PATH \
  --data_path /data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace \
  --inference_tasks $TASKS --inference_output_path $R/$ARM/trace_order8 \
  --max_prompt_len 0 --max_ans_len 1024 --no-task_generation_limits --slora_conv_mode llama3 \
  --per_device_eval_batch_size 16 --temperature 0 > $R/logs/trace_$ARM.log 2>&1
say "$ARM TRACE fast exit=$? $(grep -hoE "\[[A-Za-z0-9-]+\] n=[0-9]+ -> \{[^}]*\}" $R/logs/trace_$ARM.log | tr '\n' ' ')"
[ "$(cat $OWN/$GPU 2>/dev/null)" = "$$" ] && rm -f $OWN/$GPU
