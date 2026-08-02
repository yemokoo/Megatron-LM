#!/bin/bash
# Zero-shot evaluation of an untrained base model on the 8 TRACE benchmarks
# using the vLLM offline backend. No external network at run time
# (except 20Minuten SARI, which is skipped by default).

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0            # add ",1,2,3" and bump TP for big models
# This container's shell sets PYTHONPATH to a ~/.local path that SHADOWS the venv's
# own packages (wrong torch/transformers versions get imported otherwise). Unset it.
unset PYTHONPATH

MODEL=/path/to/local/models/Qwen2.5-7B-Instruct   # <-- local model dir
DATA=/home/work/Agent_HJ/30_flame_agent/TRACE/data/LLM-CL-Benchmark_5000
OUT=./outputs/qwen2.5-7b-instruct

python vllm_eval.py \
    --model_name_or_path "$MODEL" \
    --data_path "$DATA" \
    --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --inference_output_path "$OUT" \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --temperature 0.0 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.90
    # --apply_chat_template   # add for instruct/chat models if you want chat formatting
    # --with_sari             # add only if HF 'sari' metric is cached locally
