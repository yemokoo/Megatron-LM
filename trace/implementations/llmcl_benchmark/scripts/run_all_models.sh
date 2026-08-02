#!/bin/bash
# Evaluate SEVERAL base models on the 8 TRACE benchmarks with ONE shared env.
#
# You do NOT need a separate Python/vLLM environment per model -- one recent
# vLLM install serves Qwen / Mistral / Llama / Gemma / etc. from the same code.
# What differs per model is only RUNTIME PARAMS (TP size, context len, dtype),
# so we keep those in a per-model table below and loop over it.
#
# Only make a separate conda env / container for a model that has a hard
# dependency conflict (e.g. needs a different vLLM/torch version). In that
# case, activate that env before its line and still call the SAME vllm_eval.py.

set -e
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# This container's shell sets PYTHONPATH to a ~/.local path that SHADOWS the venv's
# own packages (wrong torch/transformers versions get imported otherwise). Unset it.
unset PYTHONPATH

DATA=/home/work/Agent_HJ/30_flame_agent/TRACE/data/LLM-CL-Benchmark_5000
TASKS=C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten

# ---- per-model table -------------------------------------------------------
# columns:  <local_model_path> | <gpu_ids> | <tensor_parallel_size> | <max_model_len>
M=/home/work/Agent_HJ/00_models
MODELS=(
  "$M/Qwen3-0.6B            | 0       | 1 | 4096"
  "$M/Qwen2.5-7B-Instruct   | 0       | 1 | 4096"
  "$M/Qwen3-8B              | 0       | 1 | 4096"
  "$M/llama3-8b-inst        | 0       | 1 | 4096"
  "$M/Qwen2.5-32B-Instruct  | 0,1     | 2 | 4096"
  "$M/Qwen3-32B             | 0,1     | 2 | 4096"
  # 70B: uses all 4 GPUs, slow -- enable when ready
  # "$M/llama3-70b-inst      | 0,1,2,3 | 4 | 4096"
  # "$M/llama3.3_70b_inst    | 0,1,2,3 | 4 | 4096"
)
# ---------------------------------------------------------------------------

for row in "${MODELS[@]}"; do
    IFS='|' read -r MODEL GPUS TP MAXLEN <<< "$row"
    MODEL=$(echo "$MODEL" | xargs); GPUS=$(echo "$GPUS" | xargs)
    TP=$(echo "$TP" | xargs);       MAXLEN=$(echo "$MAXLEN" | xargs)
    NAME=$(basename "$MODEL")
    OUT="./outputs/${NAME}"

    echo "==================================================================="
    echo ">> Evaluating $NAME  (GPUs=$GPUS, TP=$TP, max_len=$MAXLEN)"
    echo "==================================================================="

    CUDA_VISIBLE_DEVICES="$GPUS" python vllm_eval.py \
        --model_name_or_path "$MODEL" \
        --data_path "$DATA" \
        --inference_tasks "$TASKS" \
        --inference_output_path "$OUT" \
        --max_prompt_len 1024 \
        --max_ans_len 512 \
        --temperature 0.0 \
        --tensor_parallel_size "$TP" \
        --max_model_len "$MAXLEN" \
        --gpu_memory_utilization 0.90
done

echo "All done. Per-model summaries under ./outputs/<model>/summary.json"
