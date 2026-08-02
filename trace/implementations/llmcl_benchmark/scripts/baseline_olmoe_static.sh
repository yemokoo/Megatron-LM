#!/bin/bash
# Baseline (OLMoE): STATIC pre-added experts. Add N full FFN experts ONCE up front,
# freeze the backbone + original 64-way gate, and train ONLY those N experts + their
# router rows sequentially over the TRACE 8 tasks. No task-wise growth, no phase-2
# retune -- this is the control that isolates Ours's growth+phase-2.
#
# Usage:  bash scripts/baseline_olmoe_static.sh [N_EXPERTS] [MODEL]
#   N should equal Ours's final pool = 8 tasks * experts_per_task (default 8).
set -e
cd "$(dirname "$0")/.."

N_EXPERTS=${1:-8}
MODEL=${2:-/home/work/Agent_HJ/00_models/OLMoE-1B-7B-0125}
GPUS=${GPUS:-0,1,2,3}
export CUDA_VISIBLE_DEVICES="$GPUS"
NPROC=$(echo "$GPUS" | tr "," "\n" | grep -c .)
PORT=${PORT:-$((29500 + RANDOM % 1000))}
EPOCHS=${EPOCHS:-2,2,2,2,2,2,2,2}
OUT=output/baseline_static_$(basename "$MODEL")_n${N_EXPERTS}

torchrun --nproc_per_node="$NPROC" --master_port="$PORT" training/main_baseline.py \
  --baseline static_moe_ffn --static_experts "$N_EXPERTS" \
  --model_name_or_path "$MODEL" \
  --data_path data/LLM-CL-Benchmark_5000 --dataset_name all \
  --num_train_epochs "$EPOCHS" \
  --max_prompt_len 1536 --max_ans_len 512 \
  --learning_rate 1e-4 --weight_decay 0.0 --num_warmup_steps 0 \
  --per_device_train_batch_size 3 --gradient_accumulation_steps 1 \
  --moe_aux_loss_coeff 0.01 --moe_z_loss_coeff 0.001 \
  --output_dir "$OUT" --seed 1234
