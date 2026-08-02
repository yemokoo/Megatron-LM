#!/bin/bash
# Baseline (Qwen): LoRAMoE. Pre-expand N LoRA experts ONCE up front (N should equal
# Ours's final pool = 8 * experts_per_task), freeze the backbone, and train the fixed
# expert pool + router sequentially over the TRACE 8 tasks. No growth, no phase-2.
# Paper uses Qwen3-8B / 5 epochs for these baselines.
#
# Usage:  bash scripts/baseline_qwen_loramoe.sh [N_EXPERTS] [MODEL]
set -e
cd "$(dirname "$0")/.."

N_EXPERTS=${1:-8}
MODEL=${2:-/home/work/Agent_HJ/00_models/Qwen3-8B}
GPUS=${GPUS:-0,1,2,3}
export CUDA_VISIBLE_DEVICES="$GPUS"
NPROC=$(echo "$GPUS" | tr "," "\n" | grep -c .)
PORT=${PORT:-$((29500 + RANDOM % 1000))}
EPOCHS=${EPOCHS:-5,5,5,5,5,5,5,5}
OUT=output/baseline_loramoe_$(basename "$MODEL")_n${N_EXPERTS}

torchrun --nproc_per_node="$NPROC" --master_port="$PORT" training/main_baseline.py \
  --baseline static_lora_moe --static_experts "$N_EXPERTS" \
  --model_name_or_path "$MODEL" \
  --data_path data/LLM-CL-Benchmark_5000 --dataset_name all \
  --num_train_epochs "$EPOCHS" \
  --max_prompt_len 1536 --max_ans_len 512 \
  --learning_rate 1e-4 --weight_decay 0.0 --num_warmup_steps 0 \
  --per_device_train_batch_size 3 --gradient_accumulation_steps 1 \
  --lora_moe_rank 8 --lora_moe_alpha 32 --top_k 2 \
  --moe_aux_loss_coeff 0.01 --moe_z_loss_coeff 0.001 \
  --output_dir "$OUT" --seed 1234
