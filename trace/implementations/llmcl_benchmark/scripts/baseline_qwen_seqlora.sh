#!/bin/bash
# Baseline (Qwen): SeqLoRA. A SINGLE LoRA adapter (1 expert, top-1, no MoE aux/z loss)
# trained sequentially over the TRACE 8 tasks -- the classic single-shared-adapter CL
# baseline. Implemented as the static pool with N=1. Paper uses Qwen3-8B / 5 epochs.
#
# Usage:  bash scripts/baseline_qwen_seqlora.sh [MODEL]
set -e
cd "$(dirname "$0")/.."

MODEL=${1:-/home/work/Agent_HJ/00_models/Qwen3-8B}
GPUS=${GPUS:-0,1,2,3}
export CUDA_VISIBLE_DEVICES="$GPUS"
NPROC=$(echo "$GPUS" | tr "," "\n" | grep -c .)
PORT=${PORT:-$((29500 + RANDOM % 1000))}
EPOCHS=${EPOCHS:-5,5,5,5,5,5,5,5}
OUT=output/baseline_seqlora_$(basename "$MODEL")

torchrun --nproc_per_node="$NPROC" --master_port="$PORT" training/main_baseline.py \
  --baseline static_lora_moe --static_experts 1 \
  --model_name_or_path "$MODEL" \
  --data_path data/LLM-CL-Benchmark_5000 --dataset_name all \
  --num_train_epochs "$EPOCHS" \
  --max_prompt_len 1536 --max_ans_len 512 \
  --learning_rate 1e-4 --weight_decay 0.0 --num_warmup_steps 0 \
  --per_device_train_batch_size 3 --gradient_accumulation_steps 1 \
  --lora_moe_rank 8 --lora_moe_alpha 32 --top_k 1 \
  --moe_aux_loss_coeff 0.0 --moe_z_loss_coeff 0.0 \
  --output_dir "$OUT" --seed 1234
