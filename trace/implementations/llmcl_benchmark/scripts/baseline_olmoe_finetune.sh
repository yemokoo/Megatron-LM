#!/bin/bash
# Baseline (OLMoE): PLAIN FULL FINETUNE -- the whole OLMoE is trainable, trained
# sequentially over the TRACE 8 tasks. No experts, no freezing. Same harness as Ours
# (seq 2048, 2 epochs, lr 1e-4, wd 0.0, 4-GPU batch 12, ZeRO-2).
#
# Usage:  bash scripts/baseline_olmoe_finetune.sh [MODEL]
set -e
cd "$(dirname "$0")/.."

MODEL=${1:-/home/work/Agent_HJ/00_models/OLMoE-1B-7B-0125}
GPUS=${GPUS:-0,1,2,3}
export CUDA_VISIBLE_DEVICES="$GPUS"
NPROC=$(echo "$GPUS" | tr "," "\n" | grep -c .)
PORT=${PORT:-$((29500 + RANDOM % 1000))}
EPOCHS=${EPOCHS:-2,2,2,2,2,2,2,2}
OUT=output/baseline_finetune_$(basename "$MODEL")

# Full-param finetune of a 7B is memory-heavy: gradient checkpointing on.
torchrun --nproc_per_node="$NPROC" --master_port="$PORT" training/main_baseline.py \
  --baseline finetune \
  --model_name_or_path "$MODEL" \
  --data_path data/LLM-CL-Benchmark_5000 --dataset_name all \
  --num_train_epochs "$EPOCHS" \
  --max_prompt_len 1536 --max_ans_len 512 \
  --learning_rate 1e-4 --weight_decay 0.0 --num_warmup_steps 0 \
  --per_device_train_batch_size 3 --gradient_accumulation_steps 1 \
  --output_dir "$OUT" --seed 1234
