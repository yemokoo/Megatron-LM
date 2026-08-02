#!/bin/bash
# Track 1: Ours_LoRA_MoE (Qwen dense + growing LoRA experts) on TRACE 8 tasks.
# 4-GPU data-parallel (torchrun + DDP).
#
# Per-task batch + per-task gradient checkpointing come from the empirical batch
# survey (scripts/batch_survey.py -> eval_out/batch_survey_8b.json). Measured at the
# WORST-CASE 8 experts (memory grows with expert count each round -- a 1-expert probe
# OOM'd real training at task 2) and a conservative 72GB budget (DDP sees only
# ~79.25GB/rank + NCCL/bucket overhead). So these batches are safe at EVERY round.
# Checkpointing ON tasks are the long/medium ones whose OFF batch collapsed to ~1;
# it's a pure memory/compute tradeoff -- identical weights either way.
#
# IMPORTANT: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MUST match the survey,
# or a batch the survey said fits can OOM (allocator fragmentation).
#
# Usage:  bash scripts/train_Ours_LoRA_MoE.sh [EXPERTS_PER_TASK] [MODEL]
#   e.g.  bash scripts/train_Ours_LoRA_MoE.sh 1 /path/Qwen3-8B     # 8B (edit epochs!)
# Env overrides: GPUS, PORT, EPOCHS, BATCH, CKPT_TASKS, PYTHON_BIN,
#                RESUME_CHECKPOINT
set -e
cd "$(dirname "$0")/.."

EXPERTS_PER_TASK=${1:-4}
MODEL=${2:-/home/work/Agent_HJ/00_models/Qwen3-0.6B}
GPUS=${GPUS:-0,1,2,3}
export CUDA_VISIBLE_DEVICES="$GPUS"
# Avoid caching-allocator fragmentation so surveyed batch sizes transfer 1:1.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NPROC=$(echo "$GPUS" | tr "," "\n" | grep -c .)
PORT=${PORT:-$((29500 + RANDOM % 1000))}
EPOCHS=${EPOCHS:-2,2,2,2,2,2,2,2}      # paper: 2/task for 0.6B, 5/task for 8B

# Per-task train batch, in dataset order: C-STANCE,FOMC,MeetingBank,Py150,
# ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten  (from batch_survey_8b.json).
BATCH=${BATCH:-10,6,8,8,18,18,26,8}
# Tasks whose phase-1 uses gradient checkpointing (long/medium-seq ones).
CKPT_TASKS=${CKPT_TASKS:-MeetingBank,Py150,ScienceQA,20Minuten}

# Default output dir collides with an earlier partial run (rounds 0-3). Override
# with OUT=... to keep those, or delete/rename them first.
OUT=${OUT:-output/track1_$(basename "$MODEL")_ept${EXPERTS_PER_TASK}_fullsoftmax}
PYTHON_BIN=${PYTHON_BIN:-/home/work/Agent_HJ/30_flame_agent/envs/train_env/bin/python}
RESUME_ARGS=()
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
  RESUME_ARGS=(--resume_checkpoint "$RESUME_CHECKPOINT")
fi

"$PYTHON_BIN" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port="$PORT" training/main_Ours_LoRA_MoE.py \
  --model_name_or_path "$MODEL" \
  --data_path data/LLM-CL-Benchmark_5000 --dataset_name all \
  --num_train_epochs "$EPOCHS" \
  --max_prompt_len 1536 --max_ans_len 512 \
  --learning_rate 1e-4 --weight_decay "${WEIGHT_DECAY:-0.0}" --num_warmup_steps 0 \
  --adam_epsilon "${ADAM_EPSILON:-1e-8}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-constant_with_warmup}" \
  --per_device_train_batch_size "$BATCH" --gradient_accumulation_steps 1 \
  --experts_per_task "$EXPERTS_PER_TASK" --lora_moe_rank 8 --lora_moe_alpha 32 --top_k 1 \
  --routing_weight_mode full_softmax \
  --moe_aux_loss_coeff 0.01 --moe_z_loss_coeff 0.001 \
  --router_retune_epochs 1 --past_task_ratio 1.0 \
  --gradient_checkpointing_tasks "$CKPT_TASKS" \
  "${RESUME_ARGS[@]}" \
  --output_dir "$OUT" --seed 1234
