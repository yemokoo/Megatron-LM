#!/bin/bash
# Track 2: Ours_MoE_FFN (OLMoE native MoE + growing full FFN experts) on TRACE 8 tasks.
# 4-GPU DDP settings measured on A100 80GB with one appended expert per task:
# per-device phase-1 batches = 32,32,28,28,32,32,32,28;
# activation checkpointing = MeetingBank,Py150,ScienceQA,20Minuten;
# phase-2 router retune = per-device 28 with checkpointing always enabled.
#
# Usage:  bash scripts/train_Ours_MoE_FFN.sh [EXPERTS_PER_TASK] [MODEL]
#   e.g.  bash scripts/train_Ours_MoE_FFN.sh 1     # 1 new expert/task
#         bash scripts/train_Ours_MoE_FFN.sh 4     # 4 new experts/task
#         PHASE1_ROUTING=router bash scripts/train_Ours_MoE_FFN.sh 1
set -e
cd "$(dirname "$0")/.."

EXPERTS_PER_TASK=${1:-1}
MODEL=${2:-/home/work/Agent_HJ/00_models/OLMoE-1B-7B-0125}
GPUS=${GPUS:-0,1,2,3}
export CUDA_VISIBLE_DEVICES="$GPUS"
NPROC=$(echo "$GPUS" | tr "," "\n" | grep -c .)
PORT=${PORT:-$((29500 + RANDOM % 1000))}
EPOCHS=${EPOCHS:-2,2,2,2,2,2,2,2}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-auto}
PHASE1_ROUTING=${PHASE1_ROUTING:-force}
PHASE1_BATCHES=${PHASE1_BATCHES:-32,32,28,28,32,32,32,28}
PHASE1_CKPT_TASKS=${PHASE1_CKPT_TASKS:-MeetingBank,Py150,ScienceQA,20Minuten}
ROUTER_BATCH=${ROUTER_BATCH:-28}
GRAD_ACCUM=${GRAD_ACCUM:-1}
PYTHON_BIN=${PYTHON_BIN:-$PWD/.venv-llmcl/bin/python}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
RESUME_PHASE2_CHECKPOINT=${RESUME_PHASE2_CHECKPOINT:-}
START_TASK=${START_TASK:-0}
OUT=${OUT:-output/track2_$(basename "$MODEL")_ept${EXPERTS_PER_TASK}_routing${PHASE1_ROUTING}}

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Track 2 venv Python not found: $PYTHON_BIN" >&2
  exit 1
fi
if [ "$EXPERTS_PER_TASK" != "1" ]; then
  echo "WARNING: default batches were surveyed for EXPERTS_PER_TASK=1; override PHASE1_BATCHES and ROUTER_BATCH." >&2
fi

RESUME_ARGS=()
if [ -n "$RESUME_CHECKPOINT" ]; then
  RESUME_ARGS+=(--resume_checkpoint "$RESUME_CHECKPOINT" --start_task "$START_TASK")
fi
if [ -n "$RESUME_PHASE2_CHECKPOINT" ]; then
  RESUME_ARGS+=(--resume_phase2_checkpoint "$RESUME_PHASE2_CHECKPOINT" --start_task "$START_TASK")
fi

# Avoid the system torchrun entry point: it launches /usr/bin/python and the
# incompatible system Transformers. The venv Python keeps Transformers 4.51.3.
unset BNB_CUDA_VERSION
"$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node="$NPROC" --master_port="$PORT" training/main_Ours_MoE_FFN.py \
  --model_name_or_path "$MODEL" \
  --data_path data/LLM-CL-Benchmark_5000 --dataset_name all \
  --num_train_epochs "$EPOCHS" \
  --max_prompt_len 1536 --max_ans_len 512 \
  --learning_rate 1e-4 --weight_decay 0.0 --num_warmup_steps 0 \
  --per_device_train_batch_size "$PHASE1_BATCHES" \
  --gradient_checkpointing_tasks "$PHASE1_CKPT_TASKS" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --experts_per_task "$EXPERTS_PER_TASK" \
  --phase1_new_expert_routing "$PHASE1_ROUTING" \
  --moe_aux_loss_coeff 0.01 --moe_z_loss_coeff 0.001 \
  --router_retune_epochs 1 --router_retune_batch_size "$ROUTER_BATCH" \
  --past_task_ratio 1.0 \
  --olmoe_replay_path data/OLMOE0125sampling_5000_seed1234/sample_5000.jsonl \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  "${RESUME_ARGS[@]}" \
  --output_dir "$OUT" --seed 1234
