#!/bin/bash
# Paper-aligned TRACE baselines: seqlora | loramoe | ewc | gem | olora
set -euo pipefail
cd "$(dirname "$0")/.."

METHOD=${1:?usage: train_paper_baseline.sh METHOD [MODEL]}
MODEL=${2:-/home/work/Agent_HJ/00_models/Qwen3-8B}
PYTHON_BIN=${PYTHON_BIN:-/home/work/Agent_HJ/30_flame_agent/envs/train_env/bin/python}
GPUS=${GPUS:-0,1}
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NPROC=$(echo "$GPUS" | tr "," "\n" | grep -c .)
PORT=${PORT:-$((29500 + RANDOM % 1000))}

EPOCHS=${EPOCHS:-2,2,2,2,2,2,2,2}
BATCH=${BATCH:-10,6,8,8,12,18,26,8}
CKPT_TASKS=${CKPT_TASKS:-MeetingBank,Py150,ScienceQA,20Minuten}
RANK=${RANK:-8}
ALPHA=${ALPHA:-32}
LORAMOE_EXPERTS=${LORAMOE_EXPERTS:-8}
OUT=${OUT:-output/paper_${METHOD}_$(basename "$MODEL")_r${RANK}}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}

RESUME_ARGS=()
if [ -n "$RESUME_CHECKPOINT" ]; then
  RESUME_ARGS=(--resume_checkpoint "$RESUME_CHECKPOINT")
fi

unset BNB_CUDA_VERSION
"$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node="$NPROC" --master_port="$PORT" \
  training/main_paper_baselines.py \
  --method "$METHOD" \
  --model_name_or_path "$MODEL" \
  --data_path data/LLM-CL-Benchmark_5000 --dataset_name all \
  --num_train_epochs "$EPOCHS" \
  --max_prompt_len 1536 --max_ans_len 512 \
  --learning_rate 1e-4 --weight_decay "${WEIGHT_DECAY:-0}" --num_warmup_steps 0 \
  --adam_epsilon "${ADAM_EPSILON:-1e-8}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-constant_with_warmup}" \
  --per_device_train_batch_size "$BATCH" --gradient_accumulation_steps 1 \
  --gradient_checkpointing_tasks "$CKPT_TASKS" \
  --lora_rank "$RANK" --lora_alpha "$ALPHA" --lora_dropout 0 \
  --loramoe_num_experts "$LORAMOE_EXPERTS" --top_k 1 \
  --routing_weight_mode full_softmax \
  --moe_aux_loss_coeff "${MOE_AUX_LOSS_COEFF:-0}" \
  --moe_z_loss_coeff "${MOE_Z_LOSS_COEFF:-0}" \
  --ewc_lambda "${EWC_LAMBDA:-400}" \
  --gem_memory_size "${GEM_MEMORY_SIZE:-32}" \
  --gem_memory_batch_size "${GEM_MEMORY_BATCH_SIZE:-4}" \
  --gem_margin "${GEM_MARGIN:-0}" \
  --olora_lambda_orthogonal "${OLORA_LAMBDA_ORTHOGONAL:-0.5}" \
  --olora_lambda_l2 "${OLORA_LAMBDA_L2:-0}" \
  "${RESUME_ARGS[@]}" \
  --output_dir "$OUT" --seed 1234
