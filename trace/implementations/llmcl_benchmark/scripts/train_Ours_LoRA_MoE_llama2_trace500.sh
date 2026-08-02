#!/usr/bin/env bash
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/work/Agent_HJ/30_flame_agent/envs/train_env/bin/python}"
MODEL="${MODEL:-/home/work/Agent_HJ/00_models/Llama-2-7b-chat-hf}"
DATA_PATH="${DATA_PATH:-/home/work/Agent_HJ/30_flame_agent/TreeLoRA/data/LLM-CL-Benchmark/LLM-CL-Benchmark_500}"
OUT="${OUT:-$ROOT_DIR/output/ours_lora_moe_llama2_7b_trace500_r8_ept1_gb32}"
GPUS="${GPUS:-0,1}"
PORT="${PORT:-25407}"
TRAIN_LOG="${TRAIN_LOG:-$ROOT_DIR/logs/ours_lora_moe_llama2_7b_trace500_r8_ept1_gb32.log}"
AGENT_PYTHON="/home/work/Agent_HJ/30_flame_agent/envs/train_env/bin/python"
AGENT_SCRIPT="/home/work/Agent_HJ/30_flame_agent/agent_data_make.py"
AGENT_LOG="$ROOT_DIR/logs/agent_data_make_after_llama2_ours.log"
TASKS="C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten"
EPOCHS="${EPOCHS:-5,3,7,5,3,5,5,7}"
MICRO_BATCH="${MICRO_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then RESUME_ARGS=(--resume_checkpoint "$RESUME_CHECKPOINT"); fi
IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
NUM_GPUS="${#GPU_ARRAY[@]}"
GLOBAL_BATCH=$((NUM_GPUS * MICRO_BATCH * GRAD_ACCUM))
(( GLOBAL_BATCH == 32 )) || echo "Warning: effective global batch is $GLOBAL_BATCH, not 32." >&2
for task in ${TASKS//,/ }; do
  [[ -s "$DATA_PATH/$task/train.json" ]] || { echo "Missing train data for $task" >&2; exit 2; }
done
[[ -x "$PYTHON_BIN" && -s "$MODEL/model.safetensors.index.json" ]] || exit 2
mkdir -p "$OUT" "$(dirname "$TRAIN_LOG")"
cd "$ROOT_DIR"
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "LLaMA-2-7B Ours_LoRA_MoE TRACE-500"
echo "GPUs=$GPUS micro=$MICRO_BATCH accum=$GRAD_ACCUM global=$GLOBAL_BATCH"
echo "epochs=$EPOCHS rank=8 alpha=32 experts_per_task=1 top_k=1"
set +e
env -u PYTHONPATH -u BNB_CUDA_VERSION HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false   "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node="$NUM_GPUS" --master_port="$PORT"   training/main_Ours_LoRA_MoE.py   --model_name_or_path "$MODEL" --data_path "$DATA_PATH" --dataset_name "$TASKS"   --num_train_epochs "$EPOCHS" --max_prompt_len 1024 --max_ans_len 512   --learning_rate 1e-4 --weight_decay 0.0 --adam_epsilon 1e-8 --num_warmup_steps 0 --lr_scheduler_type cosine   --per_device_train_batch_size "$MICRO_BATCH" --gradient_accumulation_steps "$GRAD_ACCUM"   --experts_per_task 1 --lora_moe_rank 8 --lora_moe_alpha 32 --top_k 1 --routing_weight_mode full_softmax   --moe_aux_loss_coeff 0.01 --moe_z_loss_coeff 0.001 --router_retune_epochs 1 --past_task_ratio 1.0   --gradient_checkpointing_tasks MeetingBank,Py150,ScienceQA,20Minuten   "${RESUME_ARGS[@]}" --output_dir "$OUT" --seed 1234 --print_loss 2>&1 | tee "$TRAIN_LOG"
status=${PIPESTATUS[0]}
set -e
start_agent_data_make() {
  local reason="$1"
  if pgrep -f '[a]gent_data_make.py' >/dev/null 2>&1; then
    echo "agent_data_make.py already running; skip ($reason)"; return
  fi
  setsid env -u PYTHONPATH -u BNB_CUDA_VERSION "$AGENT_PYTHON" "$AGENT_SCRIPT"     >> "$AGENT_LOG" 2>&1 < /dev/null &
  echo "agent_data_make.py PID=$! ($reason), log=$AGENT_LOG"
}
if (( status == 0 )); then
  reason="training completed"
elif grep -Eqi 'CUDA out of memory|OutOfMemoryError|torch\.cuda\.OutOfMemoryError|exit code.*-9|return code = -9' "$TRAIN_LOG"; then
  reason="OOM/SIGKILL"
else
  reason="training exited with status $status"
fi
sleep 10
start_agent_data_make "$reason"
exit "$status"
