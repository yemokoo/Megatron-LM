#!/usr/bin/env bash
# KD-init probe: one round's expansion KD-init from a fixed starting checkpoint,
# with the old-task memory drawn from different sources, then score the expanded
# model vs its teacher on REAL held-out data (scripts/selfgen/train_kdprobe.py).
#
#   run_kdprobe.sh <round k> <arm> [SELFGEN_ROOT]
#     arm = real           stock fixed real subset (500/task, seed 2025 = lm memory)
#         = gen            generated records under SELFGEN_ROOT/<task>/records.jsonl
#
# All arms resume from the SAME checkpoint (START_CKPT, default frozen run model/k-1)
# so the only difference is what KD-init sees.  Training config = dawn/frozen runs.
set -uo pipefail
TRACE=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace
cd "$TRACE"
PY=$TRACE/.venv-runtime/bin/python
BASE=/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct
DATA=/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace
TASKS=(C-STANCE FOMC MeetingBank Py150 ScienceQA NumGLUE-cm NumGLUE-ds 20Minuten)
k=${1:?round}; arm=${2:?real|gen}; root=${3:-}
task=${TASKS[$k]}
FROZEN=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/selfgen_cl_frozen_20260901
START_CKPT=${START_CKPT:-$FROZEN/model/$((k-1))}
ROOT=${ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/kdprobe_20260901}
tag="r${k}_${arm}$([ -n "$root" ] && echo "_$(basename "$(dirname "$root")")_$(basename "$root")" || true)"
OUT=$ROOT/$tag; mkdir -p "$OUT/model" "$OUT/logs"
GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}; IFS=',' read -r -a GPUS <<< "$GPU_LIST"; NGPU=${#GPUS[@]}
[ -f "$START_CKPT/lora_moe_meta.json" ] || { echo "no start ckpt $START_CKPT"; exit 1; }
# The trainer resumes the persistent memory identities from ITS OWN output dir
# (fixed_replay_memory/task_k.json).  Seed them: real arm <- lm memory (real indices,
# seed 2025); gen arm <- the frozen run's files (generated identities).
LM_MEM=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/v3_replay1to1/v3_new_replay1to1_st_top1/fixed_replay_memory
MEM_SRC=${MEM_SRC:-$([ "$arm" = real ] && echo "$LM_MEM" || echo "$FROZEN/model/fixed_replay_memory")}
mkdir -p "$OUT/model/fixed_replay_memory"
for j in $(seq 0 $((k-1))); do cp -n "$MEM_SRC/task_${j}_${TASKS[$j]}.json" "$OUT/model/fixed_replay_memory/" || { echo "missing memory $MEM_SRC/task_${j}_${TASKS[$j]}.json"; exit 1; }; done
echo "[kdprobe] seeded memory from $MEM_SRC ($(ls $OUT/model/fixed_replay_memory | wc -l) files)"
[ "$arm" = gen ] && { [ -n "$root" ] || { echo "gen arm needs SELFGEN_ROOT"; exit 1; }; }
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
echo "[kdprobe $(date '+%F %T')] round $k ($task) arm=$arm root=$root start=$START_CKPT -> $OUT"
KDPROBE_SOURCE=$arm SELFGEN_ROOT="$root" SELFGEN_CURRENT_TASK="$task" \
KDPROBE_OUT="$OUT/kdprobe.json" KDPROBE_EVAL_N=${KDPROBE_EVAL_N:-128} \
CUDA_VISIBLE_DEVICES=$GPU_LIST \
RESUME_CONTRACT_ALLOW_DRIFT=active_stream_samples_per_primary_epoch,joint_replay_active_stream_samples_per_primary_epoch \
$PY -m torch.distributed.run --nproc_per_node=$NGPU --master_port=${PORT:-29811} \
  "$TRACE/scripts/selfgen/train_kdprobe.py" \
  --training_version v3_new_replay1to1 --model_name_or_path "$BASE" --data_path "$DATA" \
  --dataset_name all --data_output_path "$FROZEN/data_cache" --output_dir "$OUT/model" \
  --num_train_epochs 5,3,7,5,3,5,5,7 --per_device_train_batch_size 8 --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 4 --max_prompt_len 1024 --max_ans_len 512 --max_train_len 1024 \
  --learning_rate 2e-4 --weight_decay 0 --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8 \
  --train_format slora_chat_full --lr_scheduler_type cosine --num_warmup_steps 0 --warmup_ratio 0.03 \
  --gradient_checkpointing --experts_per_task 1 --lora_moe_rank 64 --lora_moe_alpha 128 \
  --lora_moe_dropout 0.05 --top_k 1 --routing_weight_mode straight_through_topk \
  --moe_aux_loss_coeff 0.01 --moe_z_loss_coeff 0.001 --seed 2025 \
  --replay_subset_ratio 0.1 --replay_distribution equal_task --replay_recency_power 1.0 \
  --replay_subset_seed 2025 --replay_selection_mode random \
  --router_replay_exposure_samples 5000 --router_retune_epochs 0 \
  --v2_memory_batch_size 0 --v2_replay_forward_batch_size 8 --v2_kd_memory_batch_size 8 \
  --v2_max_replay_batches_per_step 0 --v2_joint_replay_loss_coeff 1.0 --v2_joint_replay_objective lm \
  --v2_hidden_mse_loss_coeff 1.0 --v2_joint_new_to_replay_ratio 1 --v2_kd_loss_coeff 1.0 \
  --v2_kd_pass_multiplier 1 --v2_kd_temperature 1.0 --v2_kd_learning_rate 0 \
  --v2_kd_chunk_tokens 256 --v2_kd_token_scope nonpad --v2_new_active_memory_cap 5000 \
  --v2_new_persistent_samples_per_task 500 --v3_epoch_probe_samples 64 \
  --tokenized_train_cache_dir /data2/seonghyeonnoh/LLM-continual-learning-caches/llama31_8b_instruct/slora_chat_full_len1024 \
  --disable_training_flop_counter --stop_after_task "$task" \
  --resume_checkpoint "$START_CKPT" > "$OUT/logs/train.log" 2>&1
rc=$?
if [ -f "$OUT/kdprobe.json" ]; then echo "[kdprobe] DONE -> $OUT/kdprobe.json"; exit 0
else echo "[kdprobe] FAILED rc=$rc (see $OUT/logs/train.log)"; exit 1; fi
