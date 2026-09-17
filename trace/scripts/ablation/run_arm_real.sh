#!/usr/bin/env bash
# One ablation arm with the REAL replay memory, 8 GPUs / global batch 64.
#
# Matches selfgen_cl_frozen_20260901 (the lineage the final model comes from)
# exactly: micro 8 x 8 GPUs x accum 1 = 64, epochs 5,3,7,5,3,5,5,7, rank 64,
# 500-record persistent memory, 5000-sample active stream.  Only the ablation
# switches change between arms.
#
#   PHASE=2phase KD=on  bash run_arm_real.sh
#   PHASE=1phase KD=off NAME=1phase_nokd_rep bash run_arm_real.sh
set -uo pipefail
TRACE=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace
IMPL=$TRACE/implementations/llmcl_benchmark
PY=$TRACE/.venv-runtime/bin/python
BASE=/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct
DATA=/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace
CACHE=/data2/seonghyeonnoh/LLM-continual-learning-caches/llama31_8b_instruct/slora_chat_full_len1024
PHASE=${PHASE:?1phase|2phase}
KD=${KD:?on|off}
NAME=${NAME:-${PHASE}_kd${KD}_rep}
ROOT=${ROOT:-/data2/seonghyeonnoh/paper/ablation}
OUT=$ROOT/$NAME
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
PORT=${PORT:-29871}
NGPU=$(awk -F, '{print NF}' <<< "$GPUS")
mkdir -p "$OUT"
say() { printf '[ARM %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$ROOT/progress.log"; }

if [ -f "$OUT/7/lora_moe_meta.json" ]; then
  say "$NAME already complete -> $OUT"; exit 0
fi
RESUME=()
for t in 6 5 4 3 2 1 0; do
  if [ -f "$OUT/$t/lora_moe_meta.json" ]; then
    RESUME=(--resume_checkpoint "$OUT/$t"); say "$NAME resuming after round $t"; break
  fi
done

say "$NAME start: phase=$PHASE kd=$KD replay=real gpus=$GPUS global_batch=$((8 * NGPU))"
( cd "$IMPL" && env CUDA_VISIBLE_DEVICES="$GPUS" \
    $PY -m torch.distributed.run --nproc_per_node="$NGPU" --master_port="$PORT" \
    training/main_Ours_LoRA_MoE.py \
    --training_version v3_new_replay1to1 --model_name_or_path "$BASE" \
    --data_path "$DATA" --dataset_name all --data_output_path "$OUT/data_cache" \
    --output_dir "$OUT" --num_train_epochs 5,3,7,5,3,5,5,7 \
    --per_device_train_batch_size 8 --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 4 --max_prompt_len 1024 --max_ans_len 512 \
    --max_train_len 1024 --learning_rate 2e-4 --weight_decay 0 \
    --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8 \
    --train_format slora_chat_full --lr_scheduler_type cosine \
    --num_warmup_steps 0 --warmup_ratio 0.03 --gradient_checkpointing \
    --experts_per_task 1 --lora_moe_rank 64 --lora_moe_alpha 128 \
    --lora_moe_dropout 0.05 --top_k 1 --routing_weight_mode straight_through_topk \
    --moe_aux_loss_coeff 0.01 --moe_z_loss_coeff 0.001 --seed 2025 \
    --replay_subset_ratio 0.1 --replay_distribution equal_task \
    --replay_recency_power 1.0 --replay_subset_seed 2025 \
    --replay_selection_mode random --router_replay_exposure_samples 5000 \
    --router_retune_epochs 0 --v2_memory_batch_size 0 \
    --v2_replay_forward_batch_size 8 --v2_kd_memory_batch_size 8 \
    --v2_max_replay_batches_per_step 0 --v2_joint_replay_loss_coeff 1.0 \
    --v2_joint_replay_objective lm --v2_hidden_mse_loss_coeff 1.0 \
    --v2_joint_new_to_replay_ratio 1 --v2_kd_loss_coeff 1.0 \
    --v2_kd_pass_multiplier 1 --v2_kd_temperature 1.0 --v2_kd_learning_rate 0 \
    --v2_kd_chunk_tokens 256 --v2_kd_token_scope nonpad \
    --v2_new_active_memory_cap 5000 --v2_new_persistent_samples_per_task 500 \
    --v3_epoch_probe_samples 64 --tokenized_train_cache_dir "$CACHE" \
    --disable_training_flop_counter \
    --ablation_phase_mode "$PHASE" --ablation_kd_init "$KD" \
    --ablation_replay_source real \
    "${RESUME[@]}" ) >> "$OUT/train.log" 2>&1
rc=$?
[ -f "$OUT/7/lora_moe_meta.json" ] || { say "$NAME TRAIN FAILED rc=$rc ($OUT/train.log)"; exit 1; }
say "$NAME train done"
$PY "$TRACE/scripts/ablation/verify_ablation_run.py" "$OUT" \
  --expect-phase "$PHASE" --expect-kd "$KD" --expect-replay real \
  --log "$OUT/train.log" 2>&1 | tee -a "$ROOT/progress.log"
say "$NAME verify exit=${PIPESTATUS[0]}"
