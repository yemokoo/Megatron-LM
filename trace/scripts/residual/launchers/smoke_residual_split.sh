#!/usr/bin/env bash
# Smoke for train_residual_v3_split.py: round 0 (C-STANCE, joint loop with
# backbone BoS from task 0) -> resume -> round 1 (FOMC, self-gen replay fixture
# + BoS + primary reuse) -> evaluate 8 C-STANCE prompts through the residual
# loader.  1 epoch per round, global batch 64 over the given GPUs.
#   GPUS=0,1,2,3,4,5,6,7 bash smoke_residual_split.sh
set -uo pipefail
TRACE=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace
IMPL=$TRACE/implementations/llmcl_benchmark
PY=$TRACE/.venv-runtime/bin/python
BASE=/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct
DATA=/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace
CACHE=/data2/seonghyeonnoh/LLM-continual-learning-caches/llama31_8b_instruct/slora_chat_full_len1024
GEN=/data2/seonghyeonnoh/paper/ablation/smoke/_selfgen_fixture
OUT=${OUT:-/data2/seonghyeonnoh/paper/residual_split/smoke}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NGPU=$(awk -F, '{print NF}' <<< "$GPUS"); ACC=$(( 64 / (8 * NGPU) ))
say(){ printf '[SMOKE-RS %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$OUT/smoke.log"; }
mkdir -p "$OUT"

train(){ # round task extra...
  local t=$1 task=$2; shift 2
  ( cd "$IMPL" && env SELFGEN_ROOT="$GEN" SELFGEN_CURRENT_TASK="$task" CUDA_VISIBLE_DEVICES="$GPUS" \
      RESUME_CONTRACT_ALLOW_DRIFT=active_stream_samples_per_primary_epoch,joint_replay_active_stream_samples_per_primary_epoch \
      $PY -m torch.distributed.run --nproc_per_node="$NGPU" --master_port=29877 \
      "$TRACE/scripts/residual/train_residual_v3_split.py" \
      --training_version v3_new_replay1to1 --model_name_or_path "$BASE" \
      --data_path "$DATA" --dataset_name all --data_output_path "$OUT/data_cache" \
      --output_dir "$OUT" --num_train_epochs 1,1,7,5,3,5,5,7 \
      --per_device_train_batch_size 8 --gradient_accumulation_steps "$ACC" \
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
      --v3_epoch_probe_samples 16 --tokenized_train_cache_dir "$CACHE" \
      --disable_training_flop_counter --stop_after_task "$task" \
      --ablation_phase_mode 1phase --ablation_kd_init off --ablation_replay_source selfgen \
      "$@" ) > "$OUT/train_r$t.log" 2>&1
  local rc=$?
  [ -f "$OUT/$t/lora_moe_meta.json" ] || { say "round $t FAILED rc=$rc"; grep -a "Error\|error:" "$OUT/train_r$t.log" | grep -v Warning | head -5; return 1; }
  say "round $t ($task) done rc=$rc"
}

say "START gpus=$GPUS accum=$ACC out=$OUT"
train 0 C-STANCE || exit 1
train 1 FOMC --resume_checkpoint "$OUT/0" || exit 1
( cd "$IMPL" && CUDA_VISIBLE_DEVICES=${GPUS%%,*} $PY evaluate_Ours_LoRA_MoE.py --checkpoint_dir "$OUT/1" \
    --base_model_name_or_path "$BASE" --data_path "$DATA" --inference_tasks C-STANCE,FOMC --limit 8 \
    --inference_output_path "$OUT/eval" --max_prompt_len 0 --max_ans_len 64 --no-task_generation_limits \
    --slora_conv_mode llama3 --per_device_eval_batch_size 8 --temperature 0 ) > "$OUT/eval.log" 2>&1
say "eval exit=$? ($(grep -a "Loaded\|residual" "$OUT/eval.log" | head -2 | tr '\n' ' '))"
say "SMOKE DONE"
