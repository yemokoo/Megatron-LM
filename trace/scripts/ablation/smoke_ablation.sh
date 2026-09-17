#!/usr/bin/env bash
# Ablation switch smoke: prove each axis turns on and off, on 2 GPUs per arm.
#
# Two rounds only (C-STANCE -> FOMC, 1 epoch each) at the production global
# batch (64) and 500-record memory -- the 1-phase joint path requires at least
# one replay exposure per optimizer update, so a shrunken memory is not a valid
# smoke. Round 1 still exercises everything the switches touch: KD-init,
# the phase-1 frozen router prefix, the phase-2 router-only retune, and the
# self-generated replay source.  Every arm is checked by
# verify_ablation_run.py, which reads the tensors rather than trusting logs.
#
#   bash smoke_ablation.sh              # all five arms over three GPU pairs
#   PAIRS="0,1 2,3" ARMS="2phase_kd" bash smoke_ablation.sh
set -uo pipefail
TRACE=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace
IMPL=$TRACE/implementations/llmcl_benchmark
PY=$TRACE/.venv-runtime/bin/python
BASE=/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct
DATA=/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace
CACHE=/data2/seonghyeonnoh/LLM-continual-learning-caches/llama31_8b_instruct/slora_chat_full_len1024
ROOT=${ROOT:-/data2/seonghyeonnoh/paper/ablation/smoke}
GEN=$ROOT/_selfgen_fixture
MEM=${MEM:-500}
ARMS=${ARMS:-"1phase_kd 2phase_kd 1phase_nokd 2phase_nokd selfgen_1phase_kd"}
PAIRS=${PAIRS:-"0,1 2,3 4,5"}
mkdir -p "$ROOT" "$GEN"
say() { printf '[SMOKE %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$ROOT/smoke.log"; }

# --- self-generated replay fixture -------------------------------------------
# The generation pipeline itself is unchanged and already validated; what this
# smoke must prove is that --ablation_replay_source selfgen actually reroutes
# replay/KD onto those records.  Real TRACE rows in the generated-record
# layout are enough for that and need no GPU time.
FIXTURE_ROWS=0
[ -s "$GEN/C-STANCE/records.jsonl" ] && FIXTURE_ROWS=$(wc -l < "$GEN/C-STANCE/records.jsonl")
if [ "$FIXTURE_ROWS" -lt "$MEM" ]; then
  mkdir -p "$GEN/C-STANCE"
  $PY - <<PYEOF
import json
rows = json.load(open("$DATA/C-STANCE/train.json"))[-$((MEM * 2)):]
with open("$GEN/C-STANCE/records.jsonl", "w") as handle:
    for row in rows:
        handle.write(json.dumps(
            {"prompt": row["prompt"], "answer": row["answer"],
             "smoke_fixture": True}, ensure_ascii=False) + "\n")
print(f"fixture rows: {len(rows)}")
PYEOF
  say "selfgen fixture written -> $GEN/C-STANCE/records.jsonl"
fi

arm_flags() {   # arm -> ablation flags
  case $1 in
    1phase_kd)          echo "--ablation_phase_mode 1phase --ablation_kd_init on  --ablation_replay_source real" ;;
    2phase_kd)          echo "--ablation_phase_mode 2phase --ablation_kd_init on  --ablation_replay_source real" ;;
    1phase_nokd)        echo "--ablation_phase_mode 1phase --ablation_kd_init off --ablation_replay_source real" ;;
    2phase_nokd)        echo "--ablation_phase_mode 2phase --ablation_kd_init off --ablation_replay_source real" ;;
    selfgen_1phase_kd)  echo "--ablation_phase_mode 1phase --ablation_kd_init on  --ablation_replay_source selfgen" ;;
    *) echo "UNKNOWN" ;;
  esac
}

run_arm() {   # arm gpus port
  local arm=$1 gpus=$2 port=$3
  local out=$ROOT/$arm
  local flags; flags=$(arm_flags "$arm")
  [ "$flags" = "UNKNOWN" ] && { say "$arm: unknown arm"; return 1; }
  local ngpu; ngpu=$(awk -F, '{print NF}' <<< "$gpus")
  local entry=training/main_Ours_LoRA_MoE.py
  local -a env_extra=()
  if [[ $arm == selfgen_* ]]; then
    entry=$TRACE/scripts/selfgen/train_selfgen.py
    env_extra=(SELFGEN_ROOT="$GEN" SELFGEN_CURRENT_TASK=FOMC)
  fi
  mkdir -p "$out"
  say "$arm start on GPU $gpus ($flags)"
  ( cd "$IMPL" && env CUDA_VISIBLE_DEVICES="$gpus" "${env_extra[@]}" \
      $PY -m torch.distributed.run --nproc_per_node="$ngpu" --master_port="$port" \
      "$entry" \
      --training_version v3_new_replay1to1 --model_name_or_path "$BASE" \
      --data_path "$DATA" --dataset_name all --data_output_path "$out/data_cache" \
      --output_dir "$out" --num_train_epochs 1,1,7,5,3,5,5,7 \
      --per_device_train_batch_size 8 --gradient_accumulation_steps 4 \
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
      --v2_new_active_memory_cap 5000 --v2_new_persistent_samples_per_task "$MEM" \
      --v3_epoch_probe_samples 16 --tokenized_train_cache_dir "$CACHE" \
      --disable_training_flop_counter --stop_after_task FOMC \
      $flags ) > "$out/train.log" 2>&1
  local rc=$?
  if [ ! -f "$out/1/lora_moe_meta.json" ]; then
    say "$arm TRAIN FAILED rc=$rc -- tail:"; tail -20 "$out/train.log" | sed 's/^/    /'
    return 1
  fi
  say "$arm train done (rc=$rc)"
}

verify_arm() {   # arm
  local arm=$1 out=$ROOT/$arm
  local phase=1phase kd=on replay=real
  [[ $arm == 2phase_* ]] && phase=2phase
  [[ $arm == *_nokd ]] && kd=off
  [[ $arm == selfgen_* ]] && replay=selfgen
  say "$arm verify (phase=$phase kd=$kd replay=$replay)"
  $PY "$TRACE/scripts/ablation/verify_ablation_run.py" "$out" \
    --expect-phase "$phase" --expect-kd "$kd" --expect-replay "$replay" \
    --rounds 2 --persistent "$MEM" --log "$out/train.log" \
    2>&1 | tee -a "$ROOT/smoke.log"
  return "${PIPESTATUS[0]}"
}

# --- schedule arms over GPU pairs --------------------------------------------
read -r -a PAIR_ARR <<< "$PAIRS"
read -r -a ARM_ARR <<< "$ARMS"
say "START arms=[${ARM_ARR[*]}] pairs=[${PAIR_ARR[*]}] root=$ROOT"
pids=(); i=0
for arm in "${ARM_ARR[@]}"; do
  pair=${PAIR_ARR[$((i % ${#PAIR_ARR[@]}))]}
  port=$((29850 + i))
  ( run_arm "$arm" "$pair" "$port" ) &
  pids+=("$!")
  i=$((i + 1))
  # keep at most one arm per pair in flight
  if (( i % ${#PAIR_ARR[@]} == 0 )); then
    for p in "${pids[@]}"; do wait "$p"; done
    pids=()
  fi
done
for p in "${pids[@]}"; do wait "$p"; done

rc=0
for arm in "${ARM_ARR[@]}"; do verify_arm "$arm" || rc=1; done
say "SMOKE DONE rc=$rc (log: $ROOT/smoke.log)"
exit $rc
