#!/usr/bin/env bash
# Self-generated-replay CL, full-size replay memory (task-matched to real train data).
#
# Differences from run_selfgen_cl.sh (which stays untouched):
#   1. Replay memory = 5000 records/task, the SAME size as each task's real train
#      set, instead of 500 (10%).  --v2_new_persistent_samples_per_task moves with it,
#      otherwise the trainer would still slice only the first 500 records.
#   2. Generation is a WORK QUEUE, not a per-task shard plan.  Task cost spans 44x
#      (MeetingBank 12 seq/min vs FOMC 320), so any static split leaves cards idle
#      behind the slowest task.  Every task is cut into ~20-minute chunks, all chunks
#      go into one queue ordered longest-first (LPT), and each GPU pulls the next chunk
#      as it frees up.  Idle time is bounded by one chunk instead of one task.
#
# Training cost is unchanged: KD-init is capped by v2_new_active_memory_cap (5000
# samples/epoch) and 1-phase takes one replay batch per step, so a 10x memory buys
# per-epoch diversity, not more steps.
set -uo pipefail
TRACE=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace
cd "$TRACE"
PY=$TRACE/.venv-runtime/bin/python
BASE=/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct
DATA=/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace
SG=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/selfgen_v3_20260829
# anchor assets (anchors.json, anchor_hist_{cm,ds,py150}.json): fall back to the copy
# committed in scripts/selfgen/assets when the 08-29 probe run is not on this host
[ -f "$SG/anchors.json" ] || SG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/assets"
RUN=${RUN:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/selfgen_cl_full5k_20260831}
OUT=$RUN/model; GEN=$RUN/gen; LOG=$RUN/logs
mkdir -p "$OUT" "$GEN" "$LOG"
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NGPU=${#GPUS[@]}
TASKS=(C-STANCE FOMC MeetingBank Py150 ScienceQA NumGLUE-cm NumGLUE-ds 20Minuten)
declare -A CAP=( [C-STANCE]=256 [FOMC]=256 [MeetingBank]=1024 [Py150]=1024
                 [ScienceQA]=512 [NumGLUE-cm]=256 [NumGLUE-ds]=256 [20Minuten]=1024 )
declare -A BS=(  [C-STANCE]=256 [FOMC]=256 [MeetingBank]=128 [Py150]=128
                 [ScienceQA]=256 [NumGLUE-cm]=256 [NumGLUE-ds]=256 [20Minuten]=128 )
# Chunk = about 20 min of work on one card, from round-7 measurements (seqs/min):
# MeetingBank 12.1  Py150 22.1  ScienceQA 42.7  NumGLUE-cm 49.2  C-STANCE 160
# FOMC 320  NumGLUE-ds 320.  Below ~15 min the 1-2 min model load starts to dominate.
declare -A CHUNK=( [C-STANCE]=3200 [FOMC]=5250 [MeetingBank]=240 [Py150]=440
                   [ScienceQA]=850 [NumGLUE-cm]=980 [NumGLUE-ds]=5250 )
# Longest-first: a short chunk landing last costs at most its own length.
ORDER=(MeetingBank Py150 ScienceQA NumGLUE-cm C-STANCE FOMC NumGLUE-ds)
GEN_SEQS=${GEN_SEQS:-5250}          # 5% over the contract, for dropped sequences
GEN_MIN=${GEN_MIN:-5000}            # what the trainer must find
MEM=${MEM:-5000}                    # v2_new_persistent_samples_per_task
say() { printf '[CL %s] %s\n' "$(date '+%F %T')" "$*"; }

train_round() {   # t
  local t=$1
  local task=${TASKS[$t]}
  [ -f "$OUT/$t/lora_moe_meta.json" ] && { say "round $t ($task): skip, checkpoint exists"; return 0; }
  local -a resume=()
  [ "$t" -gt 0 ] && resume=(--resume_checkpoint "$OUT/$((t-1))")
  say "round $t ($task): train  mem=$MEM  replay<-$GEN/round_$t"
  SELFGEN_ROOT="$GEN/round_$t" SELFGEN_CURRENT_TASK="$task" CUDA_VISIBLE_DEVICES=$GPU_LIST \
    RESUME_CONTRACT_ALLOW_DRIFT=active_stream_samples_per_primary_epoch,joint_replay_active_stream_samples_per_primary_epoch \
    $PY -m torch.distributed.run --nproc_per_node=$NGPU --master_port=29783 \
    "$TRACE/scripts/selfgen/train_selfgen.py" \
    --training_version v3_new_replay1to1 --model_name_or_path "$BASE" --data_path "$DATA" \
    --dataset_name all --data_output_path "$RUN/data_cache" --output_dir "$OUT" \
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
    --v2_new_persistent_samples_per_task "$MEM" --v3_epoch_probe_samples 64 \
    --tokenized_train_cache_dir /data2/seonghyeonnoh/LLM-continual-learning-caches/llama31_8b_instruct/slora_chat_full_len1024 \
    --disable_training_flop_counter --stop_after_task "$task" \
    "${resume[@]}" > "$LOG/train_r$t.log" 2>&1
  local rc=$?
  [ -f "$OUT/$t/lora_moe_meta.json" ] || { say "round $t TRAIN FAILED rc=$rc (see $LOG/train_r$t.log)"; return 1; }
  say "round $t ($task): train done"
}

gen_chunk() {   # gpu task task_index chunk_index nseqs next dest ckpt
  local g=$1 task=$2 i=$3 k=$4 per=$5 next=$6 dest=$7 ckpt=$8
  local sd="$dest/$task/stageA.shard$k"
  local part="$dest/$task/records.part$k.jsonl"
  [ -s "$part" ] && return 0
  if [ ! -f "$sd/stats.json" ]; then
    local a; a=$($PY -c "import json;print(json.load(open('$SG/anchors.json'))['$i']['anchor'])")
    local -a extra=(--prefix-text "$a")
    case "$task" in
      NumGLUE-cm) extra=(--prefix-text "Solve the following math problem.
Question:
" --prefix-histogram "$SG/anchor_hist_cm.json") ;;
      NumGLUE-ds) extra=(--prefix-text "Solve the following math problem.
Question:
" --prefix-histogram "$SG/anchor_hist_ds.json") ;;
    esac
    CUDA_VISIBLE_DEVICES=$g $PY scripts/analysis/bos_sample_v3.py --checkpoint "$ckpt" --mode anchor \
      "${extra[@]}" --num-seqs "$per" --max-seqs 200000 \
      --max-new-tokens "${CAP[$task]}" --batch "${BS[$task]}" --no-routing-probe \
      --seed $((200 + i * 17 + k)) \
      --out-dir "$sd" --label "r${next}_${task}_s$k" > "$LOG/genA_r${next}_${task}_s$k.log" 2>&1 || return 1
  fi
  CUDA_VISIBLE_DEVICES=$g $PY scripts/analysis/answer_pass_v3.py --checkpoint "$ckpt" \
    --stage-a "$sd" --out "$part" \
    --max-answer-tokens "${CAP[$task]}" --batch 64 > "$LOG/genB_r${next}_${task}_s$k.log" 2>&1 || return 1
}

gen_for_round() {   # produce G for the NEXT round: tasks 0..t generated from model_t
  local t=$1
  local next=$((t+1))
  local dest="$GEN/round_$next"
  local ckpt="$OUT/$t"
  mkdir -p "$dest"

  # Build the chunk queue, longest task first.
  local -a queue=()
  local task i k n per
  for task in "${ORDER[@]}"; do
    for i in $(seq 0 "$t"); do [ "${TASKS[$i]}" = "$task" ] && break; done
    [ "${TASKS[$i]}" = "$task" ] || continue
    [ -f "$dest/$task/records.jsonl" ] && continue
    per=${CHUNK[$task]}
    n=$(( (GEN_SEQS + per - 1) / per ))
    per=$(( (GEN_SEQS + n - 1) / n ))
    for k in $(seq 0 $((n - 1))); do queue+=("$task $i $k $per"); done
  done
  if [ ${#queue[@]} -eq 0 ]; then say "gen for round $next: all tasks already generated"; return 0; fi
  say "gen round $next: ${#queue[@]} chunks over $NGPU GPUs"

  # GPU pool as a FIFO: a worker blocks until a card is free, hands it back when done.
  local pool; pool=$(mktemp -u); mkfifo "$pool"; exec 9<>"$pool"; rm -f "$pool"
  for g in "${GPUS[@]}"; do printf '%s\n' "$g" >&9; done
  local -a pids=()
  local spec g
  for spec in "${queue[@]}"; do
    read -r -u 9 g
    ( set -uo pipefail
      # shellcheck disable=SC2086
      gen_chunk "$g" $spec "$next" "$dest" "$ckpt"
      rc=$?
      printf '%s\n' "$g" >&9
      exit $rc ) &
    pids+=($!)
  done
  local rc=0 p
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  exec 9>&-
  [ "$rc" = 0 ] || { say "GEN FAILED (round $next)"; return 1; }

  for i in $(seq 0 "$t"); do
    task=${TASKS[$i]}
    [ -f "$dest/$task/records.jsonl" ] && continue
    cat "$dest/$task"/records.part*.jsonl > "$dest/$task/records.jsonl" 2>/dev/null
    n=$(wc -l < "$dest/$task/records.jsonl")
    if [ "$n" -lt "$GEN_MIN" ]; then
      say "GEN FAILED: $task produced $n records, the trainer needs $GEN_MIN"; return 1
    fi
    say "gen round $next / $task: $n records"
  done
}

mkdir -p "$GEN/round_0"
say "START run=$RUN  mem=$MEM  gen=$GEN_SEQS  gpus=$GPU_LIST"
for t in 0 1 2 3 4 5 6 7; do
  train_round "$t" || { say "stopping at round $t"; exit 1; }
  if [ "$t" -lt 7 ]; then
    gen_for_round "$t" || { say "stopping: generation for round $((t+1)) failed"; exit 1; }
  fi
done
say "ALL ROUNDS DONE -> $OUT"
