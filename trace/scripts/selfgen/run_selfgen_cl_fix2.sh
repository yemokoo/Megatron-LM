#!/usr/bin/env bash
# Self-generated-replay CL over the 8 TRACE tasks -- fixed generation.
#
# Same memory contract as the `lm` baseline (500 records/task, KD 5,000 x task epochs,
# one replay batch per step), so the only variable is real vs generated replay.
# Three fixes over selfgen_cl_20260831 (AA 59.09 / BWT -6.34):
#
#   1. --prompt-cue: stage A reproduced MeetingBank's '\nSummary:' only 31% of the time,
#      and a prompt without the cue made the model emit the cue AS the answer -- 55% of
#      MeetingBank "summaries" were the 8-character string "Summary:".  Measured after
#      the fix: answer median 8 -> 154 chars, top answer share 62.5% -> 1.6%.
#   2. Py150 anchor histogram: its anchor was '<s> ' alone, so generation stopped after
#      ~9 tokens and the task got 23% of the baseline's effective replay tokens.  Top-5
#      real first tokens cover 95.2%.
#   3. Work-queue generation: task cost spans 44x, so a static per-task shard plan left
#      cards idle behind MeetingBank.  Every task is cut into ~20-minute chunks, ordered
#      longest-first, and each GPU pulls the next chunk as it frees.
#
# NOT changed: generation caps.  Training truncates every record at max_train_len=1024,
# so tokens generated past that are never seen (55% of real MeetingBank records already
# lose their answer to that truncation).
set -uo pipefail
TRACE=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace
cd "$TRACE"
PY=$TRACE/.venv-runtime/bin/python
BASE=/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct
DATA=/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace
SG=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/selfgen_v3_20260829
RUN=${RUN:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/selfgen_cl_fix2_20260901}
OUT=$RUN/model; GEN=$RUN/gen; LOG=$RUN/logs
mkdir -p "$OUT" "$GEN" "$LOG"
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NGPU=${#GPUS[@]}
TASKS=(C-STANCE FOMC MeetingBank Py150 ScienceQA NumGLUE-cm NumGLUE-ds 20Minuten)
declare -A CAP=( [C-STANCE]=256 [FOMC]=256 [MeetingBank]=1024 [Py150]=1024
                 [ScienceQA]=512 [NumGLUE-cm]=256 [NumGLUE-ds]=256 )
declare -A BS=(  [C-STANCE]=256 [FOMC]=256 [MeetingBank]=128 [Py150]=128
                 [ScienceQA]=256 [NumGLUE-cm]=256 [NumGLUE-ds]=256 )
# Prompt-ending answer cue; every real prompt of these tasks ends with it (5000/5000).
# Py150 has none -- it is code continuation, not an instruction with an answer slot.
declare -A CUE=( [C-STANCE]=$'\n态度：' [FOMC]=$'\nStance:' [MeetingBank]=$'\nSummary:'
                 [ScienceQA]=$'\nAnswer:' [NumGLUE-cm]=$'\nAnswer:' [NumGLUE-ds]=$'\nAnswer:' )
declare -A HIST=( [NumGLUE-cm]="$SG/anchor_hist_cm.json" [NumGLUE-ds]="$SG/anchor_hist_ds.json"
                  [Py150]="$SG/anchor_hist_py150.json" )
# Distribution guards for stage B (answer_pass_v3_fix2.py).  Priors are the
# real train label distributions; the rest are storage-free rules.
declare -A PRIOR=( [C-STANCE]="A:0.367,B:0.328,C:0.305" [FOMC]="A:0.264,B:0.247,C:0.489"
                   [ScienceQA]="A:0.438,B:0.448,C:0.090,D:0.024" )
declare -A LABELONLY=( [C-STANCE]=1 [FOMC]=1 )
declare -A REJECT=( [MeetingBank]='^(Council|Councilmember|Councilwoman|Councilman|Thank you|Mr\.|Madam|Chair|President)' )
declare -A MINCH=( [MeetingBank]=40 )
declare -A DUPSHARE=( [Py150]=0.03 )
# cm and ds keep their own instruction text; the rest use anchors.json.
declare -A PFX=( [NumGLUE-cm]=$'Solve the following math problem.\nQuestion:\n'
                 [NumGLUE-ds]=$'Solve the following math problem.\nQuestion:\n' )
# ~20 minutes of work per chunk, from round-7 measurements (seqs/min on one card):
# MeetingBank 12.1  Py150 22.1  ScienceQA 42.7  cm 49.2  C-STANCE 160  FOMC/ds 320.
declare -A CHUNK=( [MeetingBank]=240 [Py150]=440 [ScienceQA]=640 [NumGLUE-cm]=640
                   [C-STANCE]=640 [FOMC]=640 [NumGLUE-ds]=640 )
ORDER=(MeetingBank Py150 ScienceQA NumGLUE-cm C-STANCE FOMC NumGLUE-ds)
GEN_SEQS=${GEN_SEQS:-640}     # 28% over the 500-record contract, for dropped sequences
GEN_MIN=${GEN_MIN:-500}
MEM=${MEM:-500}
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
    $PY -m torch.distributed.run --nproc_per_node=$NGPU --master_port=29793 \
    "$TRACE/scripts/selfgen/train_selfgen.py" \
    --training_version v3_new_replay1to1 --model_name_or_path "$BASE" --data_path "$DATA" \
    --dataset_name all --data_output_path "$RUN/data_cache" --output_dir "$OUT" \
    --num_train_epochs 5,3,7,5,3,5,5,7 --per_device_train_batch_size 4 --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 4 --max_prompt_len 1024 --max_ans_len 512 --max_train_len 1024 \
    --learning_rate 2e-4 --weight_decay 0 --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8 \
    --train_format slora_chat_full --lr_scheduler_type cosine --num_warmup_steps 0 --warmup_ratio 0.03 \
    --gradient_checkpointing --experts_per_task 1 --lora_moe_rank 64 --lora_moe_alpha 128 \
    --lora_moe_dropout 0.05 --top_k 1 --routing_weight_mode straight_through_topk \
    --moe_aux_loss_coeff 0.01 --moe_z_loss_coeff 0.001 --seed 2025 \
    --replay_subset_ratio 0.1 --replay_distribution equal_task --replay_recency_power 1.0 \
    --replay_subset_seed 2025 --replay_selection_mode random \
    --router_replay_exposure_samples 5000 --router_retune_epochs 0 \
    --v2_memory_batch_size 0 --v2_replay_forward_batch_size 4 --v2_kd_memory_batch_size 4 \
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
    local pfx=${PFX[$task]:-}
    [ -n "$pfx" ] || pfx=$($PY -c "import json;print(json.load(open('$SG/anchors.json'))['$i']['anchor'],end='')")
    local -a extra=(--prefix-text "$pfx")
    [ -n "${HIST[$task]:-}" ] && extra+=(--prefix-histogram "${HIST[$task]}")
    CUDA_VISIBLE_DEVICES=$g $PY scripts/analysis/bos_sample_v3.py --checkpoint "$ckpt" --mode anchor \
      "${extra[@]}" --num-seqs "$per" --max-seqs 200000 \
      --max-new-tokens "${CAP[$task]}" --batch "${BS[$task]}" --no-routing-probe \
      --seed $((200 + i * 17 + k)) \
      --out-dir "$sd" --label "r${next}_${task}_s$k" > "$LOG/genA_r${next}_${task}_s$k.log" 2>&1 || return 1
  fi
  local -a bopts=()
  [ -n "${CUE[$task]:-}" ] && bopts+=(--prompt-cue "${CUE[$task]}")
  [ -n "${PRIOR[$task]:-}" ] && bopts+=(--label-prior "${PRIOR[$task]}")
  [ -n "${LABELONLY[$task]:-}" ] && bopts+=(--label-only)
  [ -n "${REJECT[$task]:-}" ] && bopts+=(--reject-regex "${REJECT[$task]}")
  [ -n "${MINCH[$task]:-}" ] && bopts+=(--min-answer-chars "${MINCH[$task]}")
  [ -n "${DUPSHARE[$task]:-}" ] && bopts+=(--max-dup-share "${DUPSHARE[$task]}")
  CUDA_VISIBLE_DEVICES=$g $PY scripts/analysis/answer_pass_v3_fix2.py --checkpoint "$ckpt" \
    --stage-a "$sd" --out "$part" "${bopts[@]}" \
    --max-answer-tokens "${CAP[$task]}" --batch 64 > "$LOG/genB_r${next}_${task}_s$k.log" 2>&1 || return 1
}

gen_for_round() {   # produce G for the NEXT round: tasks 0..t generated from model_t
  local t=$1
  local next=$((t+1))
  local dest="$GEN/round_$next"
  local ckpt="$OUT/$t"
  mkdir -p "$dest"
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

  local pool; pool=$(mktemp -u); mkfifo "$pool"; exec 9<>"$pool"; rm -f "$pool"
  local g
  for g in "${GPUS[@]}"; do printf '%s\n' "$g" >&9; done
  local -a pids=()
  local spec
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
