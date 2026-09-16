#!/usr/bin/env bash
# Self-generated-replay CL: FULL REGENERATION every round + high-coverage generation
# + rotating exposure window.  Feasibility probe: stop after task 2 (MeetingBank).
#
# This is the setting dawn ran (every past task regenerated each round by the current
# model, nothing stored) but with the generation changes that hicov_rot introduced:
# top-p 1.0, 2600 sequences per task, 2500-record memory, wide unique pool, rotation.
# dawn (640 seqs, top-p 0.95) collapsed to AA 58.32 / BWT -8.21; frozen and hicov_rot
# (both generate once at task time) sit at 62.04 / -2.21 and 61.92 / -2.19.  Question:
# in the regenerating setting, does better generation slow the compounding collapse?
#
# MAX_ROUND limits the chain (default 2 = C-STANCE, FOMC, MeetingBank).
#
# --- original header (hicov_rot / frozen driver) ---
# Self-generated-replay CL, FROZEN generation, HIGH-COVERAGE replay set,
# ROTATING exposure window (train_selfgen_rotate.py).
#
# Fork of run_selfgen_cl_frozen.sh. Question: does raising the replay set's coverage
# of the real task distribution reduce forgetting?  Measured (improved precision/recall
# vs 1000 real train prompts, sets of 500): real subset 75.6% coverage / 88.8% precision,
# frozen's 640-sample first generation 44.5 / 68.0.  Generating 2560 per task with the
# nucleus truncation removed reaches 80.8 / 71.4 for the pool as a whole.
#
# Changes vs the frozen driver, all of them in generation/memory size:
#   --top-p 1.0 (was 0.95)   nucleus truncation was cutting the distribution's tail
#   GEN_SEQS 2560 (was 640)  four times as many sequences per task
#   MEM 2400 (was 500)       the trainer keeps that many as the persistent replay set
# The replay EXPOSURE budget (router_replay_exposure_samples 5000 x epochs) is unchanged,
# so this trades "each record seen many times" for "more distinct records", at equal compute.
# Training GPUs (GPU_LIST) and generation GPUs (GEN_GPU_LIST) are separate so training can
# keep the lm batch contract while generation uses every free card.
#
# --- original header (frozen driver) ---
# Self-generated-replay CL, FROZEN-AT-TASK-TIME generation (diagnostic run).
#
# Same as run_selfgen_cl_fix.sh (dawn run: prompt-cue, Py150 anchor histogram,
# work-queue sharding, greedy answers, NO quota/reject guards) with exactly one
# change: each task's replay set is generated ONCE, by the checkpoint saved right
# after that task was trained, and reused verbatim in every later round.
#
# Why: in the dawn run the same task was regenerated every round by an
# increasingly drifted model.  MeetingBank round 3 matched the real replay
# (len 310 vs 332, transcript openers 0.2%) but round 7 had collapsed
# (len 176, transcript 16.5%), and the two tasks whose generation was "fixed"
# lost the most (MeetingBank -9.59 -> -13.59, Py150 -1.55 -> -5.73).  Freezing
# separates "task-time generation quality" from "regeneration drift":
#   BWT recovers  -> regeneration drift is the cause, task-time quality suffices
#   BWT does not  -> task-time generation is itself insufficient
#
# Implementation: gen/round_{t} for task k links to gen/round_{k+1}/<task k>,
# which was produced by model/k.  Only the newest task is generated each round.
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
RUN=${RUN:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/selfgen_cl_regen_hicov_20260904}
OUT=$RUN/model; GEN=$RUN/gen; LOG=$RUN/logs
mkdir -p "$OUT" "$GEN" "$LOG"
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NGPU=${#GPUS[@]}
GEN_GPU_LIST=${GEN_GPU_LIST:-$GPU_LIST}
IFS=',' read -r -a GEN_GPUS <<< "$GEN_GPU_LIST"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
TASKS=(C-STANCE FOMC MeetingBank Py150 ScienceQA NumGLUE-cm NumGLUE-ds 20Minuten)
declare -A CAP=( [C-STANCE]=256 [FOMC]=256 [MeetingBank]=1024 [Py150]=1024
                 [ScienceQA]=512 [NumGLUE-cm]=256 [NumGLUE-ds]=256 )
declare -A BS=(  [C-STANCE]=384 [FOMC]=384 [MeetingBank]=256 [Py150]=256
                 [ScienceQA]=384 [NumGLUE-cm]=384 [NumGLUE-ds]=384 )
# Prompt-ending answer cue; every real prompt of these tasks ends with it (5000/5000).
# Py150 has none -- it is code continuation, not an instruction with an answer slot.
declare -A CUE=( [C-STANCE]=$'\n态度：' [FOMC]=$'\nStance:' [MeetingBank]=$'\nSummary:'
                 [ScienceQA]=$'\nAnswer:' [NumGLUE-cm]=$'\nAnswer:' [NumGLUE-ds]=$'\nAnswer:' )
declare -A HIST=( [NumGLUE-cm]="$SG/anchor_hist_cm.json" [NumGLUE-ds]="$SG/anchor_hist_ds.json"
                  [Py150]="$SG/anchor_hist_py150.json" )
# cm and ds keep their own instruction text; the rest use anchors.json.
declare -A PFX=( [NumGLUE-cm]=$'Solve the following math problem.\nQuestion:\n'
                 [NumGLUE-ds]=$'Solve the following math problem.\nQuestion:\n' )
# ~20 minutes of work per chunk, from round-7 measurements (seqs/min on one card):
# MeetingBank 12.1  Py150 22.1  ScienceQA 42.7  cm 49.2  C-STANCE 160  FOMC/ds 320.
# GEN_SEQS(2600) 를 GPU 수(8)로 정확히 나눠 한 웨이브에 들어가게: ceil(2600/8)=325.
# 짧은 task 는 청크가 커도 금방 끝나 8장을 다 채울 필요가 없어 650(=4청크)로 둔다.
declare -A CHUNK=( [MeetingBank]=325 [Py150]=325 [ScienceQA]=325 [NumGLUE-cm]=650
                   [C-STANCE]=650 [FOMC]=650 [NumGLUE-ds]=650 )
ORDER=(MeetingBank Py150 ScienceQA NumGLUE-cm C-STANCE FOMC NumGLUE-ds)
GEN_SEQS=${GEN_SEQS:-2600}     # 28% over the 500-record contract, for dropped sequences
GEN_MIN=${GEN_MIN:-2500}
MEM=${MEM:-2500}
say() { printf '[CL %s] %s\n' "$(date '+%F %T')" "$*"; }

train_round() {   # t
  local t=$1
  local task=${TASKS[$t]}
  [ -f "$OUT/$t/lora_moe_meta.json" ] && { say "round $t ($task): skip, checkpoint exists"; return 0; }
  local -a resume=()
  [ "$t" -gt 0 ] && resume=(--resume_checkpoint "$OUT/$((t-1))")
  say "round $t ($task): train  mem=$MEM  replay<-$GEN/round_$t"
  SELFGEN_ROOT="$GEN/round_$t" SELFGEN_CURRENT_TASK="$task" CUDA_VISIBLE_DEVICES=$GPU_LIST \
    RESUME_CONTRACT_ALLOW_DRIFT=${ALLOW_DRIFT:-active_stream_samples_per_primary_epoch,joint_replay_active_stream_samples_per_primary_epoch} \
    $PY -m torch.distributed.run --nproc_per_node=$NGPU --master_port=${PORT:-29808} \
    "$TRACE/scripts/selfgen/train_selfgen_rotate.py" \
    --training_version v3_new_replay1to1 --model_name_or_path "$BASE" --data_path "$DATA" \
    --dataset_name all --data_output_path "$RUN/data_cache" --output_dir "$OUT" \
    --num_train_epochs 5,3,7,5,3,5,5,7 --per_device_train_batch_size ${PDB:-8} --gradient_accumulation_steps ${GA:-1} \
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
    --v2_kd_chunk_tokens ${KD_CHUNK:-1024} --v2_kd_token_scope nonpad --v2_new_active_memory_cap 5000 \
    --v2_new_persistent_samples_per_task "$MEM" --v3_epoch_probe_samples 64 \
    --v2_new_active_unique_cap ${UNIQUE_CAP:-17500} \
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
    # Generation batch is pushed for throughput; the measured allocator overhead is
    # 1.5-2x the KV-cache arithmetic, so retry at half batch (twice) before failing.
    local bs=${BS[$task]} try
    for try in 1 2 3; do
      CUDA_VISIBLE_DEVICES=$g $PY scripts/analysis/bos_sample_v3.py --checkpoint "$ckpt" --mode anchor \
        "${extra[@]}" --num-seqs "$per" --max-seqs 200000 \
        --max-new-tokens "${CAP[$task]}" --batch "$bs" --no-routing-probe \
        --top-p ${TOPP:-1.0} --seed $((200 + i * 17 + k)) \
        --out-dir "$sd" --label "r${next}_${task}_s$k" > "$LOG/genA_r${next}_${task}_s$k.log" 2>&1 && break
      if grep -q "out of memory" "$LOG/genA_r${next}_${task}_s$k.log" 2>/dev/null && [ "$bs" -gt 32 ]; then
        bs=$((bs / 2)); say "genA OOM: $task s$k -> batch $bs 로 재시도"; rm -rf "$sd"; continue
      fi
      return 1
    done
  fi
  local -a cue=()
  [ -n "${CUE[$task]:-}" ] && cue=(--prompt-cue "${CUE[$task]}")
  local abs=${ANS_BATCH:-64} atry
  for atry in 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/analysis/answer_pass_v3_fix.py --checkpoint "$ckpt" \
      --stage-a "$sd" --out "$part" "${cue[@]}" \
      --max-answer-tokens "${CAP[$task]}" --batch "$abs" > "$LOG/genB_r${next}_${task}_s$k.log" 2>&1 && break
    if grep -q "out of memory" "$LOG/genB_r${next}_${task}_s$k.log" 2>/dev/null && [ "$abs" -gt 16 ]; then
      abs=$((abs / 2)); say "genB OOM: $task s$k -> batch $abs 로 재시도"; rm -f "$part"; continue
    fi
    return 1
  done
}

gen_for_round() {   # produce G for the NEXT round: tasks 0..t generated from model_t
  local t=$1
  local next=$((t+1))
  local dest="$GEN/round_$next"
  local ckpt="$OUT/$t"
  mkdir -p "$dest"
  # Every past task is regenerated from the current checkpoint (no reuse).
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
  say "gen round $next: ${#queue[@]} chunks over ${#GEN_GPUS[@]} GPUs (top-p ${TOPP:-1.0}, $GEN_SEQS seqs)"

  local pool; pool=$(mktemp -u); mkfifo "$pool"; exec 9<>"$pool"; rm -f "$pool"
  local g
  for g in "${GEN_GPUS[@]}"; do printf '%s\n' "$g" >&9; done
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
for t in $(seq 0 ${MAX_ROUND:-2}); do
  train_round "$t" || { say "stopping at round $t"; exit 1; }
  if [ "$t" -lt "${MAX_ROUND:-2}" ]; then
    gen_for_round "$t" || { say "stopping: generation for round $((t+1)) failed"; exit 1; }
  fi
done
say "ALL ROUNDS DONE -> $OUT"
