#!/usr/bin/env bash
# One ablation arm with SELF-GENERATED replay, 8 GPUs / global batch 64.
#
# Same frozen-generation chain as scripts/selfgen/run_selfgen_cl_frozen.sh (the
# run the final model descends from): each task's replay set is generated once,
# by the checkpoint saved right after that task was trained, and reused verbatim
# in every later round.  The only additions are the ablation switches and the
# paper/ablation output layout.
#
#   PHASE=2phase KD=on  bash run_arm_gen.sh
#   PHASE=1phase KD=off NAME=1phase_nokd_gen GPU_LIST=0,1,2,3,4,5,6,7 bash run_arm_gen.sh
#
# Layout:  $ROOT/$NAME/model/<round>       trainer checkpoints (verify here)
#          $ROOT/$NAME/model/<round>_prephase2   2-phase pre-retune checkpoints
#          $ROOT/$NAME/gen/round_<t>/<task>/records.jsonl   replay used by round t
set -uo pipefail
TRACE=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace
IMPL=$TRACE/implementations/llmcl_benchmark
PY=$TRACE/.venv-runtime/bin/python
BASE=/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct
DATA=${TRACE_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace}
CACHE=/data2/seonghyeonnoh/LLM-continual-learning-caches/llama31_8b_instruct/slora_chat_full_len1024
PHASE=${PHASE:?1phase|2phase}
KD=${KD:?on|off}
NAME=${NAME:-${PHASE}_kd${KD}_gen}
ROOT=${ROOT:-/data2/seonghyeonnoh/paper/ablation}
RUN=$ROOT/$NAME
OUT=$RUN/model; GEN=$RUN/gen; LOG=$RUN/logs
# anchor assets; fall back to the copy committed in scripts/selfgen/assets
SG=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/selfgen_v3_20260829
[ -f "$SG/anchors.json" ] || SG=$TRACE/scripts/selfgen/assets
GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NGPU=${#GPUS[@]}
PORT=${PORT:-29881}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-$TRACE/scripts/selfgen/train_selfgen.py}   # e.g. scripts/residual/train_residual_v3_split.py
GEN_SEQS=${GEN_SEQS:-640}     # 28% over the 500-record contract, for drops
GEN_MIN=${GEN_MIN:-500}
MEM=${MEM:-500}
mkdir -p "$OUT" "$GEN" "$LOG"
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
TASKS=(C-STANCE FOMC MeetingBank Py150 ScienceQA NumGLUE-cm NumGLUE-ds 20Minuten)
declare -A CAP=( [C-STANCE]=256 [FOMC]=256 [MeetingBank]=1024 [Py150]=1024
                 [ScienceQA]=512 [NumGLUE-cm]=256 [NumGLUE-ds]=256 )
declare -A BS=(  [C-STANCE]=256 [FOMC]=256 [MeetingBank]=128 [Py150]=128
                 [ScienceQA]=256 [NumGLUE-cm]=256 [NumGLUE-ds]=256 )
declare -A CUE=( [C-STANCE]=$'\n态度：' [FOMC]=$'\nStance:' [MeetingBank]=$'\nSummary:'
                 [ScienceQA]=$'\nAnswer:' [NumGLUE-cm]=$'\nAnswer:' [NumGLUE-ds]=$'\nAnswer:' )
declare -A HIST=( [NumGLUE-cm]="$SG/anchor_hist_cm.json" [NumGLUE-ds]="$SG/anchor_hist_ds.json"
                  [Py150]="$SG/anchor_hist_py150.json" )
declare -A PFX=( [NumGLUE-cm]=$'Solve the following math problem.\nQuestion:\n'
                 [NumGLUE-ds]=$'Solve the following math problem.\nQuestion:\n' )
# 8 chunks per task so generation spreads over every GPU
declare -A CHUNK=( [MeetingBank]=80 [Py150]=80 [ScienceQA]=80 [NumGLUE-cm]=80
                   [C-STANCE]=80 [FOMC]=80 [NumGLUE-ds]=80 )
ORDER=(MeetingBank Py150 ScienceQA NumGLUE-cm C-STANCE FOMC NumGLUE-ds)
say() { printf '[GEN-ARM %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$ROOT/progress.log"; }

train_round() {   # t
  local t=$1 task=${TASKS[$t]}
  [ -f "$OUT/$t/lora_moe_meta.json" ] && { say "$NAME round $t ($task): skip, checkpoint exists"; return 0; }
  local -a resume=()
  [ "$t" -gt 0 ] && resume=(--resume_checkpoint "$OUT/$((t-1))")
  say "$NAME round $t ($task): train  phase=$PHASE kd=$KD  replay<-$GEN/round_$t"
  ( cd "$IMPL" && env SELFGEN_ROOT="$GEN/round_$t" SELFGEN_CURRENT_TASK="$task" \
      CUDA_VISIBLE_DEVICES="$GPU_LIST" \
      RESUME_CONTRACT_ALLOW_DRIFT=active_stream_samples_per_primary_epoch,joint_replay_active_stream_samples_per_primary_epoch \
      $PY -m torch.distributed.run --nproc_per_node="$NGPU" --master_port="$PORT" \
      "$TRAIN_SCRIPT" \
      --training_version v3_new_replay1to1 --model_name_or_path "$BASE" \
      --data_path "$DATA" --dataset_name all --data_output_path "$RUN/data_cache" \
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
      --v2_new_active_memory_cap 5000 --v2_new_persistent_samples_per_task "$MEM" \
      --v3_epoch_probe_samples 64 --tokenized_train_cache_dir "$CACHE" \
      --disable_training_flop_counter --stop_after_task "$task" \
      --ablation_phase_mode "$PHASE" --ablation_kd_init "$KD" \
      --ablation_replay_source selfgen \
      "${resume[@]}" ) > "$LOG/train_r$t.log" 2>&1
  local rc=$?
  [ -f "$OUT/$t/lora_moe_meta.json" ] || { say "$NAME round $t TRAIN FAILED rc=$rc ($LOG/train_r$t.log)"; return 1; }
  say "$NAME round $t ($task): train done"
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
    CUDA_VISIBLE_DEVICES=$g $PY "$TRACE/scripts/analysis/bos_sample_v3.py" --checkpoint "$ckpt" --mode anchor \
      "${extra[@]}" --num-seqs "$per" --max-seqs 200000 \
      --max-new-tokens "${CAP[$task]}" --batch "${BS[$task]}" --no-routing-probe \
      --seed $((200 + i * 17 + k)) \
      --out-dir "$sd" --label "r${next}_${task}_s$k" > "$LOG/genA_r${next}_${task}_s$k.log" 2>&1 || return 1
  fi
  local -a cue=()
  [ -n "${CUE[$task]:-}" ] && cue=(--prompt-cue "${CUE[$task]}")
  CUDA_VISIBLE_DEVICES=$g $PY "$TRACE/scripts/analysis/answer_pass_v3_fix.py" --checkpoint "$ckpt" \
    --stage-a "$sd" --out "$part" "${cue[@]}" \
    --max-answer-tokens "${CAP[$task]}" --batch 64 > "$LOG/genB_r${next}_${task}_s$k.log" 2>&1 || return 1
}

gen_for_round() {   # t  -> build the replay set the NEXT round consumes
  local t=$1 next=$((t+1)) dest="$GEN/round_$((t+1))" ckpt="$OUT/$t"
  mkdir -p "$dest"
  local j jt src_round
  for j in $(seq 0 $((t-1))); do            # frozen reuse of earlier tasks
    jt=${TASKS[$j]}
    [ -f "$dest/$jt/records.jsonl" ] && continue
    src_round="$GEN/round_$((j+1))/$jt"
    [ -f "$src_round/records.jsonl" ] || { say "FROZEN: missing $src_round/records.jsonl"; return 1; }
    mkdir -p "$dest/$jt"; ln -sf "$src_round/records.jsonl" "$dest/$jt/records.jsonl"
  done
  local -a queue=(); local task i k n per
  for task in "${ORDER[@]}"; do
    for i in $(seq 0 "$t"); do [ "${TASKS[$i]}" = "$task" ] && break; done
    [ "${TASKS[$i]}" = "$task" ] || continue
    [ "$i" -eq "$t" ] || continue           # only the task just trained
    [ -f "$dest/$task/records.jsonl" ] && continue
    per=${CHUNK[$task]}; n=$(( (GEN_SEQS + per - 1) / per )); per=$(( (GEN_SEQS + n - 1) / n ))
    for k in $(seq 0 $((n - 1))); do queue+=("$task $i $k $per"); done
  done
  [ ${#queue[@]} -eq 0 ] && { say "$NAME gen round $next: nothing to do"; return 0; }
  say "$NAME gen round $next: ${#queue[@]} chunks over $NGPU GPUs"
  local pool; pool=$(mktemp -u); mkfifo "$pool"; exec 9<>"$pool"; rm -f "$pool"
  local g; for g in "${GPUS[@]}"; do printf '%s\n' "$g" >&9; done
  local -a pids=(); local spec
  for spec in "${queue[@]}"; do
    read -r -u 9 g
    ( set -uo pipefail
      # shellcheck disable=SC2086
      gen_chunk "$g" $spec "$next" "$dest" "$ckpt"; rc=$?
      printf '%s\n' "$g" >&9; exit $rc ) &
    pids+=($!)
  done
  local rc=0 p; for p in "${pids[@]}"; do wait "$p" || rc=1; done; exec 9>&-
  [ "$rc" = 0 ] || { say "$NAME GEN FAILED (round $next)"; return 1; }
  task=${TASKS[$t]}
  if [ ! -f "$dest/$task/records.jsonl" ]; then
    cat "$dest/$task"/records.part*.jsonl > "$dest/$task/records.jsonl" 2>/dev/null
    n=$(wc -l < "$dest/$task/records.jsonl")
    [ "$n" -ge "$GEN_MIN" ] || { say "$NAME GEN FAILED: $task produced $n < $GEN_MIN"; return 1; }
    say "$NAME gen round $next / $task: $n records"
  fi
}

mkdir -p "$GEN/round_0"
say "$NAME START phase=$PHASE kd=$KD gpus=$GPU_LIST global_batch=$((8 * NGPU)) run=$RUN"
for t in 0 1 2 3 4 5 6 7; do
  train_round "$t" || { say "$NAME stopping at round $t"; exit 1; }
  if [ "$t" -lt 7 ]; then
    gen_for_round "$t" || { say "$NAME stopping: generation for round $((t+1)) failed"; exit 1; }
  fi
done
say "$NAME all rounds done -> $OUT"
$PY "$TRACE/scripts/ablation/verify_ablation_run.py" "$OUT" \
  --expect-phase "$PHASE" --expect-kd "$KD" --expect-replay selfgen \
  --log "$LOG/train_r7.log" 2>&1 | tee -a "$ROOT/progress.log"
say "$NAME verify exit=${PIPESTATUS[0]}"
