#!/usr/bin/env bash
# One HP-sensitivity cell for Ours (all switches on: KD-init + self-generated
# replay + 1/2-phase router, residual expert added afterwards by add_residual.sh).
#
# Two knobs vary, nothing else:
#   ORDER_MODE=forward|reverse     TRACE task order (axis 1)
#   EXPERTS_PER_TASK / RANK / TOPK per-task expert granularity at fixed
#                                  rank sum E*RANK = 64 (axis 2)
#
#   ORDER_MODE=reverse                                   bash run_hp_cell.sh
#   EXPERTS_PER_TASK=4 RANK=16 TOPK=4                    bash run_hp_cell.sh
#   EXPERTS_PER_TASK=8 RANK=8  TOPK=8 NAME=e8_r8         bash run_hp_cell.sh
#
# Derived from scripts/ablation/run_arm_gen.sh (same frozen self-generation
# chain, same replay/KD contract).  Differences are only the two knobs, the
# task-name-keyed anchor lookup (index keys break under reordering) and the
# 20Minuten generation profile (in reverse order 20Minuten is round 0, so it
# must generate replay -- in forward order it is last and never does).
#
# Layout:  $ROOT/$NAME/model/<round>              trainer checkpoints
#          $ROOT/$NAME/model/<round>_prephase2    2-phase pre-retune checkpoints
#          $ROOT/$NAME/gen/round_<t>/<task>/records.jsonl
set -uo pipefail
TRACE=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace
IMPL=$TRACE/implementations/llmcl_benchmark
PY=$TRACE/.venv-runtime/bin/python
BASE=/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct
DATA=${TRACE_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace}
CACHE=/data2/seonghyeonnoh/LLM-continual-learning-caches/llama31_8b_instruct/slora_chat_full_len1024

ORDER_MODE=${ORDER_MODE:-forward}
EXPERTS_PER_TASK=${EXPERTS_PER_TASK:-1}
RANK=${RANK:-64}
TOPK=${TOPK:-$EXPERTS_PER_TASK}
ALPHA=${ALPHA:-$((2 * RANK))}      # keep alpha/rank = 2, i.e. the per-expert
                                   # LoRA scale of the 1 x rank-64 baseline
PHASE=${PHASE:-1phase}
KD=${KD:-on}
# Expert dispatch: "loop" (stock) walks the batch-wide union of selected
# experts; "dense" stacks every expert into two rank-space GEMMs so the
# kernel count stops scaling with the expert count.  Numerically equivalent
# (scripts/hp_sensitivity/check_dispatch_equivalence.py); with dropout > 0 the
# LoRA-input dropout mask is drawn per token instead of per (expert, token).
DISPATCH=${DISPATCH:-loop}
# KD-init budget as a fraction of the task's own training steps.  The KD and
# primary streams share samples-per-pass and pass count, so 1.0 means KD-init
# runs exactly as many updates as the task's training (what every TRACE run
# before 2026-09-18 did) and 0.5 stops it after half of them.
KD_FRACTION=${KD_FRACTION:-0.5}
if [ "$ORDER_MODE" = reverse ]; then
  NAME=${NAME:-order_reverse}
else
  NAME=${NAME:-e${EXPERTS_PER_TASK}_r${RANK}}
fi
ROOT=${ROOT:-/data2/seonghyeonnoh/paper/hp_sens}
RUN=$ROOT/$NAME
OUT=$RUN/model; GEN=$RUN/gen; LOG=$RUN/logs
SG=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/selfgen_v3_20260829
[ -f "$SG/anchors.json" ] || SG=$TRACE/scripts/selfgen/assets
GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NGPU=${#GPUS[@]}
PORT=${PORT:-29883}
GEN_SEQS=${GEN_SEQS:-640}
GEN_MIN=${GEN_MIN:-500}
MEM=${MEM:-500}
mkdir -p "$OUT" "$GEN" "$LOG"
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false

[ "$((EXPERTS_PER_TASK * RANK))" = 64 ] || \
  echo "[WARN] rank sum E*RANK = $((EXPERTS_PER_TASK * RANK)) != 64 (baseline)" >&2
[ "$TOPK" -le "$EXPERTS_PER_TASK" ] || { echo "TOPK must be <= EXPERTS_PER_TASK" >&2; exit 2; }

CANON=(C-STANCE FOMC MeetingBank Py150 ScienceQA NumGLUE-cm NumGLUE-ds 20Minuten)
declare -A EPOCH=( [C-STANCE]=5 [FOMC]=3 [MeetingBank]=7 [Py150]=5
                   [ScienceQA]=3 [NumGLUE-cm]=5 [NumGLUE-ds]=5 [20Minuten]=7 )
case $ORDER_MODE in
  forward) TASKS=("${CANON[@]}") ;;
  reverse) TASKS=(); for ((i=${#CANON[@]}-1; i>=0; i--)); do TASKS+=("${CANON[$i]}"); done ;;
  *) echo "ORDER_MODE must be forward|reverse" >&2; exit 2 ;;
esac
DATASET_ARG=$(IFS=,; echo "${TASKS[*]}")
EPOCHS_ARG=""; for t in "${TASKS[@]}"; do EPOCHS_ARG+="${EPOCH[$t]},"; done; EPOCHS_ARG=${EPOCHS_ARG%,}

# Generation profiles.  20Minuten is present here (unlike the ablation runner)
# because it leads the reversed order; its cap comes from the selfgen asset
# caps.json (1024) and it carries no answer cue -- the reference answer starts
# right after the paragraph, with no "Answer:"-style marker in the prompt.
declare -A CAP=( [C-STANCE]=256 [FOMC]=256 [MeetingBank]=1024 [Py150]=1024
                 [ScienceQA]=512 [NumGLUE-cm]=256 [NumGLUE-ds]=256 [20Minuten]=1024 )
declare -A BS=(  [C-STANCE]=256 [FOMC]=256 [MeetingBank]=128 [Py150]=128
                 [ScienceQA]=256 [NumGLUE-cm]=256 [NumGLUE-ds]=256 [20Minuten]=128 )
declare -A CUE=( [C-STANCE]=$'\n态度：' [FOMC]=$'\nStance:' [MeetingBank]=$'\nSummary:'
                 [ScienceQA]=$'\nAnswer:' [NumGLUE-cm]=$'\nAnswer:' [NumGLUE-ds]=$'\nAnswer:' )
declare -A HIST=( [NumGLUE-cm]="$SG/anchor_hist_cm.json" [NumGLUE-ds]="$SG/anchor_hist_ds.json"
                  [Py150]="$SG/anchor_hist_py150.json" )
declare -A PFX=( [NumGLUE-cm]=$'Solve the following math problem.\nQuestion:\n'
                 [NumGLUE-ds]=$'Solve the following math problem.\nQuestion:\n' )
# Target per-chunk size; gen_for_round splits GEN_SEQS into ceil(GEN_SEQS/CHUNK)
# chunks run in parallel across all $NGPU GPUs (each getting exactly one chunk
# at a time from the round-robin pool below).  These used to be tuned for a
# smaller host and left most of an 8-GPU box idle during generation (e.g.
# ScienceQA/C-STANCE/FOMC/NumGLUE-* only ever produced 1 chunk = 1 GPU used).
# 80 gives ceil(640/80)=8 chunks for every task, so a full 8-GPU host is used.
declare -A CHUNK=( [MeetingBank]=80 [Py150]=80 [ScienceQA]=80 [NumGLUE-cm]=80
                   [C-STANCE]=80 [FOMC]=80 [NumGLUE-ds]=80 [20Minuten]=80 )
# Every round generates exactly the task it just trained (gen_for_round filters
# on index == t), so listing all eight is safe.  The 8th entry only matters for
# the optional post-round-7 pass that gives the residual step a replay set for
# the last task -- without it the last task would have no self-generated replay
# and residual tuning would have to borrow another run's records.
ORDER=("${TASKS[@]}")
GEN_FINAL=${GEN_FINAL:-1}

say() { printf '[HP-%s %s] %s\n' "$NAME" "$(date '+%F %T')" "$*" | tee -a "$ROOT/progress.log"; }

anchor_for() {   # task -> anchor text (anchors.json is keyed by CANONICAL index,
                 # so look it up by its "task" field instead)
  $PY - "$SG/anchors.json" "$1" <<'PYEOF'
import json, sys
spec = json.load(open(sys.argv[1]))
for entry in spec.values():
    if entry["task"] == sys.argv[2]:
        print(entry["anchor"], end="")
        break
else:
    raise SystemExit(f"no anchor for {sys.argv[2]}")
PYEOF
}

train_round() {   # t
  local t=$1 task=${TASKS[$t]}
  [ -f "$OUT/$t/lora_moe_meta.json" ] && { say "round $t ($task): skip, checkpoint exists"; return 0; }
  local -a resume=()
  [ "$t" -gt 0 ] && resume=(--resume_checkpoint "$OUT/$((t-1))")
  say "round $t ($task): train  order=$ORDER_MODE experts=$EXPERTS_PER_TASK rank=$RANK topk=$TOPK alpha=$ALPHA phase=$PHASE kd=$KD(frac $KD_FRACTION) dispatch=$DISPATCH"
  ( cd "$IMPL" && env SELFGEN_ROOT="$GEN/round_$t" SELFGEN_CURRENT_TASK="$task" \
      CUDA_VISIBLE_DEVICES="$GPU_LIST" V3_EXPERT_DISPATCH="$DISPATCH" \
      RESUME_CONTRACT_ALLOW_DRIFT=active_stream_samples_per_primary_epoch,joint_replay_active_stream_samples_per_primary_epoch \
      $PY -m torch.distributed.run --nproc_per_node="$NGPU" --master_port="$PORT" \
      "$TRACE/scripts/selfgen/train_selfgen.py" \
      --training_version v3_new_replay1to1 --model_name_or_path "$BASE" \
      --data_path "$DATA" --dataset_name "$DATASET_ARG" --data_output_path "$RUN/data_cache" \
      --output_dir "$OUT" --num_train_epochs "$EPOCHS_ARG" \
      --per_device_train_batch_size 8 --gradient_accumulation_steps 1 \
      --per_device_eval_batch_size 4 --max_prompt_len 1024 --max_ans_len 512 \
      --max_train_len 1024 --learning_rate 2e-4 --weight_decay 0 \
      --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8 \
      --train_format slora_chat_full --lr_scheduler_type cosine \
      --num_warmup_steps 0 --warmup_ratio 0.03 --gradient_checkpointing \
      --experts_per_task "$EXPERTS_PER_TASK" --lora_moe_rank "$RANK" \
      --lora_moe_alpha "$ALPHA" --top_k "$TOPK" \
      --lora_moe_dropout 0.05 --routing_weight_mode straight_through_topk \
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
      --v3_kd_init_step_fraction "$KD_FRACTION" \
      --ablation_phase_mode "$PHASE" --ablation_kd_init "$KD" \
      --ablation_replay_source selfgen \
      "${resume[@]}" ) > "$LOG/train_r$t.log" 2>&1
  local rc=$?
  [ -f "$OUT/$t/lora_moe_meta.json" ] || { say "round $t TRAIN FAILED rc=$rc ($LOG/train_r$t.log)"; return 1; }
  say "round $t ($task): train done"
}

gen_chunk() {   # gpu task task_index chunk_index nseqs next dest ckpt
  local g=$1 task=$2 i=$3 k=$4 per=$5 next=$6 dest=$7 ckpt=$8
  local sd="$dest/$task/stageA.shard$k"
  local part="$dest/$task/records.part$k.jsonl"
  [ -s "$part" ] && return 0
  if [ ! -f "$sd/stats.json" ]; then
    local pfx=${PFX[$task]:-}
    [ -n "$pfx" ] || pfx=$(anchor_for "$task")
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
  [ ${#queue[@]} -eq 0 ] && { say "gen round $next: nothing to do"; return 0; }
  say "gen round $next: ${#queue[@]} chunks over $NGPU GPUs"
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
  [ "$rc" = 0 ] || { say "GEN FAILED (round $next)"; return 1; }
  task=${TASKS[$t]}
  if [ ! -f "$dest/$task/records.jsonl" ]; then
    cat "$dest/$task"/records.part*.jsonl > "$dest/$task/records.jsonl" 2>/dev/null
    n=$(wc -l < "$dest/$task/records.jsonl")
    [ "$n" -ge "$GEN_MIN" ] || { say "GEN FAILED: $task produced $n < $GEN_MIN"; return 1; }
    say "gen round $next / $task: $n records"
  fi
}

mkdir -p "$GEN/round_0"
cat > "$RUN/cell.json" <<JSON
{
  "name": "$NAME",
  "order_mode": "$ORDER_MODE",
  "task_order": "$DATASET_ARG",
  "epochs": "$EPOCHS_ARG",
  "experts_per_task": $EXPERTS_PER_TASK,
  "rank": $RANK,
  "top_k": $TOPK,
  "alpha": $ALPHA,
  "rank_sum_per_task": $((EXPERTS_PER_TASK * RANK)),
  "phase": "$PHASE",
  "kd_init": "$KD",
  "replay_source": "selfgen",
  "expert_dispatch": "$DISPATCH",
  "kd_init_step_fraction": $KD_FRACTION,
  "global_batch": $((8 * NGPU)),
  "gpus": "$GPU_LIST"
}
JSON
if [ "${DRY_RUN:-0}" = 1 ]; then
  # Derivation check without touching a GPU: the order, the per-task epochs that
  # must travel with it, the knobs, and one anchor lookup by task name.
  echo "cell.json:"; cat "$RUN/cell.json"
  echo "--dataset_name      $DATASET_ARG"
  echo "--num_train_epochs  $EPOCHS_ARG"
  echo "round 0 task        ${TASKS[0]}   (generates replay for round 1)"
  echo "round 7 task        ${TASKS[7]}   (last; GEN_FINAL=$GEN_FINAL -> gen/round_8)"
  echo "generating tasks    ${ORDER[*]}"
  echo "anchor(${TASKS[0]})  [$(anchor_for "${TASKS[0]}" | head -c 60)...]"
  echo "cap/bs/chunk        ${CAP[${TASKS[0]}]} / ${BS[${TASKS[0]}]} / ${CHUNK[${TASKS[0]}]}"
  echo "cue(${TASKS[0]})     [${CUE[${TASKS[0]}]:-<none>}]"
  echo "dispatch            $DISPATCH  (V3_EXPERT_DISPATCH)"
  echo "kd step fraction    $KD_FRACTION  (--v3_kd_init_step_fraction)"
  exit 0
fi
say "START order=$ORDER_MODE ($DATASET_ARG) experts=$EXPERTS_PER_TASK rank=$RANK topk=$TOPK alpha=$ALPHA phase=$PHASE kd=$KD gpus=$GPU_LIST global_batch=$((8 * NGPU)) run=$RUN"
for t in 0 1 2 3 4 5 6 7; do
  train_round "$t" || { say "stopping at round $t"; exit 1; }
  if [ "$t" -lt 7 ]; then
    gen_for_round "$t" || { say "stopping: generation for round $((t+1)) failed"; exit 1; }
  fi
done
if [ "$GEN_FINAL" = 1 ]; then
  gen_for_round 7 || { say "stopping: final-task replay generation failed"; exit 1; }
fi
say "all rounds done -> $OUT"
$PY "$TRACE/scripts/hp_sensitivity/verify_hp_run.py" "$RUN" 2>&1 | tee -a "$ROOT/progress.log"
hp_rc=${PIPESTATUS[0]}
$PY "$TRACE/scripts/ablation/verify_ablation_run.py" "$OUT" \
  --expect-phase "$PHASE" --expect-kd "$KD" --expect-replay selfgen \
  --log "$LOG/train_r7.log" 2>&1 | tee -a "$ROOT/progress.log"
say "verify hp=$hp_rc ablation=${PIPESTATUS[0]}"
