#!/usr/bin/env bash
# HP-sensitivity knob smoke: prove the two axes actually take effect, 2 GPUs per
# cell, 2 rounds each.
#
# Why these settings: the 1-phase/2-phase joint replay path needs at least one
# replay exposure per optimizer update, so the smoke keeps the production
# global batch (8 x 2 GPU x accum 4 = 64) and the 500-record memory and only
# shortens the epochs.  Round 1 is required -- round 0 alone would not show
# expert *growth* (num_experts = (r+1) * E) or the frozen router prefix.
#
# What it proves that a log line cannot:
#   * experts_per_task / rank / top_k / alpha land in every round's meta and
#     num_experts grows by exactly E per round  (axis 2, never trained before:
#     every V3 TRACE run to date is E=1 / top-1)
#   * dataset_order + stop_after_task record the reversed sequence  (axis 1)
#   * per-round wall clock, so the e8 cell's 64-way expert loop is budgeted
#     before a 4-cell queue is launched
#
#   bash smoke_hp.sh                                   # all four cells
#   CELLS="e8_r8" PAIRS="0,1" bash smoke_hp.sh
set -uo pipefail
TRACE=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace
IMPL=$TRACE/implementations/llmcl_benchmark
PY=$TRACE/.venv-runtime/bin/python
BASE=/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct
DATA=/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace
CACHE=/data2/seonghyeonnoh/LLM-continual-learning-caches/llama31_8b_instruct/slora_chat_full_len1024
ROOT=${ROOT:-/data2/seonghyeonnoh/paper/hp_sens/smoke}
MEM=${MEM:-500}
PHASE=${PHASE:-1phase}
KD=${KD:-on}
# A cell may carry an expert-dispatch override as NAME@MODE; the default
# queue times e8_r8 both ways so the loop-vs-dense speedup is measured on the
# cell where it matters (64 active experts at round 7) before a 30 h queue.
DISPATCH=${DISPATCH:-loop}
KD_FRACTION=${KD_FRACTION:-0.5}
CELLS=${CELLS:-"order_reverse e2_r32 e4_r16 e8_r8 e8_r8@dense"}
PAIRS=${PAIRS:-"0,1 2,3 4,5 6,7"}
mkdir -p "$ROOT"
say(){ printf '[SMOKE-HP %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$ROOT/smoke.log"; }

CANON=(C-STANCE FOMC MeetingBank Py150 ScienceQA NumGLUE-cm NumGLUE-ds 20Minuten)
declare -A EPOCH=( [C-STANCE]=5 [FOMC]=3 [MeetingBank]=7 [Py150]=5
                   [ScienceQA]=3 [NumGLUE-cm]=5 [NumGLUE-ds]=5 [20Minuten]=7 )

cell_spec() {   # name -> "mode E R K"
  case $1 in
    order_reverse) echo "reverse 1 64 1" ;;
    e2_r32)        echo "forward 2 32 2" ;;
    e4_r16)        echo "forward 4 16 4" ;;
    e8_r8)         echo "forward 8 8  8" ;;
    baseline_e1)   echo "forward 1 64 1" ;;
    *) echo "UNKNOWN" ;;
  esac
}

fixture() {   # task dir -- real TRACE rows in the generated-record layout.
              # The generation pipeline is already validated elsewhere; what
              # this smoke needs is only that selfgen replay resolves for the
              # first task of THIS order.
  local task=$1 dir=$2
  local rows=0
  [ -s "$dir/$task/records.jsonl" ] && rows=$(wc -l < "$dir/$task/records.jsonl")
  [ "$rows" -ge "$MEM" ] && return 0
  mkdir -p "$dir/$task"
  $PY - "$DATA/$task/train.json" "$dir/$task/records.jsonl" "$((MEM * 2))" <<'PYEOF'
import json, sys
rows = json.load(open(sys.argv[1]))[-int(sys.argv[3]):]
with open(sys.argv[2], "w") as handle:
    for row in rows:
        handle.write(json.dumps({"prompt": row["prompt"], "answer": row["answer"],
                                 "smoke_fixture": True}, ensure_ascii=False) + "\n")
print(f"fixture rows: {len(rows)}")
PYEOF
  say "fixture $task -> $dir/$task/records.jsonl"
}

run_cell() {   # name[@dispatch] gpus port
  local raw=$1 gpus=$2 port=$3
  local name=${raw%%@*} dispatch=$DISPATCH
  [ "$raw" != "$name" ] && dispatch=${raw#*@}
  local tag=$raw
  local spec; spec=$(cell_spec "$name")
  [ "$spec" = UNKNOWN ] && { say "$name: unknown cell"; return 1; }
  read -r mode E R K <<< "$spec"
  local ALPHA=$((2 * R))
  local out=$ROOT/${tag/@/_} gen=$ROOT/${tag/@/_}/_selfgen_fixture
  local -a TASKS=()
  if [ "$mode" = reverse ]; then
    for ((i=${#CANON[@]}-1; i>=0; i--)); do TASKS+=("${CANON[$i]}"); done
  else
    TASKS=("${CANON[@]}")
  fi
  local dataset_arg epochs_arg=""
  dataset_arg=$(IFS=,; echo "${TASKS[*]}")
  local idx=0
  for t in "${TASKS[@]}"; do
    if [ "$idx" -lt 2 ]; then epochs_arg+="1,"; else epochs_arg+="${EPOCH[$t]},"; fi
    idx=$((idx + 1))
  done
  epochs_arg=${epochs_arg%,}
  mkdir -p "$out"
  fixture "${TASKS[0]}" "$gen" || return 1
  local ngpu; ngpu=$(awk -F, '{print NF}' <<< "$gpus")
  say "$tag start gpu=$gpus order=$mode(${TASKS[0]}->${TASKS[1]}) E=$E rank=$R topk=$K alpha=$ALPHA phase=$PHASE kd=$KD(frac $KD_FRACTION) dispatch=$dispatch"
  local t0=$SECONDS
  ( cd "$IMPL" && env CUDA_VISIBLE_DEVICES="$gpus" \
      SELFGEN_ROOT="$gen" SELFGEN_CURRENT_TASK="${TASKS[1]}" \
      V3_EXPERT_DISPATCH="$dispatch" \
      $PY -m torch.distributed.run --nproc_per_node="$ngpu" --master_port="$port" \
      "$TRACE/scripts/selfgen/train_selfgen.py" \
      --training_version v3_new_replay1to1 --model_name_or_path "$BASE" \
      --data_path "$DATA" --dataset_name "$dataset_arg" --data_output_path "$out/data_cache" \
      --output_dir "$out" --num_train_epochs "$epochs_arg" \
      --per_device_train_batch_size 8 --gradient_accumulation_steps 4 \
      --per_device_eval_batch_size 4 --max_prompt_len 1024 --max_ans_len 512 \
      --max_train_len 1024 --learning_rate 2e-4 --weight_decay 0 \
      --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8 \
      --train_format slora_chat_full --lr_scheduler_type cosine \
      --num_warmup_steps 0 --warmup_ratio 0.03 --gradient_checkpointing \
      --experts_per_task "$E" --lora_moe_rank "$R" --lora_moe_alpha "$ALPHA" \
      --top_k "$K" --lora_moe_dropout 0.05 \
      --routing_weight_mode straight_through_topk \
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
      --disable_training_flop_counter --stop_after_task "${TASKS[1]}" \
      --v3_kd_init_step_fraction "$KD_FRACTION" \
      --ablation_phase_mode "$PHASE" --ablation_kd_init "$KD" \
      --ablation_replay_source selfgen ) > "$out/train.log" 2>&1
  local rc=$? elapsed=$((SECONDS - t0))
  cat > "$out/cell.json" <<JSON
{
  "name": "$tag",
  "order_mode": "$mode",
  "task_order": "$dataset_arg",
  "epochs": "$epochs_arg",
  "experts_per_task": $E,
  "rank": $R,
  "top_k": $K,
  "alpha": $ALPHA,
  "rank_sum_per_task": $((E * R)),
  "phase": "$PHASE",
  "kd_init": "$KD",
  "replay_source": "selfgen",
  "expert_dispatch": "$dispatch",
  "kd_init_step_fraction": $KD_FRACTION,
  "global_batch": $((8 * ngpu * 4)),
  "gpus": "$gpus",
  "smoke_seconds_2rounds": $elapsed
}
JSON
  mkdir -p "$out/model"
  for r in 0 1; do
    [ -e "$out/model/$r" ] || ln -sfn "../$r" "$out/model/$r"
  done
  if [ ! -f "$out/1/lora_moe_meta.json" ]; then
    say "$tag TRAIN FAILED rc=$rc after ${elapsed}s -- tail:"; tail -20 "$out/train.log" | sed 's/^/    /'
    return 1
  fi
  say "$tag train done rc=$rc in ${elapsed}s (2 rounds, 1 epoch each) dispatch=$dispatch"
}

verify_cell() {   # name[@dispatch]
  local name=${1/@/_} out=$ROOT/${1/@/_}
  say "$name verify"
  $PY "$TRACE/scripts/hp_sensitivity/verify_hp_run.py" "$out" --expect-rounds 2 --single-process \
    2>&1 | tee -a "$ROOT/smoke.log"
  local rc1=${PIPESTATUS[0]}
  $PY "$TRACE/scripts/ablation/verify_ablation_run.py" "$out" \
    --expect-phase "$PHASE" --expect-kd "$KD" --expect-replay selfgen \
    --rounds 2 --persistent "$MEM" --log "$out/train.log" \
    2>&1 | tee -a "$ROOT/smoke.log"
  local rc2=${PIPESTATUS[0]}
  say "$name verify hp=$rc1 ablation=$rc2"
  [ "$rc1" = 0 ] && [ "$rc2" = 0 ]
}

read -r -a PAIR_ARR <<< "$PAIRS"
read -r -a CELL_ARR <<< "$CELLS"
say "START cells=[${CELL_ARR[*]}] pairs=[${PAIR_ARR[*]}] root=$ROOT"
pids=(); i=0
for c in "${CELL_ARR[@]}"; do
  pair=${PAIR_ARR[$((i % ${#PAIR_ARR[@]}))]}
  ( run_cell "$c" "$pair" $((29870 + i)) ) &
  pids+=("$!"); i=$((i + 1))
  if (( i % ${#PAIR_ARR[@]} == 0 )); then
    for p in "${pids[@]}"; do wait "$p"; done; pids=()
  fi
done
for p in "${pids[@]}"; do wait "$p"; done
rc=0
for c in "${CELL_ARR[@]}"; do verify_cell "$c" || rc=1; done
say "wall clock (2 rounds, 1 epoch each):"
for c in "${CELL_ARR[@]}"; do
  f=$ROOT/${c/@/_}/cell.json
  [ -f "$f" ] && say "  $($PY -c "
import json;d=json.load(open('$f'))
print(f\"{d['name']:>14}  E={d['experts_per_task']} rank={d['rank']} topk={d['top_k']} dispatch={d['expert_dispatch']:>5}  {d.get('smoke_seconds_2rounds','?')}s\")")"
done
say "SMOKE DONE rc=$rc (log: $ROOT/smoke.log)"
exit $rc
