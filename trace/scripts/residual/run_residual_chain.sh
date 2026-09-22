#!/usr/bin/env bash
# Residual-expert LoRA-MoE, 1-phase, TRACE 8 tasks -- two replay sources in one script.
#
#   REPLAY_SOURCE=selfgen (default)  every round regenerates the replay for each past task from the
#                                    previous checkpoint: prompt = the fixed chat header, the "\n\n"
#                                    right after it is forced to that task's expert at every layer,
#                                    everything after routes freely (scripts/bos_token/regen_pool.py, one model load per GPU).
#                                    Answers come from a second pass under the same training header
#                                    and guard, with the task expert forced at the answer position.
#                                    1000 samples/task -> label normalisation for the closed-label
#                                    tasks (C-STANCE, FOMC; mapped with the choice list inside the
#                                    generated prompt itself) -> dedupe by prompt -> up to 500 records.
#   REPLAY_SOURCE=real               no generation at all: the trainer's own fixed replay memory
#                                    (500 real training examples per past task).  Ablation baseline.
#
# Training recipe (identical in both modes): residual (skip) expert present from task 0; a new task
# expert starts with LoRA B = 0 and its router row copied from the residual row, so the forward pass
# right after expansion is exactly the pre-expansion one (KD-init without a KD pass); one phase per
# task -- the primary branch masks the residual and trains the new expert plus every real router row,
# while the router-FT branch (replay + backbone-BoS pseudo task + a slice of the current batch) trains
# every router row with all experts frozen.  The chat-template header is routed through the skip
# connection at every layer and carries no loss.
#
# Usage:
#   REPLAY_SOURCE=real    RUN_DIR=/path/out bash scripts/residual/run_residual_chain.sh
#   REPLAY_SOURCE=selfgen RUN_DIR=/path/out bash scripts/residual/run_residual_chain.sh
#   (optional) SEED_CKPT=/path/to/model  reuses an existing round-0 checkpoint instead of training it
set -uo pipefail
TRACE=${TRACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
IMPL=$TRACE/implementations/llmcl_benchmark
PY=${TRACE_PYTHON:-$TRACE/.venv-runtime/bin/python}
BASE=${SLORA_LLAMA31_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}
DATA=${TRACE_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace}
CACHE=${TOKEN_CACHE:-/data2/seonghyeonnoh/LLM-continual-learning-caches/llama31_8b_instruct/slora_chat_full_len1024}
SEED_CKPT=${SEED_CKPT:-}
R=${RUN_DIR:?set RUN_DIR}
REPLAY_SOURCE=${REPLAY_SOURCE:-selfgen}
ANCHORS=$TRACE/scripts/selfgen/assets/anchors.json
DEC=none
GEN_PER_TASK=${GEN_PER_TASK:-1000}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NGPU=$(awk -F, '{print NF}' <<< "$GPUS")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
case "$REPLAY_SOURCE" in selfgen|real) ;; *) echo "REPLAY_SOURCE must be selfgen or real" >&2; exit 2 ;; esac
mkdir -p "$R/model" "$R/gen/round_0" "$R/cond" "$R/logs"
say(){ printf '[CHAIN %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$R/chain.log"; }
TASKS=(C-STANCE FOMC MeetingBank Py150 ScienceQA NumGLUE-cm NumGLUE-ds 20Minuten)
declare -A CAP=( [C-STANCE]=256 [FOMC]=256 [MeetingBank]=1024 [Py150]=1024 [ScienceQA]=512 [NumGLUE-cm]=256 [NumGLUE-ds]=256 )
declare -A CUE=( [C-STANCE]=$'\n态度：' [FOMC]=$'\nStance:' [MeetingBank]=$'\nSummary:' [Py150]='' [ScienceQA]=$'\nAnswer:' [NumGLUE-cm]=$'\nAnswer:' [NumGLUE-ds]=$'\nAnswer:' )

# 0. optional round-0 seed (a checkpoint trained with this same recipe); without it round 0 trains here
if [ -n "$SEED_CKPT" ] && [ ! -f "$R/model/0/lora_moe_meta.json" ]; then
  cp -a "$SEED_CKPT/0" "$R/model/0"
  mkdir -p "$R/model/fixed_replay_memory"
  cp -a "$SEED_CKPT/fixed_replay_memory/." "$R/model/fixed_replay_memory/" 2>/dev/null || true
  $PY - "$SEED_CKPT" "$R/model" <<'SEEDEOF'
import json, sys
src, out = sys.argv[1], sys.argv[2]
w = json.load(open(f"{src}/training_workload.json")); w["tasks"] = [t for t in w["tasks"] if t["round"] == 0]
if isinstance(w.get("totals"), dict):
    for k in list(w["totals"]):
        if isinstance(w["totals"][k], (int, float)): w["totals"][k] = None
json.dump(w, open(f"{out}/training_workload.json", "w"), indent=2)
open(f"{out}/epoch_probe.jsonl", "w").writelines(l for l in open(f"{src}/epoch_probe.jsonl") if json.loads(l)["round"] == 0)
SEEDEOF
  say "model/0 seeded from $SEED_CKPT"
fi

train_round(){   # t
  local t=$1
  local task=${TASKS[$t]} port=$((29970+t))
  [ -f "$R/model/$t/lora_moe_meta.json" ] && return 0
  local -a resume=(); [ "$t" -gt 0 ] && resume=(--resume_checkpoint "$R/model/$((t-1))")
  say "train round $t ($task): header guard ($DEC) + header loss OFF, replay=$REPLAY_SOURCE, GPUs=$GPUS"
  local -a src_env=()
  if [ "$REPLAY_SOURCE" = selfgen ]; then
    src_env=(SELFGEN_ROOT="$R/gen/round_$t" SELFGEN_CURRENT_TASK="$task" SELFGEN_ALLOW_SHORT=1)
  fi
  ( cd "$IMPL" && env "${src_env[@]}" CUDA_VISIBLE_DEVICES=$GPUS \
      BOS_GUARD_HEADER=1 BOS_GUARD_DECISION=$DEC HEADER_NO_LOSS=1 \
      RESUME_CONTRACT_ALLOW_DRIFT=active_stream_samples_per_primary_epoch,joint_replay_active_stream_samples_per_primary_epoch \
      $PY -m torch.distributed.run --nproc_per_node=$NGPU --master_port=$port \
      "$TRACE/scripts/residual/train_residual_v3_split_bosguard.py" \
      --training_version v3_new_replay1to1 --model_name_or_path "$BASE" \
      --data_path "$DATA" --dataset_name all --data_output_path "$R/data_cache" \
      --output_dir "$R/model" --num_train_epochs 5,3,7,5,3,5,5,7 \
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
      --disable_training_flop_counter --stop_after_task "$task" \
      --ablation_phase_mode 1phase --ablation_kd_init off --ablation_replay_source "$REPLAY_SOURCE" \
      "${resume[@]}" ) > "$R/logs/train_r$t.log" 2>&1
  [ -f "$R/model/$t/lora_moe_meta.json" ] || { say "EVENT: STEP_FAILED train_r$t"; exit 1; }
  say "round $t ($task) trained"
}

force_file(){   # j n_experts -> path of the (32 x n+1) all-layer tensor forcing E_j (others -inf), applied at "\n\n" only
  local j=$1 n=$2
  local f=$R/cond/force_E${j}_of${n}.pt
  [ -f "$f" ] || $PY - "$j" "$n" "$f" <<'PYEOF'
import sys, torch
j, n, f = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
b = torch.full((32, n + 1), float("-inf")); b[:, j] = 0.0
torch.save({"bias": b, "n_layers": 32, "n_slots": n + 1, "force_expert": j}, f)
PYEOF
  echo "$f"
}

gen_round(){   # t : regenerate tasks 0..t-1 from model/(t-1) into gen/round_t, 8 GPUs x (task chunks)
  local t=$1
  local ckpt=$R/model/$((t-1)) dest=$R/gen/round_$t
  local nexp; nexp=$($PY -c "import json;print(json.load(open('$ckpt/lora_moe_meta.json'))['num_experts'])")
  local per=$((GEN_PER_TASK / 8)) j task k
  local pending=0
  for j in $(seq 0 $((t-1))); do [ -f "$dest/${TASKS[$j]}/records.jsonl" ] || pending=1; done
  [ "$pending" = 1 ] || return 0
  say "regen round $t: tasks 0..$((t-1)) from model/$((t-1)) (forced \\n\\n routing, $GEN_PER_TASK docs/task, GPUs=$GPUS)"
  mkdir -p "$dest"
  # work-queue pool: one process per GPU x PROCS_PER_GPU pull (task, shard) jobs, longest tasks first, so a GPU that
  # finishes a short shard immediately takes the next job instead of waiting for the long ones
  local rc=0
  $PY "$TRACE/scripts/bos_token/regen_pool.py" --checkpoint "$ckpt" --dest "$dest" --round "$t" \
      --num-tasks "$t" --gpus "$GPUS" --per-task "$GEN_PER_TASK" --shards "${GEN_SHARDS:-8}" --long-shards "${GEN_LONG_SHARDS:-32}" --procs-per-gpu "${PROCS_PER_GPU:-2}" \
      --overrides "$R/gen/regen_overrides.json" --guard-decision $DEC > "$dest/regen_pool.log" 2>&1 || rc=1
  [ "$rc" = 0 ] || { say "EVENT: STEP_FAILED regen round $t"; exit 1; }
  for j in $(seq 0 $((t-1))); do
    task=${TASKS[$j]}
    [ -f "$dest/$task/records.jsonl" ] && continue
    cat "$dest/$task"/records.part*.jsonl > "$dest/$task/records.all.jsonl"
    $PY - "$dest/$task" "$j" "$ANCHORS" "$GEN_PER_TASK" <<'PYEOF' > "$dest/$task/select.log" 2>&1
import json, re, sys, glob, random
d, j, anchors_path, req = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
anchor = json.load(open(anchors_path))[j]["anchor"]
strict = {"0": r"\n?(.+?)\n对象：\n(.+?)\n态度：$", "1": r"\n?(.+?)\nStance:$"}.get(j)
pat = re.compile(r"^" + re.escape(anchor.rstrip("\n")) + strict, re.S) if strict else None
rows = [json.loads(l) for l in open(f"{d}/records.all.jsonl")]

def normalize_label(prompt, answer):
    """Closed-label tasks (C-STANCE, FOMC) only: map a word-form answer back to its letter using the
    choice list that is already present in the generated prompt (e.g. "A. dovish, B. hawkish,
    C. neutral"), so no external label knowledge is used.  Returns None when it cannot be mapped."""
    a = answer.strip().rstrip(".").strip()
    if len(a) == 1 and a.isalpha():
        return a.upper() if a.upper() in ("A", "B", "C") else None
    for letter, text in re.findall(r"([A-C])\s*[.．]\s*([^,，.。\n]+)", prompt):
        if a.lower() == text.strip().lower():
            return letter
    return None

seen, keep, dropped = set(), [], 0       # dedupe; closed-label tasks also get answer normalisation
for r in rows:
    if j in ("0", "1"):
        lab = normalize_label(r["prompt"], r["answer"])
        if lab is None:
            dropped += 1; continue
        r = {**r, "answer": lab}
    if r["prompt"] in seen: continue
    seen.add(r["prompt"]); keep.append(r)
docs = [json.loads(l) for f in sorted(glob.glob(f"{d}/stageA.s*/docs.jsonl")) for l in open(f)]
st = {"requested": req, "docs": len(docs), "starts_with_anchor": sum(1 for x in docs if x["starts_with_anchor"]),
      "stageB": len(rows), "unique": len(keep), "label_dropped": dropped, "yield_unique": round(len(keep) / req, 4)}
random.Random(0).shuffle(keep); keep = keep[:500]
with open(f"{d}/records.jsonl", "w") as fh:
    for r in keep: fh.write(json.dumps({"prompt": r["prompt"], "answer": r["answer"]}, ensure_ascii=False) + "\n")
st["selected"] = len(keep)
json.dump(st, open(f"{d}/select_stats.json", "w"), indent=1); print(json.dumps(st))
PYEOF
    say "regen round $t / $task: $(tail -1 "$dest/$task/select.log")"
    [ -s "$dest/$task/records.jsonl" ] || { say "EVENT: STEP_FAILED select $task round $t (no usable records)"; exit 1; }
  done
}

# ---- side probe (does NOT feed training): the professor's soft method -- last-layer bias learned for task j on
# model/j (header guard, loss after header), later experts masked (-inf) at the last layer; regenerate
# C-STANCE / FOMC from model/t and record the yield next to the forced-routing regen of the next round.
train_soft_bias(){   # task ckpt out port
  local task=$1 ckpt=$2 out=$3 port=$4
  [ -f "$out/last_bias.pt" ] && return 0
  say "soft bias: train $task last-layer bias on $(basename "$ckpt")"
  ( cd "$TRACE" && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PY -m torch.distributed.run --nproc_per_node=$NGPU --master_port=$port \
      scripts/bos_token/train_bos_token.py --checkpoint "$ckpt" \
      --train-json "$DATA/$task/train.json" --eval-json "$DATA/$task/eval.json" --eval-num 200 \
      --method last_bias --site router_logits --loss-scope after_header --bos-guard --guard-header --guard-decision $DEC \
      --lr 1e-2 --epochs 1 --micro-batch 4 --grad-accum 8 --eval-every 40 --out-dir "$out" ) > "$R/logs/softbias_$task.log" 2>&1
  [ -f "$out/last_bias.pt" ] || { say "EVENT: STEP_FAILED softbias_$task"; exit 1; }
  say "soft bias $task: $($PY -c "import torch;print([round(x,2) for x in torch.load('$out/last_bias.pt',weights_only=False)['bias'].tolist()])")"
}
soft_probe(){   # t j : regenerate task j from model/t with soft bias_j (+ last-layer mask of experts > j)
  local t=$1 j=$2
  local task=${TASKS[$j]} ckpt=$R/model/$t dest=$R/probe_soft/round_$t/${TASKS[$j]}
  [ -f "$dest/select_stats.json" ] && return 0
  mkdir -p "$dest"
  local nexp; nexp=$($PY -c "import json;print(json.load(open('$ckpt/lora_moe_meta.json'))['num_experts'])")
  local bias=$dest/last_bias_e${nexp}_masked.pt
  $PY - "$R/cond/soft_$task/last_bias.pt" "$nexp" "$bias" <<'PYEOF'
import sys, torch
lb = torch.load(sys.argv[1], map_location="cpu", weights_only=False); n = int(sys.argv[2])
v = lb["bias"]; k = v.numel() - 1
lb["bias"] = torch.cat([v[:k], torch.full((n - k,), float("-inf")), v[k:]]); torch.save(lb, sys.argv[3])
PYEOF
  say "soft probe round $t / $task: last-layer bias + last-layer mask, $GEN_PER_TASK docs"
  local per=$((GEN_PER_TASK / 8)) pids=() k
  for k in 0 1 2 3 4 5 6 7; do
    ( CUDA_VISIBLE_DEVICES=$k $PY "$TRACE/scripts/bos_token/gen_doc.py" --checkpoint "$ckpt" \
        --last-bias-file "$bias" --prompt-mode chat_header --bos-guard --guard-header --guard-decision $DEC --task-index "$j" \
        --num-seqs "$per" --batch "$per" --max-new-tokens "${CAP[$task]}" --seed $((2000 + t*100 + j*10 + k)) \
        --out-dir "$dest/stageA.s$k" > "$dest/genA_s$k.log" 2>&1 || exit 1
      CUDA_VISIBLE_DEVICES=$k $PY "$TRACE/scripts/analysis/answer_pass_v3_fix.py" --checkpoint "$ckpt" \
        --stage-a "$dest/stageA.s$k" --out "$dest/records.part$k.jsonl" --prompt-cue "${CUE[$task]}" \
        --bos-guard --guard-header --guard-decision $DEC --max-answer-tokens "${CAP[$task]}" --batch 32 > "$dest/genB_s$k.log" 2>&1 || exit 1 ) &
    pids+=($!); sleep 2
  done
  local rc=0 p; for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ "$rc" = 0 ] || { say "EVENT: STEP_FAILED soft probe round $t $task"; return 0; }
  cat "$dest"/records.part*.jsonl > "$dest/records.all.jsonl"
  $PY - "$dest" "$j" "$ANCHORS" "$GEN_PER_TASK" <<'PYEOF' > "$dest/select.log" 2>&1
import json, re, sys, glob, collections
d, j, anchors_path, req = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
anchor = json.load(open(anchors_path))[j]["anchor"]
pat = re.compile(r"^" + re.escape(anchor.rstrip("\n")) + {"0": r"\n?(.+?)\n对象：\n(.+?)\n态度：$", "1": r"\n?(.+?)\nStance:$"}[j], re.S)
rows = [json.loads(l) for l in open(f"{d}/records.all.jsonl")]
seen, keep = set(), []
for r in rows:
    m = pat.match(r["prompt"])
    if not m or not (20 <= len(m.group(1)) <= 400) or r["answer"].strip() not in ("A","B","C") or r["prompt"] in seen: continue
    seen.add(r["prompt"]); keep.append(r)
docs = [json.loads(l) for f in sorted(glob.glob(f"{d}/stageA.s*/docs.jsonl")) for l in open(f)]
kind = collections.Counter("C-STANCE" if (x["user"] or "").startswith("判断") else "FOMC" if "monetary policy" in (x["user"] or "") else "other" for x in docs)
st = {"requested": req, "kinds": dict(kind), "starts_with_anchor": sum(1 for x in docs if x["starts_with_anchor"]), "passed": len(keep), "yield": round(len(keep)/req, 4)}
json.dump(st, open(f"{d}/select_stats.json", "w"), indent=1); print(json.dumps(st, ensure_ascii=False))
PYEOF
  say "SOFT round $t / $task: $(tail -1 "$dest/select.log")"
}

for t in 0 1 2 3 4 5 6 7; do
  [ "$t" -gt 0 ] && [ "$REPLAY_SOURCE" = selfgen ] && gen_round "$t"
  train_round "$t"
done

# sparse15 evaluation with the guard on
if [ ! -f "$R/model/sparse15_summary.json" ]; then
  say "sparse15 eval (guard $DEC) start"
  ( cd "$TRACE" && SPARSE15_EVAL_EXTRA_ARGS="--bos_guard --guard_header --guard_decision $DEC" SPARSE15_CONV_MODE=llama3_template \
      SLORA_LLAMA31_PATH=$BASE $PY scripts/run_ours_sparse15_optimized.py \
      --run-dir "$R/model" --method "${METHOD_NAME:-residual_1phase_$REPLAY_SOURCE}" --gpus "$GPUS" ) > "$R/logs/eval.log" 2>&1
  [ -f "$R/model/sparse15_summary.json" ] || { say "EVENT: STEP_FAILED sparse15"; exit 1; }
  $PY - "$R/model/sparse15_summary.json" <<'PYEOF' | tee -a "$R/chain.log"
import json, sys
s = json.load(open(sys.argv[1])); tasks = s["tasks"]; diag = s["diagonal_scores_rounds_1_to_7"]; final = s["final_scores_round_8"]
print("AA %.2f  F %.2f  LA %.2f" % (s["final_average"], -s["BWT"], (sum(diag)+final[7])/8))
print("diag :", {t: round(v,2) for t,v in zip(tasks, diag)})
print("final:", {t: round(v,2) for t,v in zip(tasks, final)})
PYEOF
fi
say "EVENT: ALL_DONE"
