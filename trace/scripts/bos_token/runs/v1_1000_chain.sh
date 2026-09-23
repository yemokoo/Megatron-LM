#!/usr/bin/env bash
# Header-guarded training (header 36 tokens residual at every layer, header tokens carry no loss)
# + forced "\n\n" routing for replay generation:
#   round t: regenerate EVERY previous task j<t from model/(t-1) by forcing the "\n\n" position to E_j at
#            all layers (other slots -inf; every other position routes freely), 1000 docs/task spread over
#            8 GPUs, strict/lenient filter -> 500 records; train task t on them; discard; repeat.
#   end    : sparse15 evaluation (guard on).
set -uo pipefail
TRACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IMPL="$TRACE/implementations/llmcl_benchmark"
PY="${TRACE_PYTHON:-$TRACE/.venv-runtime/bin/python}"
BASE="${SLORA_LLAMA31_PATH:-$TRACE/models/Llama-3.1-8B-Instruct}"
DATA="${TRACE_DATA_ROOT:-$TRACE/data/trace}"
CACHE="${TRACE_TOKEN_CACHE_DIR:-$TRACE/cache/llama31_8b_instruct/slora_chat_full_len1024}"
SRC0="${V1_CSTANCE_MODEL_ROOT:-$TRACE/seed_v1/model}" # contains 0/lora_moe_meta.json
R="${TRACE_RUN_ROOT:-$TRACE/results/header_forced_v1_1000}"
ANCHORS="$TRACE/scripts/selfgen/assets/anchors.json"
export TRACE_DATA_ROOT="$DATA"
DEC=none
GEN_PER_TASK=1000          # 125 per GPU, select at most 500
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
mkdir -p "$R/model" "$R/gen/round_0" "$R/cond" "$R/logs"
say(){ printf '[HF1K %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$R/chain.log"; }
TASKS=(C-STANCE FOMC MeetingBank Py150 ScienceQA NumGLUE-cm NumGLUE-ds 20Minuten)
declare -A CAP=( [C-STANCE]=256 [FOMC]=256 [MeetingBank]=1024 [Py150]=1024 [ScienceQA]=512 [NumGLUE-cm]=256 [NumGLUE-ds]=256 )
declare -A CUE=( [C-STANCE]=$'\n态度：' [FOMC]=$'\nStance:' [MeetingBank]=$'\nSummary:' [Py150]='' [ScienceQA]=$'\nAnswer:' [NumGLUE-cm]=$'\nAnswer:' [NumGLUE-ds]=$'\nAnswer:' )

# 0. seed model/0
if [ ! -f "$R/model/0/lora_moe_meta.json" ]; then
  [ -f "$SRC0/0/lora_moe_meta.json" ] || { say "EVENT: STEP_FAILED missing v1 C-STANCE seed: $SRC0/0"; exit 1; }
  for seed_file in "$SRC0/training_workload.json" "$SRC0/epoch_probe.jsonl" "$SRC0/fixed_replay_memory/task_0_C-STANCE.json"; do
    [ -f "$seed_file" ] || { say "EVENT: STEP_FAILED missing v1 seed file: $seed_file"; exit 1; }
  done
  cp -a "$SRC0/0" "$R/model/0"
  mkdir -p "$R/model/fixed_replay_memory"; cp -a "$SRC0/fixed_replay_memory/task_0_C-STANCE.json" "$R/model/fixed_replay_memory/" 2>/dev/null || true
  $PY - "$SRC0" "$R/model" <<'PYEOF'
import json, sys
src, out = sys.argv[1], sys.argv[2]
w = json.load(open(f"{src}/training_workload.json")); w["tasks"] = [t for t in w["tasks"] if t["round"] == 0]
if isinstance(w.get("totals"), dict):
    for k in list(w["totals"]):
        if isinstance(w["totals"][k], (int, float)): w["totals"][k] = None
json.dump(w, open(f"{out}/training_workload.json", "w"), indent=2)
open(f"{out}/epoch_probe.jsonl", "w").writelines(l for l in open(f"{src}/epoch_probe.jsonl") if json.loads(l)["round"] == 0)
PYEOF
  say "model/0 seeded from v1 header_forced (C-STANCE)"
fi

train_round(){   # t
  local t=$1
  local task=${TASKS[$t]} port=$((29970+t))
  [ -f "$R/model/$t/lora_moe_meta.json" ] && return 0
  local -a resume=(); [ "$t" -gt 0 ] && resume=(--resume_checkpoint "$R/model/$((t-1))")
  say "train round $t ($task): header guard ($DEC) + header loss OFF, replay <- gen/round_$t, 8 GPUs"
  ( cd "$IMPL" && env SELFGEN_ROOT="$R/gen/round_$t" SELFGEN_CURRENT_TASK="$task" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      BOS_GUARD_HEADER=1 BOS_GUARD_DECISION=$DEC HEADER_NO_LOSS=1 SELFGEN_ALLOW_SHORT=1 \
      RESUME_CONTRACT_ALLOW_DRIFT=active_stream_samples_per_primary_epoch,joint_replay_active_stream_samples_per_primary_epoch \
      $PY -m torch.distributed.run --nproc_per_node=8 --master_port=$port \
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
      --ablation_phase_mode 1phase --ablation_kd_init off --ablation_replay_source selfgen \
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
  say "regen round $t: tasks 0..$((t-1)) from model/$((t-1)) (forced \\n\\n routing, $GEN_PER_TASK docs/task, 8 GPUs)"
  mkdir -p "$dest"
  for j in $(seq 0 $((t-1))); do force_file "$j" "$nexp" >/dev/null; done
  $PY "$TRACE/scripts/bos_token/regen_queue_v1.py" --checkpoint "$ckpt" --dest "$dest" \
    --cond-dir "$R/cond" --num-experts "$nexp" --round "$t" --num-tasks "$t" \
    --default-per-task "$GEN_PER_TASK" --gpus 0,1,2,3,4,5,6,7 \
    --guard-decision "$DEC" > "$dest/regen_queue.log" 2>&1 || { say "EVENT: STEP_FAILED regen round $t"; exit 1; }
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
keep = []                      # drop only broken records (structure / no answer); no dedup, no length cut
for r in rows:
    p, a = r["prompt"], r["answer"].strip()
    if pat:
        if not pat.match(p) or a not in ("A", "B", "C"): continue
    else:
        if not p.startswith(anchor.rstrip("\n")) or not a: continue
    keep.append(r)
docs = [json.loads(l) for f in sorted(glob.glob(f"{d}/stageA.s*/docs.jsonl")) for l in open(f)]
st = {"requested": req, "docs": len(docs), "starts_with_anchor": sum(1 for x in docs if x["starts_with_anchor"]),
      "stageB": len(rows), "passed": len(keep), "unique": len({r["prompt"] for r in keep}), "yield": round(len(keep) / req, 4)}
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
  ( cd "$TRACE" && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PY -m torch.distributed.run --nproc_per_node=8 --master_port=$port \
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
  [ "$t" -gt 0 ] && gen_round "$t"
  train_round "$t"
  [ "$t" = 0 ] && train_soft_bias C-STANCE "$R/model/0" "$R/cond/soft_C-STANCE" 29990
  [ "$t" = 1 ] && train_soft_bias FOMC "$R/model/1" "$R/cond/soft_FOMC" 29991
  [ "$t" -ge 1 ] && soft_probe "$t" 0
  [ "$t" -ge 2 ] && soft_probe "$t" 1
done

# sparse15 evaluation with the guard on
if [ ! -f "$R/model/sparse15_summary.json" ]; then
  say "sparse15 eval (guard $DEC) start"
  ( cd "$TRACE" && SPARSE15_EVAL_EXTRA_ARGS="--bos_guard --guard_header --guard_decision $DEC" \
      SLORA_LLAMA31_PATH=$BASE $PY scripts/run_ours_sparse15_optimized.py \
      --run-dir "$R/model" --method header_forced_v1_1000 --gpus 0,1,2,3,4,5,6,7 ) > "$R/logs/eval.log" 2>&1
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
