#!/usr/bin/env bash
# Full pipeline for GOAL_bos_cstance_fomc.md: generate C-STANCE replay two ways
# (BoS-token vs production anchor), train a FOMC round from each on the same
# 1phase_kd_rep round-0 checkpoint, then score C-STANCE+FOMC for both plus the
# real-replay reference.
set -uo pipefail
TRACE=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace
IMPL=$TRACE/implementations/llmcl_benchmark
PY=$TRACE/.venv-runtime/bin/python
CK0=/data2/seonghyeonnoh/paper/ablation/1phase_kd_rep/0
ROOT=/data2/seonghyeonnoh/paper/bos_token/fomc_arms
DATA=/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace
BASE=/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct
BOS_FILE=/data2/seonghyeonnoh/paper/bos_token/cstance_1phase_kd_rep/lr1e-3/bos_token.pt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT"
say() { printf '[FOMC-ARM %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$ROOT/progress.log"; }

ANCHOR=$($PY -c "import json;print(json.load(open('$TRACE/scripts/selfgen/assets/anchors.json'))['0']['anchor'],end='')")

##############################################################################
# Step 1: BoS-token C-STANCE replay
##############################################################################
step1() {
  local d=$ROOT/bos_gen
  if [ -f "$d/gen/round_1/C-STANCE/records.jsonl" ]; then say "step1: already have records.jsonl, skip"; return 0; fi
  mkdir -p "$d/gen/round_1/C-STANCE"
  say "step1: generating 640 BoS-token docs on GPU0"
  CUDA_VISIBLE_DEVICES=0 $PY "$TRACE/scripts/bos_token/gen_doc.py" --checkpoint "$CK0" \
    --bos-token-file "$BOS_FILE" --num-seqs 640 --batch 64 --max-new-tokens 320 \
    --seed 1 --out-dir "$d/stageA" > "$d/gen_stageA.log" 2>&1 || { say "step1: stageA FAILED"; return 1; }
  say "step1: stage B (greedy answers)"
  CUDA_VISIBLE_DEVICES=0 $PY "$TRACE/scripts/analysis/answer_pass_v3_fix.py" --checkpoint "$CK0" \
    --stage-a "$d/stageA" --out "$d/records.all.jsonl" --prompt-cue $'\n态度：' \
    --max-answer-tokens 256 --batch 64 > "$d/gen_stageB.log" 2>&1 || { say "step1: stageB FAILED"; return 1; }
  say "step1: selecting 500 clean records"
  $PY - "$d" "$ANCHOR" <<'PYEOF' > "$d/select.log" 2>&1 || { echo "step1 select FAILED" >> "$ROOT/progress.log"; exit 1; }
import json, re, random, sys
d, anchor = sys.argv[1], sys.argv[2]
pat = re.compile(r"^" + re.escape(anchor) + r"(.+?)\n对象：\n(.+?)\n态度：$", re.S)
rows = [json.loads(l) for l in open(f"{d}/records.all.jsonl")]
seen, keep = set(), []
for r in rows:
    m = pat.match(r["prompt"])
    if not m:
        continue
    body, obj = m.group(1), m.group(2)
    if not (20 <= len(body) <= 400):
        continue
    if r["answer"].strip() not in ("A", "B", "C"):
        continue
    if r["prompt"] in seen:
        continue
    seen.add(r["prompt"])
    keep.append(r)
stats = {"total": len(rows), "passed": len(keep),
         "answer_hist": {a: sum(1 for r in keep if r["answer"].strip() == a) for a in "ABC"},
         "mean_body_len": sum(len(pat.match(r["prompt"]).group(1)) for r in keep) / max(len(keep), 1)}
if len(keep) < 500:
    raise SystemExit(f"only {len(keep)} passed, need 500 -- rerun stage A/B with more seqs")
random.Random(0).shuffle(keep)
keep = keep[:500]
with open(f"{d}/gen/round_1/C-STANCE/records.jsonl", "w") as fh:
    for r in keep:
        fh.write(json.dumps({"prompt": r["prompt"], "answer": r["answer"]}, ensure_ascii=False) + "\n")
stats["selected"] = len(keep)
json.dump(stats, open(f"{d}/select_stats.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(stats, ensure_ascii=False))
PYEOF
  say "step1: DONE $(cat "$d/select_stats.json")"
}

##############################################################################
# Step 2: production-style anchor C-STANCE replay (8-way parallel, GPUs 0-7)
##############################################################################
step2() {
  local d=$ROOT/anchor_gen
  if [ -f "$d/gen/round_1/C-STANCE/records.jsonl" ]; then say "step2: already have records.jsonl, skip"; return 0; fi
  mkdir -p "$d/gen/round_1/C-STANCE"
  say "step2: generating 8x80 anchor docs across GPUs 0-7"
  local pids=() k
  for k in 0 1 2 3 4 5 6 7; do
    ( set -uo pipefail
      CUDA_VISIBLE_DEVICES=$k $PY "$TRACE/scripts/analysis/bos_sample_v3.py" --checkpoint "$CK0" --mode anchor \
        --prefix-text "$ANCHOR" --num-seqs 80 --max-seqs 200000 --max-new-tokens 256 --batch 80 \
        --no-routing-probe --seed $((200 + k)) --out-dir "$d/stageA.shard$k" --label "r1_C-STANCE_s$k" \
        > "$d/genA_s$k.log" 2>&1 || exit 1
      CUDA_VISIBLE_DEVICES=$k $PY "$TRACE/scripts/analysis/answer_pass_v3_fix.py" --checkpoint "$CK0" \
        --stage-a "$d/stageA.shard$k" --out "$d/records.part$k.jsonl" --prompt-cue $'\n态度：' \
        --max-answer-tokens 256 --batch 64 > "$d/genB_s$k.log" 2>&1 || exit 1
    ) &
    pids+=($!)
    sleep 6
  done
  local rc=0 p; for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ "$rc" = 0 ] || { say "step2: FAILED (see $d/gen*_s*.log)"; return 1; }
  cat "$d"/records.part*.jsonl > "$d/gen/round_1/C-STANCE/records.jsonl"
  local n; n=$(wc -l < "$d/gen/round_1/C-STANCE/records.jsonl")
  say "step2: DONE $n records -> $d/gen/round_1/C-STANCE/records.jsonl"
  [ "$n" -ge 500 ] || { say "step2: only $n records, need >=500"; return 1; }
}

##############################################################################
# Step 3: train one FOMC round (8 GPUs) from a given C-STANCE replay
##############################################################################
train_round() {   # name gen_root
  local name=$1 gen_root=$2
  local run=$ROOT/$name
  local out=$run/model
  if [ -f "$out/1/lora_moe_meta.json" ]; then say "$name: round 1 checkpoint exists, skip training"; return 0; fi
  mkdir -p "$out" "$run/logs"
  [ -d "$out/0" ] || { say "$name: copying round-0 checkpoint"; cp -r "$CK0" "$out/0"; }
  [ -d "$out/fixed_replay_memory" ] || cp -r /data2/seonghyeonnoh/paper/ablation/2phase_kd_gen/model/fixed_replay_memory "$out/"
  [ -f "$gen_root/C-STANCE/records.jsonl" ] || { say "$name: missing replay records at $gen_root"; return 1; }
  say "$name: FOMC round training start (8 GPU, selfgen replay <- $gen_root)"
  ( cd "$IMPL" && env SELFGEN_ROOT="$run/gen/round_1" SELFGEN_CURRENT_TASK=FOMC \
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      RESUME_CONTRACT_ALLOW_DRIFT=active_stream_samples_per_primary_epoch,joint_replay_active_stream_samples_per_primary_epoch,replay_source \
      $PY -m torch.distributed.run --nproc_per_node=8 --master_port=29891 \
      "$TRACE/scripts/selfgen/train_selfgen.py" \
      --training_version v3_new_replay1to1 --model_name_or_path "$BASE" \
      --data_path "$DATA" --dataset_name all --data_output_path "$run/data_cache" \
      --output_dir "$out" --num_train_epochs 5,3,7,5,3,5,5,7 \
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
      --v3_epoch_probe_samples 64 --tokenized_train_cache_dir /data2/seonghyeonnoh/LLM-continual-learning-caches/llama31_8b_instruct/slora_chat_full_len1024 \
      --disable_training_flop_counter --stop_after_task FOMC \
      --ablation_phase_mode 1phase --ablation_kd_init on \
      --ablation_replay_source selfgen \
      --resume_checkpoint "$out/0" ) > "$run/logs/train_r1.log" 2>&1
  local rc=$?
  if [ ! -f "$out/1/lora_moe_meta.json" ]; then say "$name: TRAIN FAILED rc=$rc ($run/logs/train_r1.log)"; return 1; fi
  say "$name: FOMC round training done"
}

##############################################################################
# Step 4: eval C-STANCE+FOMC on round-1 checkpoint
##############################################################################
eval_round1() {   # name checkpoint_dir out_dir
  local name=$1 ckpt=$2 outdir=$3 gpu=$4
  say "$name: eval start on GPU$gpu ($ckpt)"
  mkdir -p "$outdir"
  ( cd "$IMPL" && CUDA_VISIBLE_DEVICES=$gpu $PY -u evaluate_Ours_LoRA_MoE.py --checkpoint_dir "$ckpt" \
      --base_model_name_or_path "$BASE" --data_path "$DATA" --inference_tasks C-STANCE,FOMC \
      --inference_output_path "$outdir" --summary_filename r1_cstance_fomc.summary.json \
      --max_prompt_len 0 --max_ans_len 1024 --no-task_generation_limits --slora_conv_mode llama3 \
      --per_device_eval_batch_size 16 --temperature 0 ) > "$outdir/eval.log" 2>&1
  local rc=$?
  [ -f "$outdir/r1_cstance_fomc.summary.json" ] || { say "$name: EVAL FAILED rc=$rc ($outdir/eval.log)"; return 1; }
  say "$name: eval done -> $outdir/r1_cstance_fomc.summary.json"
}

write_result() {
  $PY - <<PYEOF
import json
from pathlib import Path
root = Path("$ROOT")
rows = [("bos_gen", root/"bos_gen/evaluation/order1/r1_cstance_fomc.summary.json"),
        ("anchor_gen", root/"anchor_gen/evaluation/order1/r1_cstance_fomc.summary.json"),
        ("real_replay (ref)", root/"ref_real/evaluation/order1/r1_cstance_fomc.summary.json")]
lines = ["# RESULT: BoS-token vs anchor-generated C-STANCE replay -> FOMC round\n",
         "Round-0 C-STANCE reference (1phase_kd_rep, real replay): 58.35\n",
         "| arm | C-STANCE@r1 | FOMC@r1 | C-STANCE forgetting (58.35 - score) |",
         "|---|---|---|---|"]
select_stats = root / "bos_gen/select_stats.json"
for name, p in rows:
    if p.is_file():
        d = json.loads(p.read_text())
        c, f = d.get("C-STANCE", {}), d.get("FOMC", {})
        def sc(x):
            return x.get("accuracy", x.get("score", next(iter(x.values()), None))) if isinstance(x, dict) else x
        cs, fs = sc(c), sc(f)
        forget = (58.35 - cs) if isinstance(cs, (int, float)) else None
        lines.append(f"| {name} | {cs} | {fs} | {forget} |")
    else:
        lines.append(f"| {name} | (missing: {p}) | | |")
if select_stats.is_file():
    lines.append("\n## bos_gen replay selection stats\n")
    lines.append("```\n" + select_stats.read_text() + "\n```")
(root / "RESULT.md").write_text("\n".join(lines) + "\n")
print((root / "RESULT.md").read_text())
PYEOF
}

say "PIPELINE START"
step1 || { say "PIPELINE ABORT at step1"; exit 1; }
step2 || { say "PIPELINE ABORT at step2"; exit 1; }
train_round bos_gen "$ROOT/bos_gen/gen/round_1" || { say "PIPELINE ABORT at bos_gen training"; exit 1; }
train_round anchor_gen "$ROOT/anchor_gen/gen/round_1" || { say "PIPELINE ABORT at anchor_gen training"; exit 1; }
say "training chain done; starting parallel eval"
eval_round1 bos_gen "$ROOT/bos_gen/model/1" "$ROOT/bos_gen/evaluation/order1" 0 &
p1=$!
eval_round1 anchor_gen "$ROOT/anchor_gen/model/1" "$ROOT/anchor_gen/evaluation/order1" 1 &
p2=$!
eval_round1 real_replay_ref /data2/seonghyeonnoh/paper/ablation/1phase_kd_rep/1 "$ROOT/ref_real/evaluation/order1" 2 &
p3=$!
rc=0
wait $p1 || rc=1
wait $p2 || rc=1
wait $p3 || rc=1
write_result
say "PIPELINE DONE rc=$rc"
