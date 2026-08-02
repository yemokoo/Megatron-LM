#!/bin/bash
# Per-task batch survey, ONE TASK PER PROCESS (fresh CUDA allocator each), 4 GPUs
# per wave, 2 waves, then merge. One task per process avoids the cross-task
# caching-allocator fragmentation that underestimated 2nd-in-process tasks.
cd "$(dirname "$0")/../.." || exit 1
export PATH=/home/work/Agent_HJ/30_flame_agent/envs/train_env/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
# Avoid caching-allocator fragmentation: small-B probes early in a search fragment
# the 81GB so a slightly larger B fails to find a contiguous block and OOMs far
# below true capacity (C-STANCE OOM'd at B=17/~54GB). expandable_segments makes
# the survey measure REAL capacity. MUST also be set in real training so the
# surveyed batch sizes transfer 1:1.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/work/Agent_HJ/30_flame_agent/envs/train_env/bin/python
MODEL=/home/work/Agent_HJ/00_models/Qwen3-8B
PARTS=eval_out/batch_survey_parts2
mkdir -p logs "$PARTS"

# wave = "task_gpu0 task_gpu1 task_gpu2 task_gpu3"; long+short mixed for balance
WAVES=("MeetingBank Py150 C-STANCE NumGLUE-cm" "20Minuten ScienceQA FOMC NumGLUE-ds")

w=1
for wave in "${WAVES[@]}"; do
  echo "=== WAVE $w: $wave ==="
  read -r t0 t1 t2 t3 <<< "$wave"
  gpu=0
  for t in "$t0" "$t1" "$t2" "$t3"; do
    echo "  GPU$gpu -> $t"
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/batch_survey.py \
      --model_name_or_path "$MODEL" \
      --data_path data/LLM-CL-Benchmark_5000 \
      --data_output_path /tmp/data_files_survey_g$gpu/ \
      --tasks "$t" --out "$PARTS/$t.json" \
      > "logs/batch_survey2_${t}.log" 2>&1 &
    gpu=$((gpu + 1))
  done
  wait
  echo "=== WAVE $w done ==="
  w=$((w + 1))
done

echo "=== MERGE ==="
$PY - <<'EOF'
import json, glob
ALL=["C-STANCE","FOMC","MeetingBank","Py150","ScienceQA","NumGLUE-cm","NumGLUE-ds","20Minuten"]
merged={}; gpu_total=None
for f in sorted(glob.glob("eval_out/batch_survey_parts2/*.json")):
    d=json.load(open(f)); merged.update(d["results"]); gpu_total=d.get("gpu_total_mib")
json.dump({"gpu_total_mib":gpu_total,"results":merged},
          open("eval_out/batch_survey_8b.json","w"),indent=2)
print(f"GPU total = {gpu_total:.0f} MiB\n")
print(f"{'task':12s} {'maxlen':>6s} {'OFF max':>8s} {'OFF peak':>9s} {'ON max':>7s} {'ON peak':>8s}  RECOMMEND")
for t in ALL:
    r=merged.get(t)
    if not r: print(f"{t:12s}  (missing)"); continue
    off=r["off"]; on=r.get("on")
    offb=f"{off['max_batch']}{'+' if off['capped'] else ''}"
    offp=f"{off['peak_mib']:.0f}" if off['peak_mib'] else "-"
    onb=f"{on['max_batch']}{'+' if on['capped'] else ''}" if on else "-"
    onp=f"{on['peak_mib']:.0f}" if on and on['peak_mib'] else "-"
    rec=r["recommend"]; recs=f"b={rec['batch']} ckpt={'ON' if rec['grad_ckpt'] else 'off'}"
    print(f"{t:12s} {r['max_tok_len']:6d} {offb:>8s} {offp:>9s} {onb:>7s} {onp:>8s}  {recs}")
comma=",".join(str(merged[t]["recommend"]["batch"]) for t in ALL if t in merged)
print(f"\n--per_device_train_batch_size {comma}\n   (order: {','.join(ALL)})")
EOF
echo "DONE -> eval_out/batch_survey_8b.json"
