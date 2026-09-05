#!/usr/bin/env bash
# Live progress bars for a multi-worker census run.
#   watch_census_progress.sh [census_output_dir] [refresh_seconds]
OUT="${1:-/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_conv_census_20260817/full_census_scale32}"
EVERY="${2:-1}"
bar() { local p=$1 w=40; local f=$(( p*w/100 )); printf "%s%s" "$(printf '#%.0s' $(seq 1 $f 2>/dev/null))" "$(printf '.%.0s' $(seq 1 $((w-f)) 2>/dev/null))"; }
t0=$(date +%s)
while true; do
  [ -t 1 ] && clear
  echo "census: $OUT      $(date '+%H:%M:%S')   elapsed $(( ($(date +%s)-t0)/60 ))m"
  echo
  tot_done=0; tot_all=0
  for d in "$OUT"/worker_*; do
    [ -d "$d" ] || continue
    w=$(basename "$d")
    if [ -f "$d/progress.json" ]; then
      read done all win < <(python3 -c "import json;d=json.load(open('$d/progress.json'));print(d.get('next_batch_index',0),d.get('expected_batches',1),d.get('processed_windows',0))" 2>/dev/null)
    else done=0; all=1; win=0; fi
    [ "$all" -gt 0 ] || all=1
    p=$(( done*100/all )); tot_done=$((tot_done+done)); tot_all=$((tot_all+all))
    printf "%-11s [%s] %3d%%  batch %5d/%-5d  windows %8s\n" "$w" "$(bar $p)" "$p" "$done" "$all" "$win"
  done
  [ "$tot_all" -gt 0 ] && { P=$(( tot_done*100/tot_all )); echo; printf "%-11s [%s] %3d%%  %d/%d batches\n" "TOTAL" "$(bar $P)" "$P" "$tot_done" "$tot_all"; }
  echo
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | awk -F, '{printf "GPU%s %5dMiB %3d%%   ", $1, $2, $3} END{print ""}'
  if grep -q "SCALE32-MERGE" "$(dirname "$OUT")/logs/scale32_status.tsv" 2>/dev/null && [ "$P" -ge 100 ] 2>/dev/null; then echo; echo "DONE"; break; fi
  sleep "$EVERY"
done
