#!/usr/bin/env bash
# Live view of the whole scale-32 chain: cal32 wiki / cal32 code / Code32 full.
CONV=/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_conv_census_20260817
CODE=/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_gt_full_census_20260816/analysis/cka_gt_full_census_v1
EVERY="${1:-10}"
bar(){ local p=$1 w=40; local f=$((p*w/100)); printf '%*s' $f '' | tr ' ' '#'; printf '%*s' $((w-f)) '' | tr ' ' '.'; }
section(){ # title dir
  local title=$1 dir=$2 td=0 ta=0 any=0
  echo "== $title  ($dir)"
  for d in "$dir"/worker_*; do [ -d "$d" ] || continue; any=1
    read done all win < <(python3 -c "import json;d=json.load(open('$d/progress.json'));print(d.get('next_batch_index',0),d.get('expected_batches',1),d.get('processed_windows',0))" 2>/dev/null || echo "0 1 0")
    [ "$all" -gt 0 ] || all=1; p=$((done*100/all)); td=$((td+done)); ta=$((ta+all))
    printf "  %-11s [%s] %3d%%  batch %5d/%-5d  windows %8s\n" "$(basename $d)" "$(bar $p)" $p $done $all $win
  done
  [ $any = 1 ] && { P=$((td*100/ta)); printf "  %-11s [%s] %3d%%\n" TOTAL "$(bar $P)" $P; } || echo "  (대기)"
  echo
}
t0=$(date +%s)
while true; do
  [ -t 1 ] && clear
  echo "scale-32 chain   $(date '+%H:%M:%S')   elapsed $(( ($(date +%s)-t0)/60 ))m"; echo
  section "cal32 wiki (conv pair, 4 GPU)"  "$CONV/calibration_wiki_scale32"
  section "cal32 code (conv pair, 4 GPU)"  "$CONV/calibration_code_scale32"
  section "Code32 full census (Code pair, 8 GPU)" "$CODE/full_census_scale32"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | awk -F, '{printf "GPU%s %5dMiB %3d%%   ",$1,$2,$3} END{print ""}'
  echo; tail -3 "$CONV/logs/scale32_status.tsv" 2>/dev/null | sed 's/^/  /'
  sleep "$EVERY"
done
