#!/usr/bin/env bash
# Live view of both CKA-32 chains: current stage, step, loss, latest probes.
#   watch_cka32_chains.sh [refresh_seconds]
EVERY="${1:-10}"
ROOT=/data2/seonghyeonnoh/LLM-continual-learning-runs
declare -A CH=( [hidden_mse_c10]="0-7" )
latest_log(){ # study -> newest a_to_b_freeze / expansion log
  ls -t "$1"/*/logs/*.log 2>/dev/null | grep -vE "gpu_usage|run_metadata" | head -1
}
show(){
  local obj=$1 study=$ROOT/cka32_${GT_SET:-top3relL2pack}_chain_${obj}_20260819 st=$study/logs/status.tsv log
  echo "==== chain: $obj   (GPU ${CH[$obj]})"
  [ -f "$st" ] && tail -2 "$st" | sed 's/^/  /' || echo "  (status 없음)"
  log=$(latest_log "$study"); [ -n "$log" ] || { echo; return; }
  echo "  log: $(echo $log | sed "s|$ROOT/||")"
  # step / loss line
  grep -oE "iteration +[0-9]+/ *[0-9]+ \|.*lm loss: [0-9.E+-]+" "$log" 2>/dev/null | tail -1 \
    | sed -E 's/iteration +([0-9]+)\/ *([0-9]+).*consumed samples: *([0-9]+).*elapsed time per iteration \(ms\): *([0-9.]+).*lm loss: ([0-9.E+-]+).*/  step \1\/\2   \4 ms\/it   lm_loss \5/'
  # replay loss if present
  grep -oE "iteration +[0-9]+/.*" "$log" 2>/dev/null | tail -1 | tr '|' '\n' \
    | grep -E "joint_replay/(lm loss|old hidden mse loss)" | sed -E 's/^ */    replay /' | head -2
  # probes: last row per probe
  echo "  probes (latest):"
  grep -hE "^probe .* at iteration" "$log" 2>/dev/null | sed -E 's/probe ([a-z_]+) at iteration ([0-9]+) \| local_iteration: ([0-9]+) \| next_token_acc: ([0-9.]+).*/\1 \3 \4/' \
    | awk '{a[$1]=$0} END{for(k in a){split(a[k],f," "); printf "    %-20s step %-5s acc %s\n",f[1],f[2],f[3]}}' | sort
  echo
}
t0=$(date +%s)
while true; do
  [ -t 1 ] && clear
  echo "CKA-32 top4% chains   $(date '+%H:%M:%S')   elapsed $(( ($(date +%s)-t0)/60 ))m"; echo
  show hidden_mse_c10
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | awk -F, '{printf "GPU%s %2dG %3d%%  ",$1,$2/1024,$3} END{print ""}'
  sleep "$EVERY"
done
