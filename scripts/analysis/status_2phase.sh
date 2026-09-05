#!/usr/bin/env bash
# One-screen status of the 2-phase 4-run set: per run/stage -> saved step, current iteration, s/it, latest probes.
#   bash scripts/analysis/status_2phase.sh            (once)      |   watch -n 60 bash scripts/analysis/status_2phase.sh
R="${1:-/data2/seonghyeonnoh/LLM-continual-learning-runs/g2_2phase_4runs_20260827}"
date '+%F %H:%M'; grep -aE "start|done|FAILED" "$R/logs/lanes_0123.log" 2>/dev/null | sed -E 's/^\[0123 [0-9-]+ //' | cut -c1-90
for l in hyb_kd1 ffn_kd0 ffn_kd1 hyb_kd0; do [ -f "$R/logs/$l.log" ] || continue
  echo "== $l  ($(grep -aE '^\[2P ' "$R/logs/$l.log" | tail -1 | sed -E 's/^\[2P [0-9-]+ ([0-9:]+) [a-z]+\/kd[01]\] /\1 /' | cut -c1-70))"
  for s in code_kd_init code_task code_router_ft conv_kd_init conv_task conv_router_ft; do d="$R/$l/$s"; [ -d "$d" ] || continue
    f=$(ls -t "$d"/logs/*.log 2>/dev/null | head -1); it=$(grep -aoE 'iteration +[0-9]+/ +[0-9]+ \| .*elapsed time per iteration \(ms\): [0-9.]+' "$f" 2>/dev/null | tail -1 | sed -E 's/iteration +([0-9]+)\/ +([0-9]+) \|.*\(ms\): ([0-9.]+)/\1\/\2 @\3ms/')
    pr=$(grep -aoE 'probe [a-z]+_probe at iteration [0-9]+ \| local_iteration: [0-9]+ \| next_token_acc: [0-9.]+' "$f" 2>/dev/null | tail -3 | sed -E 's/probe ([a-z]+)_probe at iteration [0-9]+ \| local_iteration: ([0-9]+) \| next_token_acc: ([0-9.]{5}).*/\1@\2=\3/' | tr '\n' ' ')
    printf "   %-15s saved=%-5s %-22s %s\n" "$s" "$(cat "$d/latest_checkpointed_iteration.txt" 2>/dev/null || echo -)" "$it" "$pr"; done; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | head -4 | tr '\n' ' '; echo
