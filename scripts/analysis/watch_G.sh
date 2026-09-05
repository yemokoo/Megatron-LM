#!/usr/bin/env bash
# Compact live view of a Megatron run.log: step | time/iter | ETA | lm loss | TFLOP/s | probe acc.
#   bash scripts/analysis/watch_G.sh [run.log] [once]   (default: G stage-3 log; "once" = print history and exit)
L="${1:-/data2/seonghyeonnoh/LLM-continual-learning-runs/selfgen_replay_G_20260827/conversation_1phase_gen/logs/run.log}"
src(){ if [ "${2:-}" = once ]; then cat "$L"; else tail -n 400 -F "$L"; fi; }
src "$L" "${2:-}" | grep --line-buffered -aE "elapsed time per iteration|probe .* next_token_acc" | while IFS= read -r line; do
  if [[ "$line" == *"elapsed time per iteration"* ]]; then
    ts=$(grep -oE "^ *\[[0-9-]+ [0-9:]+\]" <<<"$line" | tr -d '[] ' | cut -c11-15)
    it=$(grep -oE "iteration +[0-9]+/ *[0-9]+" <<<"$line" | tr -d ' ' | sed 's/iteration//'); cur=${it%%/*}; tot=${it##*/}
    ms=$(grep -oE "elapsed time per iteration \(ms\): [0-9.]+" <<<"$line" | awk '{print $NF}')
    loss=$(grep -oE " lm loss: [0-9.Ee+-]+" <<<"$line" | head -1 | awk '{printf "%.3f",$NF}')
    rloss=$(grep -oE "replay/lm loss: [0-9.Ee+-]+" <<<"$line" | awk '{printf "%.3f",$NF}')
    tf=$(grep -oE "TFLOP/s/GPU\): [0-9.]+" <<<"$line" | awk '{print $NF}')
    eta=$(awk -v c="$cur" -v t="$tot" -v m="$ms" 'BEGIN{s=(t-c)*m/1000; printf "%dh%02dm", s/3600, (s%3600)/60}')
    printf "%s  step %4s/%s  %5.1fs/it  ETA %s  conv_loss %s  replay_loss %s  %s TFLOP/s\n" "$ts" "$cur" "$tot" "$(awk -v m=$ms 'BEGIN{print m/1000}')" "$eta" "$loss" "${rloss:-?}" "${tf:-?}"
  else
    p=$(grep -oE "probe [a-z]+_probe" <<<"$line" | head -1 | awk '{print $2}' | sed "s/_probe//"); li=$(grep -oE "local_iteration: [0-9]+" <<<"$line" | awk '{print $NF}'); acc=$(grep -oE "next_token_acc: [0-9.]+" <<<"$line" | awk '{printf "%.4f",$NF}')
    printf "        step %4s  probe %-12s acc %s\n" "$li" "$p" "$acc"
  fi
done
