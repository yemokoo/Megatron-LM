#!/usr/bin/env bash
set -uo pipefail

# Compact view of a CKA contextual-replay run: step, loss, time, probe scores.
# Everything else in the Megatron log is suppressed.
#
#   watch_cka_replay_run.sh [run_dir] [-f]

RUN="${1:-/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_gt_contextual_replay_20260817/cka-bundle95-contextual-hidden_mse-c10-600step}"
LOG="$RUN/logs/a_to_b_freeze.log"
[[ -s "$LOG" ]] || { echo "no log yet: $LOG" >&2; exit 1; }

render() {
    awk '
    /^ *iteration +[0-9]+\/ *[0-9]+/ {
        step=""; ms=""; loss=""
        if (match($0, /iteration +[0-9]+\/ *[0-9]+/)) {
            s=substr($0, RSTART, RLENGTH); sub(/iteration +/,"",s); split(s,a,"/"); step=a[1]
        }
        if (match($0, /elapsed time per iteration \(ms\): *[0-9.]+/)) {
            s=substr($0, RSTART, RLENGTH); sub(/.*: */,"",s); ms=s
        }
        if (match($0, /lm loss: *[0-9.eE+-]+/)) {
            s=substr($0, RSTART, RLENGTH); sub(/.*: */,"",s); loss=s
        }
        printf "step %5s | loss %-10s | %6.1fs/it\n", step, loss, ms/1000
        next
    }
    /^probe .* at iteration/ {
        name=$2
        local_it=""; acc=""; ppl=""
        if (match($0, /local_iteration: *[0-9]+/)) { s=substr($0,RSTART,RLENGTH); sub(/.*: */,"",s); local_it=s }
        if (match($0, /next_token_acc: *[0-9.]+/))  { s=substr($0,RSTART,RLENGTH); sub(/.*: */,"",s); acc=s }
        if (match($0, /ppl: *[0-9.eE+-]+/))         { s=substr($0,RSTART,RLENGTH); sub(/.*: */,"",s); ppl=s }
        printf "  >> %-20s step %-5s acc %-10s ppl %s\n", name, local_it, acc, ppl
        next
    }
    ' "$LOG"
}

if [[ "${2:-}" == "-f" ]]; then
    render
    tail -n0 -F "$LOG" | stdbuf -oL grep --line-buffered -E "^ *iteration +[0-9]+/|^probe .* at iteration" \
    | stdbuf -oL sed -E \
        -e 's/^ *iteration +([0-9]+)\/ *[0-9]+.*elapsed time per iteration \(ms\): *([0-9.]+).*lm loss: *([0-9.eE+-]+).*/step \1 | loss \3 | \2ms\/it/' \
        -e 's/^probe ([a-z_]+) at iteration [0-9]+ \| local_iteration: ([0-9]+) \| next_token_acc: ([0-9.]+) \| ppl: ([0-9.eE+-]+).*/  >> \1 step \2 acc \3 ppl \4/'
else
    render
fi
