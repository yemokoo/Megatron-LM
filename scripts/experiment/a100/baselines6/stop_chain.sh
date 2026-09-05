#!/bin/bash
# Cleanly stop the six-baseline chain so it can be resumed later.
#
# The trainer runs with --exit-signal-handler, so SIGTERM on the worker
# processes makes them save a checkpoint at the next iteration boundary and
# exit 0.  train_stage.sh then finds an incomplete tracker, returns non-zero,
# and run_all.sh (set -e) stops rather than starting the next baseline.
#
# Resume afterwards with launch_full_background.sh against the same output
# root: complete stages are skipped and the interrupted one resumes from the
# checkpoint written here.

set -uo pipefail

BASELINES6_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${BASELINES6_OUTPUT_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/baselines6_wiki_code_conversation_20260816}"
PID_FILE="$OUTPUT_ROOT/chain.pid"
WAIT_SECONDS="${STOP_WAIT_SECONDS:-300}"

if [ ! -s "$PID_FILE" ]; then
    echo "no chain.pid under $OUTPUT_ROOT; nothing to stop"
    exit 0
fi

chain_pid="$(tr -d '[:space:]' < "$PID_FILE")"
if ! [[ "$chain_pid" =~ ^[0-9]+$ ]]; then
    echo "ERROR: malformed chain.pid: $chain_pid" >&2
    exit 1
fi
if ! kill -0 "$chain_pid" 2>/dev/null; then
    echo "chain PID $chain_pid is already gone; nothing to stop"
    exit 0
fi

# launch_full_background.sh starts the chain under setsid, so the chain PID is
# also the process-group ID that every stage process belongs to.
#
# Match the trainer ranks specifically, not everything whose command line
# mentions the entrypoint: torchrun's elastic agent carries the script path too.
# Signalling the agent is worse than useless here -- it raises SignalException
# and tears its workers down immediately, so Megatron never reaches the next
# checkpoint_and_decide_exit and --exit-signal-handler saves nothing.  The real
# ranks are the agent's direct children; SIGTERM those and each one saves at its
# next iteration boundary and exits 0.
mapfile -t launchers < <(
    pgrep -g "$chain_pid" -f 'pretrain_gpt_baselines6\.py' 2>/dev/null | while read -r pid; do
        case "$(ps -o args= -p "$pid" 2>/dev/null)" in
            *torch.distributed.run*|*torch/distributed/run.py*) printf '%s\n' "$pid" ;;
        esac
    done
)
workers=()
for launcher in "${launchers[@]:-}"; do
    [ -n "$launcher" ] || continue
    while read -r child; do
        [ -n "$child" ] && workers+=("$child")
    done < <(pgrep -P "$launcher" 2>/dev/null)
done
if [ "${#workers[@]}" -eq 0 ]; then
    echo "chain PID $chain_pid is alive but no trainer is running (likely between stages)"
    echo "stopping the chain wrapper directly"
    kill -TERM -- "-$chain_pid" 2>/dev/null || kill -TERM "$chain_pid" 2>/dev/null
    exit 0
fi

echo "[STOP] signalling trainer workers: ${workers[*]}"
echo "[STOP] each worker saves at its next iteration boundary; allow up to ${WAIT_SECONDS}s"
for pid in "${workers[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
done

# Wait on the chain wrapper: it exits once the interrupted stage reports an
# incomplete checkpoint. Waiting on the workers alone would race the save.
waited=0
while [ "$waited" -lt "$WAIT_SECONDS" ]; do
    if ! kill -0 "$chain_pid" 2>/dev/null; then
        echo "[STOP] chain exited cleanly after ${waited}s"
        for dir in "$OUTPUT_ROOT"/*/*/latest_checkpointed_iteration.txt; do
            [ -f "$dir" ] || continue
            printf '  %s -> %s\n' "${dir%/latest_checkpointed_iteration.txt}" "$(tr -d '[:space:]' < "$dir")"
        done
        exit 0
    fi
    sleep 5
    waited=$(( waited + 5 ))
done

echo "ERROR: chain PID $chain_pid still alive after ${WAIT_SECONDS}s" >&2
echo "inspect $OUTPUT_ROOT/chain.log before forcing it down with: kill -TERM -- -$chain_pid" >&2
exit 1
