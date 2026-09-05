#!/bin/bash
# Wait for the GRPO queue to finish BOTH of its runs and fully release its
# GPUs, then hand the remaining baselines6 work to four 2-GPU lanes.
#
# The GRPO top-level script owns both the scalar and discrete runs and only
# exits once both are done, so its pid is the primary signal.  It is not a
# child of this shell, so kill -0 polling is the only option; wait(1) would
# fail with "not a child".
#
# That pid alone is not enough to start taking GPUs, though: if the top-level
# script were killed, its trainers and reward services would keep running and
# keep training.  Never pull a GPU out from under a live trainer.  So after the
# pid goes away this also waits until no GRPO child process remains and until
# the GRPO devices actually report free memory.  All three must agree.
#
# GRPO *failing* does not block this -- either way the GPUs come free, and the
# baselines6 work is unrelated.  Its exit status is only reported.

set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GRPO_PID="${GRPO_PID:-516789}"
GRPO_ROOT="${GRPO_ROOT:-/data2/seonghyeonnoh/androidflux-rl/queued_runs/first_error_20260816_234931}"
GRPO_GPUS="${GRPO_GPUS:-0 1 2 3 4 5}"
GRPO_CHILD_PATTERN="${GRPO_CHILD_PATTERN:-run_checkpoint_qwen2b_lora\.sh|reward_service|launch_grpo}"
POLL_SECONDS="${GRPO_POLL_SECONDS:-60}"
SETTLE_SECONDS="${GRPO_SETTLE_SECONDS:-30}"
PER_GPU_FREE_MB="${GRPO_PER_GPU_FREE_MB:-2000}"

stamp() { date '+%F %T'; }

# --- 1. the top-level queue: exits only after scalar AND discrete ------------
echo "[CHAIN] waiting on GRPO pid $GRPO_PID (poll ${POLL_SECONDS}s) $(stamp)"
while kill -0 "$GRPO_PID" 2>/dev/null; do
    sleep "$POLL_SECONDS"
done
echo "[CHAIN] GRPO pid $GRPO_PID exited $(stamp)"

scalar_status="$(cat "$GRPO_ROOT/full/scalar/exit_status" 2>/dev/null || echo '?')"
discrete_status="$(cat "$GRPO_ROOT/full/discrete/exit_status" 2>/dev/null || echo '?')"
if [ -f "$GRPO_ROOT/CHAIN_COMPLETE" ]; then
    echo "[CHAIN] GRPO CHAIN_COMPLETE (scalar=$scalar_status discrete=$discrete_status)"
else
    echo "[CHAIN] GRPO left no CHAIN_COMPLETE (scalar=$scalar_status discrete=$discrete_status)"
    echo "[CHAIN] proceeding regardless; this only needs the GPUs, not GRPO's result"
fi

# --- 2. no GRPO trainer or reward service may still be alive ----------------
while :; do
    # Count with wc, not `pgrep -c`: pgrep prints 0 *and* exits 1 when nothing
    # matches, so a `|| echo 0` fallback appends a second line and every later
    # integer test dies with "integer expression expected".
    alive="$(pgrep -f "$GRPO_CHILD_PATTERN" 2>/dev/null | wc -l)"
    [ "$alive" -eq 0 ] && break
    echo "[CHAIN] $alive GRPO child process(es) still alive; waiting $(stamp)"
    sleep "$SETTLE_SECONDS"
done
echo "[CHAIN] no GRPO child processes remain $(stamp)"

# --- 3. the devices themselves must report free -----------------------------
while :; do
    busy=""
    for gpu in $GRPO_GPUS; do
        used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)"
        [ "$used" -gt "$PER_GPU_FREE_MB" ] && busy="$busy $gpu(${used}MB)"
    done
    [ -z "$busy" ] && break
    echo "[CHAIN] GRPO GPUs still holding memory:$busy $(stamp)"
    sleep "$SETTLE_SECONDS"
done
echo "[CHAIN] GRPO GPUs ($GRPO_GPUS) are free $(stamp)"

# --- 4. only now release GPUs 6,7 from the single-chain run ------------------
# The trainer runs with --exit-signal-handler, so the in-flight stage saves a
# checkpoint before exiting and its lane resumes it from there.
echo "[CHAIN] stopping the single-chain baselines6 run $(stamp)"
bash "$DIR/stop_chain.sh" || echo "[CHAIN] stop_chain reported non-zero; continuing"

for _ in $(seq 1 60); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            | awk '{total += $1} END {print total+0}')"
    [ "$used" -lt 8000 ] && break
    echo "[CHAIN] total GPU memory in use: ${used} MB; waiting $(stamp)"
    sleep 10
done

echo "[CHAIN] starting lanes $(stamp)"
exec bash "$DIR/run_lanes.sh"
