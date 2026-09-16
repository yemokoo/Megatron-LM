#!/usr/bin/env bash
# Scheduler for the 6-cell Ours HP sweep. Claims a free GPU pair, runs one
# cell, repeats. Never touches a card that already has a compute process.
#
#   bash sched_ours_hp.sh            # run the queue
#   DRY_RUN=1 bash sched_ours_hp.sh  # print the plan and exit
#
# Card policy: only the GPUs listed in ALLOWED are ever considered, and a pair
# is claimed only after both cards have been idle for IDLE_STREAK consecutive
# polls. Idleness is judged from nvidia-smi's compute-apps list, so another
# user's job is never displaced.
set -uo pipefail

O="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_ROOT="${SWEEP_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/ours_hp_sweep_20260912}"
ALLOWED="${ALLOWED:-0 1 2 3 4 5 6 7}"
POLL="${POLL:-120}"
IDLE_STREAK="${IDLE_STREAK:-3}"
DRY_RUN="${DRY_RUN:-0}"
MINISETS=/data2/seonghyeonnoh/LLM-continual-learning-data/router_finetune_miniset_repeats/seed1234

# cell : replay label : kd iters
#
# Centre (0p1pctx200, 360) is deliberately absent: it is the accepted run at
# ours_hyb_kd360_sub0p1_20260908, which is also the DoF 1x point. Re-running it
# would burn a cell to reproduce a number we already have.
QUEUE=(
  "r0p01_kd180:0p01pctx2000:180"
  "r0p01_kd360:0p01pctx2000:360"
  "r0p01_kd720:0p01pctx2000:720"
  "r0p1_kd180:0p1pctx200:180"
  "r0p1_kd720:0p1pctx200:720"
  "r1_kd360:1pctx20:360"
)

mkdir -p "$SWEEP_ROOT/logs"
LOG="$SWEEP_ROOT/logs/sched.log"
say(){ printf '[SCHED %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

busy_gpus(){
  # index list of every card currently holding a compute process
  local map busy=""
  map="$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader)"
  while IFS=, read -r bus _rest; do
    bus="$(echo "$bus" | xargs)"
    local idx
    idx="$(echo "$map" | awk -F', ' -v b="$bus" '$2==b{print $1}')"
    [ -n "$idx" ] && busy="$busy $idx"
  done < <(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader)
  echo "$busy"
}

cell_done(){
  local root="$1"
  local vp="$root/conv_1phase/latest_checkpointed_iteration.txt"
  [ -f "$vp" ] && [ "$(tr -d '[:space:]' < "$vp")" = "1800" ]
}

if [ "$DRY_RUN" = 1 ]; then
  echo "sweep root : $SWEEP_ROOT"
  echo "allowed    : $ALLOWED"
  printf '%-14s %-14s %-9s %-8s %s\n' CELL REPLAY KD_ITERS STATUS SUBSET
  for spec in "${QUEUE[@]}"; do
    IFS=: read -r cell label kd <<< "$spec"
    st="pending"; cell_done "$SWEEP_ROOT/$cell" && st="done"
    [ -d "$MINISETS/$label/wiki/train" ] || st="$st (SUBSET MISSING)"
    printf '%-14s %-14s %-9s %-8s %s\n' "$cell" "$label" "$kd" "$st" "$MINISETS/$label"
  done
  echo
  echo "centre cell r0p1_kd360 = /data2/seonghyeonnoh/LLM-continual-learning-runs/ours_hyb_kd360_sub0p1_20260908 (reused, not queued)"
  exit 0
fi

declare -A streak
port=46000
for spec in "${QUEUE[@]}"; do
  IFS=: read -r cell label kd <<< "$spec"
  root="$SWEEP_ROOT/$cell"
  if cell_done "$root"; then say "$cell already complete, skipping"; continue; fi
  if [ ! -d "$MINISETS/$label/wiki/train" ]; then
    say "$cell BLOCKED: subset $label missing -- run prepare_0p01pct_subset.sh first"
    continue
  fi

  pair=""
  while [ -z "$pair" ]; do
    busy="$(busy_gpus)"
    free=()
    for g in $ALLOWED; do
      if [[ " $busy " == *" $g "* ]]; then streak[$g]=0
      else streak[$g]=$(( ${streak[$g]:-0} + 1 )); fi
      (( ${streak[$g]:-0} >= IDLE_STREAK )) && free+=("$g")
    done
    if [ "${#free[@]}" -ge 2 ]; then
      pair="${free[0]},${free[1]}"
    else
      say "$cell waiting for a free pair (idle>=${IDLE_STREAK} polls); free now: ${free[*]:-none}"
      sleep "$POLL"
    fi
  done

  say "$cell -> GPU $pair (replay=$label kd=$kd)"
  SUB_LABEL="$label" KD_ITERS="$kd" CELL="$cell" SWEEP_ROOT="$SWEEP_ROOT" \
  GPUS="$pair" NPROC=2 PB="$port" \
    bash "$O/job_ours_hp.sh" >> "$SWEEP_ROOT/logs/$cell.out" 2>&1
  rc=$?
  say "$cell exit=$rc"
  port=$((port + 10))
  for g in $ALLOWED; do streak[$g]=0; done
done
say "queue drained"
