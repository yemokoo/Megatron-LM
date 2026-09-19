#!/usr/bin/env bash
# This host's ablation queue: the three KD-on arms, serial, 8 GPUs each.
#
# The other host runs the four KD-off arms (see README.md).  Each arm here is
# resumable -- rerunning this script skips finished rounds and continues.
#
#   setsid nohup bash run_host_a.sh > /data2/seonghyeonnoh/paper/ablation/host_a.log 2>&1 &
set -uo pipefail
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-/data2/seonghyeonnoh/paper/ablation}
GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}
mkdir -p "$ROOT"
say() { printf '[HOST-A %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$ROOT/progress.log"; }

say "queue start: 1phase_kd_rep -> 2phase_kd_rep -> 2phase_kd_gen (gpus $GPU_LIST)"

# 1) the published arm, re-trained at world 8 / global batch 64 so every cell
#    of the table shares the final model's training condition
PHASE=1phase KD=on NAME=1phase_kd_rep ROOT="$ROOT" GPUS="$GPU_LIST" PORT=29871 \
  bash "$D/run_arm_real.sh" || say "1phase_kd_rep failed; continuing"

# 2) phase-mode axis with real replay
PHASE=2phase KD=on NAME=2phase_kd_rep ROOT="$ROOT" GPUS="$GPU_LIST" PORT=29873 \
  bash "$D/run_arm_real.sh" || say "2phase_kd_rep failed; continuing"

# 3) phase-mode axis with self-generated replay (train -> generate -> repeat)
PHASE=2phase KD=on NAME=2phase_kd_gen ROOT="$ROOT" GPU_LIST="$GPU_LIST" PORT=29875 \
  bash "$D/run_arm_gen.sh" || say "2phase_kd_gen failed; continuing"

say "queue done"
