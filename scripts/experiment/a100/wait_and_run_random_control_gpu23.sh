#!/usr/bin/env bash
set -uo pipefail

# Hold until GPUs 2 and 3 are idle, then run the random-control replay arm.
# Nothing else is touched: the script only ever starts its own run.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA=/data2/seonghyeonnoh/LLM-continual-learning-data
STUDY=/data2/seonghyeonnoh/LLM-continual-learning-runs/random_control_20260817
GPUS="${GPUS:-2,3}"
FREE_MIB="${FREE_MIB:-2000}"
POLL="${POLL:-120}"
mkdir -p "$STUDY/logs"
STATUS="$STUDY/logs/status.tsv"

gpus_idle() {
    local ids used
    IFS=',' read -ra ids <<< "$GPUS"
    for id in "${ids[@]}"; do
        used=$(nvidia-smi --id="$id" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null) || return 1
        (( used < FREE_MIB )) || return 1
    done
    return 0
}

echo "$(date -Is) WAIT gpus=$GPUS threshold=${FREE_MIB}MiB" | tee -a "$STATUS"
until gpus_idle; do sleep "$POLL"; done
echo "$(date -Is) GPUS-FREE starting Code random control" | tee -a "$STATUS"

env TRAIN_ITERS=1800 CUDA_VISIBLE_DEVICES="$GPUS" NPROC_PER_NODE=2 MASTER_PORT=36790 \
    REPLAY_MINISET_DIR="$DATA/random_replay_miniset_code_0p1pct_20260817/train" \
    REPLAY_GT_PATH= \
    REPLAY_SAMPLES=$((4147 * 200)) \
    STUDY_ROOT="$STUDY" \
    LABEL=random-0p1pct-code-hidden_mse-c10-1800step \
    bash "$REPO/scripts/experiment/a100/run_g2_cka_gt_contextual_replay_mha.sh" \
    >> "$STUDY/logs/code_random_1800.log" 2>&1
echo "$(date -Is) DONE Code random control rc=$?" | tee -a "$STATUS"
