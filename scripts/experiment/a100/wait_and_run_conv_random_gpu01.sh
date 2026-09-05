#!/usr/bin/env bash
set -uo pipefail

# After Stage A (E16->E24 expansion + KD-init) completes on GPUs 0,1, train the
# Conversation random control from that same checkpoint.  Only this script's own
# run is ever started; nothing else is touched.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA=/data2/seonghyeonnoh/LLM-continual-learning-data
CHAIN=/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_conversation_chain_20260817
KD_INIT="$CHAIN/01_expansion_kd_init_e16_to_e24_step600"
STUDY=/data2/seonghyeonnoh/LLM-continual-learning-runs/random_control_20260817
GPUS="${GPUS:-0,1}"
FREE_MIB="${FREE_MIB:-2000}"
POLL="${POLL:-120}"
mkdir -p "$STUDY/logs"
STATUS="$STUDY/logs/status.tsv"

kd_init_complete() {
    [[ -d "$KD_INIT/iter_0000600" ]] || return 1
    [[ "$(cat "$KD_INIT/latest_checkpointed_iteration.txt" 2>/dev/null)" == "600" ]] || return 1
    return 0
}

gpus_idle() {
    local ids used
    IFS=',' read -ra ids <<< "$GPUS"
    for id in "${ids[@]}"; do
        used=$(nvidia-smi --id="$id" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null) || return 1
        (( used < FREE_MIB )) || return 1
    done
    return 0
}

echo "$(date -Is) WAIT stage-A checkpoint then gpus=$GPUS" | tee -a "$STATUS"
until kd_init_complete; do sleep "$POLL"; done
echo "$(date -Is) KD-INIT-READY $KD_INIT" | tee -a "$STATUS"
until gpus_idle; do sleep "$POLL"; done
echo "$(date -Is) GPUS-FREE starting Conversation random control" | tee -a "$STATUS"

env TASK=conversation TRAIN_ITERS=1800 \
    CUDA_VISIBLE_DEVICES="$GPUS" NPROC_PER_NODE=2 MASTER_PORT=36795 \
    SOURCE="$KD_INIT" SOURCE_REQUIRED_ITERS=600 \
    REPLAY_MINISET_DIR="$DATA/random_replay_miniset_conversation_budget4147_20260817/train" \
    REPLAY_GT_PATH= \
    REPLAY_SAMPLES=$((4147 * 200)) \
    STUDY_ROOT="$STUDY" \
    LABEL=random-budget4147-conversation-hidden_mse-c10-1800step \
    bash "$REPO/scripts/experiment/a100/run_g2_cka_gt_contextual_replay_mha.sh" \
    >> "$STUDY/logs/conversation_random_1800.log" 2>&1
echo "$(date -Is) DONE Conversation random control rc=$?" | tee -a "$STATUS"
