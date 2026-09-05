#!/usr/bin/env bash
set -uo pipefail

# Unattended GPU 0,1 chain:
#   1. small pilot census, validated before anything large runs
#   2. full threshold-free Conversation CKA census (2 workers)
#   3. Conversation random-control training from the Stage-A KD-init
#
# Deliberately NOT automated: threshold derivation, GT locking and the CKA-GT
# training arm. Those bake in a selection decision, so they wait for review.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
DATA=/data2/seonghyeonnoh/LLM-continual-learning-data
CHAIN=/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_conversation_chain_20260817
KD_INIT="$CHAIN/01_expansion_kd_init_e16_to_e24_step600"
CENSUS=/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_conv_census_20260817
RANDOM_STUDY=/data2/seonghyeonnoh/LLM-continual-learning-runs/random_control_20260817
STATUS="$CENSUS/logs/overnight_status.tsv"
mkdir -p "$CENSUS/logs" "$RANDOM_STUDY/logs"
say() { echo "$(date -Is) $*" | tee -a "$STATUS"; }

gpus_idle() {
    for id in 0 1; do
        used=$(nvidia-smi --id="$id" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null) || return 1
        (( used < 2000 )) || return 1
    done
}

say "WAIT stage-A checkpoint"
until [[ -d "$KD_INIT/iter_0000600" && "$(cat "$KD_INIT/latest_checkpointed_iteration.txt" 2>/dev/null)" == 600 ]]; do sleep 60; done
say "STAGE-A-READY"
until gpus_idle; do sleep 60; done
say "GPUS-FREE"

# ---- 1. pilot: one worker, few windows, then validate the output shape ----
say "PILOT-START 2048 windows"
GPU=0 WORKER_INDEX=0 WORKER_COUNT=1 MAX_WINDOWS=2048 WINDOW_BATCH_SIZE=32 \
    CENSUS_OUTPUT="$CENSUS/pilot" CACHE_PATH="$CENSUS/data_cache/pilot" \
    LOG_PATH="$CENSUS/logs/pilot.log" \
    bash scripts/analysis/run_cka_conv_census_mha.sh 0 >> "$CENSUS/logs/pilot_launcher.log" 2>&1
rc=$?
if (( rc != 0 )); then say "PILOT-FAILED rc=$rc — stopping, nothing large was run"; exit 1; fi
CENSUS_ROOT="$CENSUS/pilot" python3 scripts/analysis/check_conv_census_pilot.py \
    >> "$CENSUS/logs/pilot_check.log" 2>&1
if (( $? != 0 )); then say "PILOT-INVALID — stopping before the full census"; exit 1; fi
say "PILOT-OK"

# ---- 2. full census, two workers over the whole window axis ----
say "CENSUS-START full 2745838 windows, 2 workers"
for w in 0 1; do
    GPU=$w WORKER_INDEX=$w WORKER_COUNT=2 WINDOW_BATCH_SIZE=192 FORWARD_SUBBATCH_SIZE=128 \
        CENSUS_OUTPUT="$CENSUS/full_census" CACHE_PATH="$CENSUS/data_cache/worker_$w" \
        LOG_PATH="$CENSUS/logs/worker_00$w.log" \
        bash scripts/analysis/run_cka_conv_census_mha.sh "$w" \
        >> "$CENSUS/logs/worker_launcher_$w.log" 2>&1 &
done
wait
say "CENSUS-WORKERS-DONE"
PYTHONNOUSERSITE=1 /data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin/python \
    scripts/analysis/cka_gt_full_census.py merge-workers \
    --output-dir "$CENSUS/full_census" --worker-count 2 \
    >> "$CENSUS/logs/merge.log" 2>&1
say "CENSUS-MERGED rc=$?"

# ---- 3. fill the rest of the night with the verified random control arm ----
say "CONV-RANDOM-START"
env TASK=conversation TRAIN_ITERS=1800 \
    CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 MASTER_PORT=36795 \
    SOURCE="$KD_INIT" SOURCE_REQUIRED_ITERS=600 \
    REPLAY_MINISET_DIR="$DATA/random_replay_miniset_conversation_budget4147_20260817/train" \
    REPLAY_GT_PATH= REPLAY_SAMPLES=$((4147 * 200)) \
    STUDY_ROOT="$RANDOM_STUDY" \
    LABEL=random-budget4147-conversation-hidden_mse-c10-1800step \
    bash scripts/experiment/a100/run_g2_cka_gt_contextual_replay_mha.sh \
    >> "$RANDOM_STUDY/logs/conversation_random_1800.log" 2>&1
say "CONV-RANDOM-DONE rc=$?"
say "OVERNIGHT-COMPLETE"
