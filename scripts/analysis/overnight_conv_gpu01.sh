#!/usr/bin/env bash
set -uo pipefail

# Unattended GPU 0,1 chain. Everything here is *extraction* — no threshold or
# GT decision is baked in, so a review in the morning can still change the
# selector without re-running any GPU work.
#
#   1. pilot census, validated before anything large starts
#   2. full Conversation CKA census (2 workers)
#   3. Wiki + Code calibration censuses  (the old domains that set thresholds)
#   4. threshold derivation from those histograms (CPU)
#   5. Conversation random-control training, to use the remaining hours
#
# Left for review: candidate extraction, the targeted T/M pass, GT lock and the
# CKA-GT training arm.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PYENV=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin/python
DATA=/data2/seonghyeonnoh/LLM-continual-learning-data
VER=$DATA/flamedata2.data2-verified-backup
CHAIN=/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_conversation_chain_20260817
KD_INIT="$CHAIN/01_expansion_kd_init_e16_to_e24_step600"
C=/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_conv_census_20260817
RSTUDY=/data2/seonghyeonnoh/LLM-continual-learning-runs/random_control_20260817
STATUS="$C/logs/overnight_status.tsv"
CAL_WINDOWS="${CAL_WINDOWS:-300000}"
mkdir -p "$C/logs" "$RSTUDY/logs"
say() { echo "$(date -Is) $*" | tee -a "$STATUS"; }

gpus_idle() {
    for id in 0 1; do
        used=$(nvidia-smi --id="$id" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null) || return 1
        (( used < 2000 )) || return 1
    done
}

run_census() {  # name manifest_dir data_prefix output max_windows gpu worker count config
    local name=$1 manifest=$2 prefix=$3 out=$4 maxw=$5 gpu=$6 widx=$7 wcnt=$8 config=$9
    GPU="$gpu" WORKER_INDEX="$widx" WORKER_COUNT="$wcnt" MAX_WINDOWS="$maxw" \
        WINDOW_BATCH_SIZE=192 FORWARD_SUBBATCH_SIZE=128 \
        ANALYSIS_CONFIG="$config" \
        DATA_PREFIX="$prefix" CENSUS_MANIFEST="$manifest/manifest.json" \
        CENSUS_OUTPUT="$out" CACHE_PATH="$C/data_cache/${name}_$widx" \
        LOG_PATH="$C/logs/${name}_worker$widx.log" MASTER_PORT=$((36870 + widx)) \
        bash scripts/analysis/run_cka_conv_census_mha.sh "$gpu" \
        >> "$C/logs/${name}_launcher$widx.log" 2>&1
}

say "WAIT stage-A checkpoint"
until [[ -d "$KD_INIT/iter_0000600" && "$(cat "$KD_INIT/latest_checkpointed_iteration.txt" 2>/dev/null)" == 600 ]]; do sleep 60; done
say "STAGE-A-READY"
until gpus_idle; do sleep 60; done
say "GPUS-FREE"

say "PILOT-START 2048 windows"
run_census pilot "$C/manifest" "$DATA/conversation_merged_train_20260817/train_text_document" \
    "$C/pilot" 2048 0 0 1 "$C/analysis_config_conversation.json"
CENSUS_ROOT="$C/pilot" $PYENV scripts/analysis/check_conv_census_pilot.py >> "$C/logs/pilot_check.log" 2>&1
if (( $? != 0 )); then say "PILOT-INVALID — stopped before any long run"; exit 1; fi
say "PILOT-OK"

say "CONV-CENSUS-START 2745838 windows, 2 workers"
for w in 0 1; do
    run_census conv "$C/manifest" "$DATA/conversation_merged_train_20260817/train_text_document" \
        "$C/full_census" 0 "$w" "$w" 2 "$C/analysis_config_conversation.json" &
done
wait
PYTHONNOUSERSITE=1 $PYENV scripts/analysis/cka_gt_full_census.py merge-workers \
    --output-dir "$C/full_census" --worker-count 2 >> "$C/logs/merge_conv.log" 2>&1
say "CONV-CENSUS-DONE rc=$?"

for dom in wiki code; do
    say "CAL-CENSUS-START $dom capped=$CAL_WINDOWS"
    for w in 0 1; do
        run_census "cal_$dom" "$C/manifest_$dom" "$VER/$dom/train/train_text_document" \
            "$C/calibration_$dom" "$CAL_WINDOWS" "$w" "$w" 2 "$C/analysis_config_cal_$dom.json" &
    done
    wait
    PYTHONNOUSERSITE=1 $PYENV scripts/analysis/cka_gt_full_census.py merge-workers \
        --output-dir "$C/calibration_$dom" --worker-count 2 >> "$C/logs/merge_cal_$dom.log" 2>&1
    say "CAL-CENSUS-DONE $dom rc=$?"
done

say "THRESHOLDS-START"
$PYENV scripts/analysis/derive_thresholds_from_census.py \
    --old-census "wiki=$C/calibration_wiki" --old-census "code=$C/calibration_code" \
    --output "$C/thresholds_from_old_domains.json" >> "$C/logs/thresholds.log" 2>&1
say "THRESHOLDS-DONE rc=$?"

say "CONV-RANDOM-START"
env TASK=conversation TRAIN_ITERS=1800 \
    CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 MASTER_PORT=36795 \
    SOURCE="$KD_INIT" SOURCE_REQUIRED_ITERS=600 \
    REPLAY_MINISET_DIR="$DATA/random_replay_miniset_conversation_budget4147_20260817/train" \
    REPLAY_GT_PATH= STUDY_ROOT="$RSTUDY" \
    LABEL=random-budget4147-conversation-hidden_mse-c10-1800step \
    bash scripts/experiment/a100/run_g2_cka_gt_contextual_replay_mha.sh \
    >> "$RSTUDY/logs/conversation_random_1800.log" 2>&1
say "CONV-RANDOM-DONE rc=$?"
say "OVERNIGHT-COMPLETE"
