#!/usr/bin/env bash
set -uo pipefail

# The random-control mirror of the CKA chain, on GPUs 2,3:
#
#   1. Code LM + random-0.1% replay, 1800 steps   (from the Wiki KD-init)
#   2. expand E16->E24 + Wiki/Code output-KD init, 600 steps
#   3. Conversation LM + random replay, 1800 steps (from that KD-init)
#
# Every arm keeps the 200-epoch / 20%-budget exposure the CKA arms use, so the
# only difference from the CKA chain is which tokens get replayed.
# Only this script's own runs are ever started.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
DATA=/data2/seonghyeonnoh/LLM-continual-learning-data
STUDY=/data2/seonghyeonnoh/LLM-continual-learning-runs/random_control_20260817
CODE_OUT="$STUDY/random-0p1pct-code-hidden_mse-c10-1800step"
KD_INIT="$STUDY/02_expansion_kd_init_e16_to_e24_step600"
GPUS="${GPUS:-2,3}"
FREE_MIB="${FREE_MIB:-2000}"
POLL="${POLL:-120}"
mkdir -p "$STUDY/logs"
STATUS="$STUDY/logs/status.tsv"
say() { echo "$(date -Is) $*" | tee -a "$STATUS"; }

gpus_idle() {
    local ids used
    IFS=',' read -ra ids <<< "$GPUS"
    for id in "${ids[@]}"; do
        used=$(nvidia-smi --id="$id" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null) || return 1
        (( used < FREE_MIB )) || return 1
    done
}

complete_at() {  # dir iters
    [[ -d "$1/iter_$(printf '%07d' "$2")" ]] || return 1
    [[ "$(cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null)" == "$2" ]] || return 1
}

# The expansion step reads the source run's metadata from <ckpt>/logs; a run
# whose logs landed elsewhere would otherwise stall the chain here.
ensure_metadata() {
    local ckpt=$1 fallback=$2
    [[ -s "$ckpt/logs/run_metadata.json" ]] && return 0
    [[ -s "$fallback" ]] || return 1
    mkdir -p "$ckpt/logs" && cp -n "$fallback" "$ckpt/logs/run_metadata.json"
}

say "CHAIN-WAIT gpus=$GPUS"
until gpus_idle; do sleep "$POLL"; done
say "GPUS-FREE"

# ---- 1. Code + random replay ----
if complete_at "$CODE_OUT" 1800; then
    say "SKIP code-random already complete"
else
    say "CODE-RANDOM-START"
    env TRAIN_ITERS=1800 CUDA_VISIBLE_DEVICES="$GPUS" NPROC_PER_NODE=2 MASTER_PORT=36790 \
        REPLAY_MINISET_DIR="$DATA/random_replay_miniset_code_0p1pct_20260817/train" \
        REPLAY_GT_PATH= STUDY_ROOT="$STUDY" \
        LABEL=random-0p1pct-code-hidden_mse-c10-1800step \
        bash scripts/experiment/a100/run_g2_cka_gt_contextual_replay_mha.sh \
        >> "$STUDY/logs/code_random_1800.log" 2>&1
    say "CODE-RANDOM-DONE rc=$?"
    complete_at "$CODE_OUT" 1800 || { say "CODE-RANDOM-INCOMPLETE — stopping"; exit 1; }
fi

# ---- 2. expand to 24 experts and re-init on Wiki+Code ----
if complete_at "$KD_INIT" 600; then
    say "SKIP expansion already complete"
else
    ensure_metadata "$CODE_OUT" "$STUDY/logs/run_metadata.json" || true
    say "EXPANSION-START E16->E24 from code-random"
    env CUDA_VISIBLE_DEVICES="$GPUS" NPROC_PER_NODE=2 MASTER_PORT=34811 PYTHONNOUSERSITE=1 \
        FLAME_ENV=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100 \
        PYTHON_BIN=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin/python \
        PATH=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin:/usr/bin:/bin \
        CUDA_HOME=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100 \
        FLAME_DATA_ROOT="$DATA/flamedata2.data2-verified-backup" \
        TOKENIZER_MODEL=/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6 \
        WANDB_MODE=offline \
        SOURCE_WEIGHTS_DIR="$CODE_OUT" SOURCE_REQUIRED_ITERS=1800 \
        SOURCE_NUM_EXPERTS=16 NUM_EXPERTS=24 OLD_MODEL_KL_NUM_EXPERTS=16 \
        OLD_MODEL_KL_COEFF=1 OLD_MODEL_KL_TEMPERATURE=1 \
        TRAIN_ITERS=600 MICRO_BATCH_SIZE=32 GLOBAL_BATCH_SIZE=2304 \
        SAVE_INTERVAL=600 EVAL_INTERVAL=600 PROBE_EVAL_INTERVAL=100 \
        SECONDARY_PROBE_EVAL_INTERVAL=100 TERTIARY_PROBE_EVAL_INTERVAL=100 \
        RUN_ID=random-conv-expansion-kd-init-e16to24-step600 \
        TRAIN_WEIGHTS="$KD_INIT" LOCAL_BASE="$STUDY/local" LOCAL_SSD_ROOT="$STUDY/scratch/kd_init" \
        bash scripts/experiment/a100/run_g2_ffn_only_conversation_expert_distill_init_wikicode_mha.sh \
        >> "$STUDY/logs/02_expansion_kd_init.log" 2>&1
    say "EXPANSION-DONE rc=$?"
    complete_at "$KD_INIT" 600 || { say "EXPANSION-INCOMPLETE — stopping"; exit 1; }
fi

# ---- 3. Conversation + random replay ----
say "CONV-RANDOM-START"
env TASK=conversation TRAIN_ITERS=1800 \
    CUDA_VISIBLE_DEVICES="$GPUS" NPROC_PER_NODE=2 MASTER_PORT=36797 \
    SOURCE="$KD_INIT" SOURCE_REQUIRED_ITERS=600 \
    REPLAY_MINISET_DIR="$DATA/random_replay_miniset_conversation_budget4147_20260817/train" \
    REPLAY_GT_PATH= STUDY_ROOT="$STUDY" \
    LABEL=random-budget4147-conversation-from-random-chain-1800step \
    bash scripts/experiment/a100/run_g2_cka_gt_contextual_replay_mha.sh \
    >> "$STUDY/logs/conversation_random_from_random_chain.log" 2>&1
say "CONV-RANDOM-DONE rc=$?"
say "RANDOM-CHAIN-COMPLETE"
