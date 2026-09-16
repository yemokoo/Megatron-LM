#!/usr/bin/env bash
# Ours HP sweep, one cell: hybrid(ffn+attn shared router) w352/r256,
#   code kd_init → code 1phase 1800 → conv kd_init → conv 1phase 1800
#
# Forked from the chain that produced the accepted centre run
# (ours_hyb_kd360_sub0p1_20260908).  That script already parameterised the
# replay axis via SUB; the only thing added here is the KD-init step axis,
# which was hardcoded to 360 in five places (TRAIN_ITERS, SAVE_INTERVAL and
# the completion checks for both kd stages, plus DISTILL_SOURCE_REQUIRED_ITERS
# for both 1phase stages).  Everything else -- wiki source, stage scripts,
# micro batches, replay budget, seed -- is byte-identical to the centre run, so
# a cell differs from the centre in exactly the two swept quantities.
#
#   usage: SUB_LABEL=0p01pctx2000 KD_ITERS=180 GPUS=0,1 NPROC=2 \
#          bash job_ours_hp.sh
set -uo pipefail

P=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning
A=$P/scripts/experiment/a100
HG=/data2/seonghyeonnoh/LLM-continual-learning-runs/hf_g2_wiki_code_conversation/g2_wiki_code_conversation
FLAME=/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup
MINISETS=/data2/seonghyeonnoh/LLM-continual-learning-data/router_finetune_miniset_repeats/seed1234

SUB_LABEL="${SUB_LABEL:?set SUB_LABEL, e.g. 0p1pctx200}"
KD_ITERS="${KD_ITERS:?set KD_ITERS, e.g. 360}"
SUB="$MINISETS/$SUB_LABEL"
CELL="${CELL:-r${SUB_LABEL}_kd${KD_ITERS}}"
SWEEP_ROOT="${SWEEP_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/ours_hp_sweep_20260912}"
ROOT="${ROOT:-$SWEEP_ROOT/$CELL}"

GPUS="${GPUS:-0,1}"; NPROC="${NPROC:-2}"
MB_KD="${MB_KD:-32}"; MB_1P="${MB_1P:-64}"
REPLAY_SAMPLES="${REPLAY_SAMPLES:-829440}"
PB="${PB:-46000}"
TASK_ITERS="${TASK_ITERS:-1800}"

[ -d "$SUB/wiki/train" ] && [ -d "$SUB/code/train" ] || {
    echo "[ERROR] replay miniset missing: $SUB (build it with prepare_0p01pct_subset.sh)" >&2
    exit 2; }

cd "$P"; mkdir -p "$ROOT/scratch" "$ROOT/weights" "$ROOT/g2" "$ROOT/logs"
say(){ printf '[OURSHP %s %s] %s\n' "$CELL" "$(date '+%F %T')" "$*" | tee -a "$ROOT/logs/chain.log"; }
at(){ [ -f "$1/latest_checkpointed_iteration.txt" ] && tr -d '[:space:]' < "$1/latest_checkpointed_iteration.txt" || echo ""; }

export TOKENIZER_MODEL=/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6
export FLAME_DATA_ROOT=$FLAME STAGE_INPUTS_TO_SCRATCH=0 DIRECT_LOCAL_SAVE=1 PAUSE_SECONDS=0 PYTHONNOUSERSITE=1
export NO_SAVE_OPTIM=1 GLOBAL_BATCH_SIZE=2304 WANDB_MODE=offline SEED=1234 LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND=0
export PATH=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin:/usr/bin:/bin
export CUDA_VISIBLE_DEVICES=$GPUS NPROC_PER_NODE=$NPROC
export LOCAL_SSD_ROOT="$ROOT/scratch" LOCAL_WEIGHTS="$ROOT/weights" G2_ROOT="$ROOT/g2"
export GUARD_GRACE_SECONDS=100000 GUARD_POLL_SECONDS=60
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WIKI=$HG/sources/ffn_attn_shared_router/wiki
CKD=$ROOT/code_kd; CP=$ROOT/code_1phase; VKD=$ROOT/conv_kd; VP=$ROOT/conv_1phase

# Both divisions must be exact or Megatron silently reshapes the global batch.
(( 2304 % (MB_KD * NPROC) == 0 )) || { say "FAIL: 2304 % (${MB_KD}x${NPROC}) != 0"; exit 2; }
(( 2304 % (MB_1P * NPROC) == 0 )) || { say "FAIL: 2304 % (${MB_1P}x${NPROC}) != 0"; exit 2; }
(( REPLAY_SAMPLES % (MB_1P * NPROC) == 0 )) || { say "FAIL: replay ${REPLAY_SAMPLES} % (${MB_1P}x${NPROC}) != 0"; exit 2; }

say "start  replay=$SUB_LABEL  kd_iters=$KD_ITERS  GPU $GPUS (w$NPROC)  mb kd$MB_KD/1p$MB_1P  budget $REPLAY_SAMPLES"

if [ "$(at "$CKD")" != "$KD_ITERS" ]; then
  say "1/4 code kd_init $KD_ITERS"
  STAGE1_WEIGHTS_DIR="$WIKI" SOURCE_REQUIRED_ITERS=1800 \
  RUN_ID=ourshp-$CELL-code-kd TRAIN_WEIGHTS="$CKD" TRAIN_ITERS=$KD_ITERS SAVE_INTERVAL=$KD_ITERS \
  MICRO_BATCH_SIZE=$MB_KD MASTER_PORT=$((PB+0)) \
    bash "$A/run_g2_shared_router_code_expert_distill_init_mha.sh" logits >> "$ROOT/logs/code_kd.log" 2>&1
  [ "$(at "$CKD")" = "$KD_ITERS" ] || { say "FAIL code_kd"; exit 1; }
else say "1/4 skip"; fi

if [ "$(at "$CP")" != "$TASK_ITERS" ]; then
  say "2/4 code 1phase $TASK_ITERS (replay subset $SUB_LABEL, budget $REPLAY_SAMPLES)"
  STAGE1_WEIGHTS_DIR="$WIKI" SOURCE_REQUIRED_ITERS=1800 \
  RESUME_FROM_WEIGHTS="$CKD" DISTILL_SOURCE_REQUIRED_ITERS=$KD_ITERS RESUME_LOAD_OPTIM=0 RESUME_RESET_ITERATION=1 \
  JOINT_REPLAY_DATASET="$SUB/wiki/train" JOINT_REPLAY_TOTAL_SAMPLES=$REPLAY_SAMPLES \
  RUN_ID=ourshp-$CELL-code-1phase TRAIN_WEIGHTS="$CP" TRAIN_ITERS=$TASK_ITERS SAVE_INTERVAL=$TASK_ITERS \
  MICRO_BATCH_SIZE=$MB_1P MASTER_PORT=$((PB+1)) \
    bash "$A/run_g2_shared_router_code_wiki_joint_lm_allrouter_mha_replaymb.sh" >> "$ROOT/logs/code_1phase.log" 2>&1
  [ "$(at "$CP")" = "$TASK_ITERS" ] || { say "FAIL code_1phase"; exit 1; }
else say "2/4 skip"; fi

if [ "$(at "$VKD")" != "$KD_ITERS" ]; then
  say "3/4 conv kd_init $KD_ITERS"
  STAGE1_WEIGHTS_DIR="$CP" SOURCE_REQUIRED_ITERS=1800 \
  RUN_ID=ourshp-$CELL-conv-kd TRAIN_WEIGHTS="$VKD" TRAIN_ITERS=$KD_ITERS SAVE_INTERVAL=$KD_ITERS \
  MICRO_BATCH_SIZE=$MB_KD MASTER_PORT=$((PB+2)) \
    bash "$A/run_g2_shared_router_conversation_expert_distill_init_wikicode_mha.sh" logits >> "$ROOT/logs/conv_kd.log" 2>&1
  [ "$(at "$VKD")" = "$KD_ITERS" ] || { say "FAIL conv_kd"; exit 1; }
else say "3/4 skip"; fi

if [ "$(at "$VP")" != "$TASK_ITERS" ]; then
  say "4/4 conv 1phase $TASK_ITERS (replay wiki+code, budget $REPLAY_SAMPLES)"
  STAGE1_WEIGHTS_DIR="$CP" SOURCE_REQUIRED_ITERS=1800 \
  RESUME_FROM_WEIGHTS="$VKD" DISTILL_SOURCE_REQUIRED_ITERS=$KD_ITERS RESUME_LOAD_OPTIM=0 RESUME_RESET_ITERATION=1 \
  JOINT_REPLAY_DATASET="$SUB/wiki/train" JOINT_REPLAY_SECONDARY_DATASET="$SUB/code/train" \
  JOINT_REPLAY_TOTAL_SAMPLES=$REPLAY_SAMPLES \
  RUN_ID=ourshp-$CELL-conv-1phase TRAIN_WEIGHTS="$VP" TRAIN_ITERS=$TASK_ITERS SAVE_INTERVAL=$TASK_ITERS \
  MICRO_BATCH_SIZE=$MB_1P MASTER_PORT=$((PB+3)) \
    bash "$A/run_g2_shared_router_conversation_wikicode_joint_lm_allrouter_mha_replaymb.sh" >> "$ROOT/logs/conv_1phase.log" 2>&1
  [ "$(at "$VP")" = "$TASK_ITERS" ] || { say "FAIL conv_1phase"; exit 1; }
else say "4/4 skip"; fi

say "DONE code_kd=$(at "$CKD") code_1phase=$(at "$CP") conv_kd=$(at "$VKD") conv_1phase=$(at "$VP")"
