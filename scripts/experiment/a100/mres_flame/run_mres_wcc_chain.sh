#!/usr/bin/env bash
# Mass-transfer reservoir on the G2 shared-router hybrid (FFN + QKVO LoRA experts), wiki -> code -> conv.
#
#   wiki  e8 from scratch (G2 recipe: top-4, FFN 352, QKVO full-rank LoRA r256, 1800 it, gbs 2304)
#         with the reservoir from task 0: router-only replay pass on the wiki 0.1%x200 subset trains
#         its margin while alpha refills 0 -> 8.
#   code  expand 8 -> 16 straight from wiki (no KD-init): new rows := r_res, new FFN down := 0,
#         new QKVO LoRA B = 0 (stock), alpha 8 -> 0 -> refill 8.  1-phase 1800 it: primary Code LM
#         (new experts + all router rows) + router-only replay on wiki+code subsets (+ margin).
#   conv  expand 16 -> 24 the same way; replay on wiki+code+conv subsets.
# Everything not listed (optimizer, LR schedule, aux/z loss, top-4, 8 experts per task, budgets,
# seed, micro batches) follows the Ours hybrid chain (ours_hyb_kd360_sub0p1_20260908) minus KD-init.
#
#   GPUS=0,1,2,3 bash scripts/experiment/a100/mres_flame/run_mres_wcc_chain.sh
set -uo pipefail

P=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning
A=$P/scripts/experiment/a100
FLAME=/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup
SUB=${SUB:-/data2/seonghyeonnoh/LLM-continual-learning-data/router_finetune_miniset_repeats/seed1234/0p1pctx200}
ROOT=${ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/mres_flame_wcc_20260925}
GPUS=${GPUS:-0,1,2,3}; NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
MB_WIKI=${MB_WIKI:-72}; MB_1P=${MB_1P:-64}
REPLAY_SAMPLES=${REPLAY_SAMPLES:-829440}
ITERS=${ITERS:-1800}
PB=${PB:-46200}
MRES_ARGS="--moe-mass-reservoir --mres-alpha-end ${MRES_ALPHA_END:-8} --mres-delta ${MRES_DELTA:-0.5} --mres-lambda ${MRES_LAMBDA:-0.1} --mres-warmup-frac ${MRES_WARMUP_FRAC:-0.05}"

mkdir -p "$ROOT/logs" "$ROOT/scratch"
say(){ printf '[MRES %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$ROOT/logs/chain.log"; }
at(){ [ -f "$1/latest_checkpointed_iteration.txt" ] && tr -d '[:space:]' < "$1/latest_checkpointed_iteration.txt" || echo ""; }
for m in $MB_WIKI $MB_1P; do
  (( 2304 % (m * NPROC) == 0 )) || { say "FAIL: 2304 % (${m}x${NPROC}) != 0"; exit 2; }
  (( REPLAY_SAMPLES % (m * NPROC) == 0 )) || { say "FAIL: replay ${REPLAY_SAMPLES} % (${m}x${NPROC}) != 0"; exit 2; }
done
for t in wiki code conversation; do
  [ -d "$SUB/$t/train" ] || { say "FAIL: missing replay subset $SUB/$t/train"; exit 2; }
done

cd "$P"
export PATH=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin:/usr/bin:/bin
export PYTHON_BIN=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin/python
export FLAME_DATA_ROOT=$FLAME LOCAL_DATASET="$P/.local/dataset"
source "$A/common.sh"
set +e   # common.sh turns on -e; stage failures must reach the FAIL lines below
export TOKENIZER_MODEL=/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6
export FLAME_DATA_ROOT=$FLAME PYTHONNOUSERSITE=1 WANDB_MODE=offline SEED=1234
export PATH=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin:/usr/bin:/bin
export CUDA_VISIBLE_DEVICES=$GPUS NPROC_PER_NODE=$NPROC
export LOCAL_SSD_ROOT="$ROOT/scratch" DIRECT_LOCAL_SAVE=1 NO_SAVE_OPTIM=1 PAUSE_SECONDS=0 STAGE_INPUTS_TO_SCRATCH=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GLOBAL_BATCH_SIZE=2304 TRAIN_ITERS=$ITERS SAVE_INTERVAL=$ITERS EVAL_INTERVAL=$ITERS LOG_INTERVAL=20
export NUM_QUERY_GROUPS=16 MOE_ROUTER_TOPK=4 MOE_FFN_HIDDEN_SIZE=352
export ATTN_LORA_RANK=256 ATTN_LORA_ALPHA=256 ATTN_FULL_RANK_LORA_RANK=256 ATTN_FULL_RANK_LORA_ALPHA=256
export ATTN_FULL_RANK_LORA_TARGETS=qkvo ATTN_FULL_RANK_LORA_ACTIVE_TARGETS=""
export MOE_AUX_LOSS_COEFF=0.01 MOE_Z_LOSS_COEFF=0.001
export MOE_GROUPED_GEMM=1 ATTN_LORA_GROUPED_GEMM=1 MOE_PERMUTE_FUSION=0 MOE_ROUTER_DTYPE=fp32
export MODEL_CONFIG_SCRIPT=configs/model/flame-shared-router-hybrid-experts.sh
export PROBE_EVAL_ITERS=25 SECONDARY_PROBE_EVAL_ITERS=25 TERTIARY_PROBE_EVAL_ITERS=25
export PROBE_EVAL_INTERVAL=50 SECONDARY_PROBE_EVAL_INTERVAL=50 TERTIARY_PROBE_EVAL_INTERVAL=50
export EXTRA_MEGATRON_ARGS="$MRES_ARGS"
WIKI=$ROOT/wiki; CODE=$ROOT/code; CONV=$ROOT/conversation

say "start GPUs $GPUS (w$NPROC) mb wiki$MB_WIKI/1p$MB_1P replay budget $REPLAY_SAMPLES; $MRES_ARGS"

# ---------------------------------------------------------------- 1. wiki (task 0)
if [ "$(at "$WIKI")" != "$ITERS" ]; then
  say "1/3 wiki e8 from scratch + reservoir (replay: wiki subset)"
  env RUN_ID=mres-wiki TRAIN_WEIGHTS="$WIKI" NUM_LAYERS=9 HIDDEN_SIZE=1024 FFN_HIDDEN_SIZE=5472 \
      NUM_EXPERTS=8 MICRO_BATCH_SIZE=$MB_WIKI MASTER_PORT=$((PB+0)) \
      PROBE_DATASET="$(probe_dir_for_task wiki)" PROBE_NAME=wiki_probe \
      SECONDARY_PROBE_DATASET="$(probe_dir_for_task code)" SECONDARY_PROBE_NAME=code_probe \
      TERTIARY_PROBE_DATASET="$(probe_dir_for_task conversation)" TERTIARY_PROBE_NAME=conversation_probe \
      MOE_JOINT_REPLAY_LM=1 JOINT_REPLAY_DATASET="$SUB/wiki/train" JOINT_REPLAY_TOTAL_SAMPLES=$REPLAY_SAMPLES \
      WANDB_PROJECT="" \
    bash "$A/pretrain_wiki_shared_router_hybrid_local_bf16.sh" >> "$ROOT/logs/wiki.log" 2>&1
  [ "$(at "$WIKI")" = "$ITERS" ] || { say "FAIL wiki (see $ROOT/logs/wiki.log, $WIKI/logs/run.log)"; exit 1; }
else say "1/3 wiki done, skip"; fi

# common 1-phase environment (mirrors run_g2_shared_router_*_joint_lm_allrouter_mha_replaymb.sh)
export RESUME_FROM_WEIGHTS="" RESUME_LOAD_OPTIM=0 RESUME_RESET_ITERATION=1
export SHARED_ROUTER_HYBRID_TRAIN_ALL_EXPERTS_AND_ROUTER=0 SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS=1
export SHARED_ROUTER_HYBRID_FREEZE_PREEXISTING_ONLY=0 SHARED_ROUTER_HYBRID_PARTIAL_FREEZE_MASK=""
export SHARED_ROUTER_HYBRID_TOPK_WITH_ALL_NEW_EXPERTS=0
export MOE_JOINT_REPLAY_LM=1 JOINT_REPLAY_DATA_WEIGHT_MODE=equal_dataset JOINT_REPLAY_TOTAL_SAMPLES=$REPLAY_SAMPLES
export ENABLE_OLD_MODEL_KL=0 OLD_MODEL_KL_COEFF=0.0 OLD_MODEL_KL_TEMPERATURE=1.0
export MOE_EXPANSION_DISTILL_MODE=none ROUTER_MEMORY_KL_COEFF=0.0 ROUTER_MEMORY_INTERVAL=0
export MOE_NEW_EXPERT_LR_RAMP_STEPS=0 MICRO_BATCH_SIZE=$MB_1P
export LR_DECAY_ITERS=$ITERS LR_WSD_DECAY_ITERS=$((ITERS / 10)) LR_WARMUP_FRACTION=0.01
export TRAIN_LOG_STEP_TIME_ONLY=0 PROBE_MICRO_BATCH_SIZE=32 RUN_INITIAL_PROBE_EVAL=1 RUN_INITIAL_VALID_EVAL=1
export SOURCE_REQUIRED_ITERS=$ITERS LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND=1
export WANDB_PROJECT=""

# ---------------------------------------------------------------- 2. code (task 1)
if [ "$(at "$CODE")" != "$ITERS" ]; then
  say "2/3 code: expand 8->16 from wiki (reservoir copy, no KD), 1-phase $ITERS (replay: wiki+code subsets)"
  env RUN_ID=mres-code TRAIN_WEIGHTS="$CODE" STAGE1_WEIGHTS_DIR="$WIKI" \
      SOURCE_NUM_EXPERTS=8 NUM_EXPERTS=16 MASTER_PORT=$((PB+1)) \
      TRAIN_DATASET="$(dataset_dir_for_task code)" DATASET_NAME=code_train_mres_replay \
      DATASET_SOURCE="Code primary LM + wiki/code router-only replay (mass reservoir)" \
      JOINT_REPLAY_DATASET="$SUB/wiki/train" JOINT_REPLAY_SECONDARY_DATASET="$SUB/code/train" \
      JOINT_REPLAY_TERTIARY_DATASET="" \
      PROBE_DATASET="$(probe_dir_for_task code)" PROBE_NAME=code_probe \
      SECONDARY_PROBE_DATASET="$(probe_dir_for_task wiki)" SECONDARY_PROBE_NAME=wiki_probe \
      TERTIARY_PROBE_DATASET="$(probe_dir_for_task conversation)" TERTIARY_PROBE_NAME=conversation_probe \
      PROBE_STEP_OFFSET=$ITERS SECONDARY_PROBE_STEP_OFFSET=$ITERS TERTIARY_PROBE_STEP_OFFSET=$ITERS \
      WANDB_STEP_OFFSET=$ITERS \
    bash "$P/scripts/experiment/continual_shared_router_hybrid_replaymb_local_bf16.sh" >> "$ROOT/logs/code.log" 2>&1
  [ "$(at "$CODE")" = "$ITERS" ] || { say "FAIL code (see $ROOT/logs/code.log)"; exit 1; }
else say "2/3 code done, skip"; fi

# ---------------------------------------------------------------- 3. conversation (task 2)
if [ "$(at "$CONV")" != "$ITERS" ]; then
  say "3/3 conv: expand 16->24 from code (reservoir copy, no KD), 1-phase $ITERS (replay: wiki+code+conv subsets)"
  env RUN_ID=mres-conv TRAIN_WEIGHTS="$CONV" STAGE1_WEIGHTS_DIR="$CODE" \
      SOURCE_NUM_EXPERTS=16 NUM_EXPERTS=24 MASTER_PORT=$((PB+2)) \
      TRAIN_DATASET="$(dataset_dir_for_task conversation)" DATASET_NAME=conversation_train_mres_replay \
      DATASET_SOURCE="Conversation primary LM + wiki/code/conv router-only replay (mass reservoir)" \
      JOINT_REPLAY_DATASET="$SUB/wiki/train" JOINT_REPLAY_SECONDARY_DATASET="$SUB/code/train" \
      JOINT_REPLAY_TERTIARY_DATASET="$SUB/conversation/train" \
      PROBE_DATASET="$(probe_dir_for_task conversation)" PROBE_NAME=conversation_probe \
      SECONDARY_PROBE_DATASET="$(probe_dir_for_task wiki)" SECONDARY_PROBE_NAME=wiki_probe \
      TERTIARY_PROBE_DATASET="$(probe_dir_for_task code)" TERTIARY_PROBE_NAME=code_probe \
      PROBE_STEP_OFFSET=$((2 * ITERS)) SECONDARY_PROBE_STEP_OFFSET=$((2 * ITERS)) \
      TERTIARY_PROBE_STEP_OFFSET=$((2 * ITERS)) WANDB_STEP_OFFSET=$((2 * ITERS)) \
    bash "$P/scripts/experiment/continual_shared_router_hybrid_replaymb_local_bf16.sh" >> "$ROOT/logs/conv.log" 2>&1
  [ "$(at "$CONV")" = "$ITERS" ] || { say "FAIL conv (see $ROOT/logs/conv.log)"; exit 1; }
else say "3/3 conv done, skip"; fi

say "DONE wiki=$(at "$WIKI") code=$(at "$CODE") conv=$(at "$CONV")"
