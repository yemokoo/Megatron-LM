#!/usr/bin/env bash
# Router-FT loss ablation on the mass-reservoir hybrid chain (wiki -> code -> conv).
#
# Everything is run_mres_wcc_chain.sh (reservoir, expansion, warm-up, alpha refill, margin, primary
# pass, LR schedule, budgets, seed, order, and the replay set: code replays wiki+code, conv replays
# wiki+code+conv) except the loss of the router-only replay pass on the code/conv stages:
#   REPLAY_SPLIT=1 (default)  old-task replay samples get the arm's objective, current-task samples
#               keep the LM loss (--moe-joint-replay-current-task-dataset-id: code=1, conv=2), so
#               the lm arm is exactly run_mres_wcc_chain.sh and the KD arms change only the loss on
#               old data.  REPLAY_SPLIT=0: the previous design -- old tasks only, all of them KD.
#   replay loss on old-task samples, one of (ARM=)
#     lm        LM loss on the replay tokens (control: the original objective)
#     routerkl  per-layer router KL to the pre-expansion model, no LM: at every router layer
#               KL(softmax(teacher gating) || softmax(student gating)) over the experts, the teacher
#               zero-padded on the new experts, averaged over layers (--moe-joint-replay-old-data-router-kl)
#     logitkd   final-logit KL to the pre-expansion model, no LM (--moe-joint-replay-old-data-kd)
#     hiddenkl  layer-wise feature KL to the pre-expansion model, no LM: per layer
#               KL(softmax(h_teacher/T) || softmax(h_student/T)) over the hidden dim, averaged over
#               HKL_LAYERS (--moe-joint-replay-old-data-hidden-kl)
# The teacher is the pre-expansion source model (wiki for code, this arm's code for conv), frozen,
# forwarded at alpha = alpha_end (mass_reservoir.teacher_alpha).  The reservoir margin stays on the
# replay pass in every arm.  Replay gradients stay router-only (non-router grads are restored).
#
# The wiki stage does not depend on the arm and is shared: the finished wiki of
# run_mres_wcc_chain.sh (mres_flame_wcc_20260925/wiki) copied to ROOT/wiki, or WIKI=<dir>.
# It needs latest_checkpointed_iteration.txt (= ITERS), iter_0001800/ and logs/run_metadata.json.
# A missing wiki is an error; TRAIN_WIKI=1 trains it here instead (flock-guarded).
#
#   bash scripts/experiment/a100/mres_flame/run_mres_wcc_routerft_loss.sh        # ARMS in series, 8 GPUs
#   ARMS="logitkd" bash ...                                                        # a subset / other order
#   ARM=lm GPUS=0,1,2,3 MB_1P=64 REPLAY_MB=64 PB=46220 bash ...                    # one arm
#   SMOKE=1 bash ...        # all arms, code+conv at ITERS=20 into ROOT=..._smoke, same wiki, same replay ratio
set -uo pipefail

if [ -z "${ARM:-}" ]; then          # serial: each arm on all GPUS, first finished arm first
  for a in ${ARMS:-routerkl hiddenkl}; do
    ARM=$a bash "${BASH_SOURCE[0]}" || exit $?
  done
  exit 0
fi

RUNS=/data2/seonghyeonnoh/LLM-continual-learning-runs
if [ "${SMOKE:-0}" = "1" ]; then
  # 20 updates per stage; 9216 = 24 replay micro-batches of 48x8, the full run's 2160/1800 per update
  : "${ITERS:=20}" "${REPLAY_SAMPLES:=9216}" "${LOG_INTERVAL:=1}" "${PROBE_INTERVAL:=10}" "${PROBE_ITERS:=5}"
  : "${ROOT:=$RUNS/mres_flame_wcc_routerft_loss_smoke}" "${WIKI:=$RUNS/mres_flame_wcc_routerft_loss/wiki}"
fi
P=${P:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
A=$P/scripts/experiment/a100
FLAME=/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup
SUB=${SUB:-/data2/seonghyeonnoh/LLM-continual-learning-data/router_finetune_miniset_repeats/seed1234/0p1pctx200}
ROOT=${ROOT:-$RUNS/mres_flame_wcc_routerft_loss}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}; NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
MB_WIKI=${MB_WIKI:-72}; MB_1P=${MB_1P:-48}   # 2304 = 48 x 8 x 6 on 8 GPUs
REPLAY_MB=${REPLAY_MB:-$MB_1P}   # keep equal across arms; lower it for all arms if logitkd OOMs
REPLAY_SAMPLES=${REPLAY_SAMPLES:-829440}
ITERS=${ITERS:-1800}
WIKI_ITERS=${WIKI_ITERS:-1800}
PB=${PB:-46200}
MRES_ARGS="--moe-mass-reservoir --mres-alpha-end ${MRES_ALPHA_END:-8} --mres-delta ${MRES_DELTA:-0.5} --mres-lambda ${MRES_LAMBDA:-0.1} --mres-warmup-frac ${MRES_WARMUP_FRAC:-0.05}"

case "$ARM" in
  lm)
    ARM_ENV=(ENABLE_OLD_MODEL_KL=0 OLD_MODEL_KL_COEFF=0.0)
    ARM_ARGS="" ;;
  routerkl)
    # coeff 0 on the logit KL: ENABLE_OLD_MODEL_KL=1 only passes --moe-old-model-kl-load
    ARM_ENV=(ENABLE_OLD_MODEL_KL=1 OLD_MODEL_KL_COEFF=0.0)
    ARM_ARGS="--moe-joint-replay-old-data-router-kl --moe-old-router-kl-coeff ${RKL_COEFF:-1.0}" ;;
  logitkd)
    ARM_ENV=(ENABLE_OLD_MODEL_KL=1 OLD_MODEL_KL_COEFF=${KD_COEFF:-1.0} OLD_MODEL_KL_TEMPERATURE=${KD_T:-1.0})
    ARM_ARGS="--moe-joint-replay-old-data-kd" ;;
  hiddenkl)
    # coeff 0 on the logit KL: ENABLE_OLD_MODEL_KL=1 only passes --moe-old-model-kl-load
    ARM_ENV=(ENABLE_OLD_MODEL_KL=1 OLD_MODEL_KL_COEFF=0.0)
    ARM_ARGS="--moe-joint-replay-old-data-hidden-kl --moe-old-hidden-kl-coeff ${HKL_COEFF:-1.0} --moe-old-hidden-kl-temperature ${HKL_T:-1.0} --moe-old-hidden-kl-layers ${HKL_LAYERS:-all}" ;;
  *) echo "unknown ARM=$ARM (lm|routerkl|hiddenkl|logitkd)" >&2; exit 2 ;;
esac

REPLAY_SPLIT=${REPLAY_SPLIT:-1}
if [ "$REPLAY_SPLIT" = 1 ]; then
  CODE_SECONDARY="$SUB/code/train"; CONV_TERTIARY="$SUB/conversation/train"
  if [ "$ARM" = lm ]; then SPLIT_CODE=""; SPLIT_CONV=""
  else SPLIT_CODE="--moe-joint-replay-current-task-dataset-id 1"; SPLIT_CONV="--moe-joint-replay-current-task-dataset-id 2"; fi
else
  CODE_SECONDARY=""; CONV_TERTIARY=""; SPLIT_CODE=""; SPLIT_CONV=""
fi
WIKI=${WIKI:-$ROOT/wiki}; CODE=$ROOT/$ARM/code; CONV=$ROOT/$ARM/conversation
mkdir -p "$ROOT/logs" "$ROOT/$ARM/scratch"
say(){ printf '[MRES-%s %s] %s\n' "$ARM" "$(date '+%F %T')" "$*" | tee -a "$ROOT/logs/chain.log"; }
at(){ [ -f "$1/latest_checkpointed_iteration.txt" ] && tr -d '[:space:]' < "$1/latest_checkpointed_iteration.txt" || echo ""; }
(( 2304 % (MB_WIKI * NPROC) == 0 )) || { say "FAIL: 2304 % (${MB_WIKI}x${NPROC}) != 0"; exit 2; }
(( 2304 % (MB_1P * NPROC) == 0 )) || { say "FAIL: 2304 % (${MB_1P}x${NPROC}) != 0"; exit 2; }
(( REPLAY_SAMPLES % (MB_WIKI * NPROC) == 0 )) || { say "FAIL: replay ${REPLAY_SAMPLES} % (${MB_WIKI}x${NPROC}) != 0"; exit 2; }
(( REPLAY_SAMPLES % (REPLAY_MB * NPROC) == 0 )) || { say "FAIL: replay ${REPLAY_SAMPLES} % (${REPLAY_MB}x${NPROC}) != 0"; exit 2; }
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
export LOCAL_SSD_ROOT="$ROOT/$ARM/scratch" DIRECT_LOCAL_SAVE=1 NO_SAVE_OPTIM=1 PAUSE_SECONDS=0 STAGE_INPUTS_TO_SCRATCH=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GLOBAL_BATCH_SIZE=2304 TRAIN_ITERS=$ITERS SAVE_INTERVAL=$ITERS EVAL_INTERVAL=$ITERS LOG_INTERVAL=${LOG_INTERVAL:-20}
export NUM_QUERY_GROUPS=16 MOE_ROUTER_TOPK=4 MOE_FFN_HIDDEN_SIZE=352
export ATTN_LORA_RANK=256 ATTN_LORA_ALPHA=256 ATTN_FULL_RANK_LORA_RANK=256 ATTN_FULL_RANK_LORA_ALPHA=256
export ATTN_FULL_RANK_LORA_TARGETS=qkvo ATTN_FULL_RANK_LORA_ACTIVE_TARGETS=""
export MOE_AUX_LOSS_COEFF=0.01 MOE_Z_LOSS_COEFF=0.001
export MOE_GROUPED_GEMM=1 ATTN_LORA_GROUPED_GEMM=1 MOE_PERMUTE_FUSION=0 MOE_ROUTER_DTYPE=fp32
export MODEL_CONFIG_SCRIPT=configs/model/flame-shared-router-hybrid-experts.sh
PI=${PROBE_INTERVAL:-50}; PN=${PROBE_ITERS:-25}
export PROBE_EVAL_ITERS=$PN SECONDARY_PROBE_EVAL_ITERS=$PN TERTIARY_PROBE_EVAL_ITERS=$PN
export PROBE_EVAL_INTERVAL=$PI SECONDARY_PROBE_EVAL_INTERVAL=$PI TERTIARY_PROBE_EVAL_INTERVAL=$PI

say "start${SMOKE:+ SMOKE} iters $ITERS GPUS $GPUS (w$NPROC) mb wiki$MB_WIKI/1p$MB_1P/replay$REPLAY_MB replay budget $REPLAY_SAMPLES; $MRES_ARGS | ${ARM_ENV[*]} $ARM_ARGS"

# ---------------------------------------------------------------- 1. wiki (task 0, shared by all arms)
# identical to run_mres_wcc_chain.sh: nothing is expanded yet, so there is no router-FT loss to vary;
# the wiki-subset replay pass is what trains the reservoir margin.
[ "$(at "$WIKI")" = "$WIKI_ITERS" ] || [ "${TRAIN_WIKI:-0}" = "1" ] || {
  say "FAIL: no finished wiki at $WIKI (copy mres_flame_wcc_20260925/wiki there, set WIKI=, or TRAIN_WIKI=1)"; exit 2; }
[ "$(at "$WIKI")" != "$WIKI_ITERS" ] || [ -f "$WIKI/logs/run_metadata.json" ] || {
  say "FAIL: $WIKI/logs/run_metadata.json missing (the 1-phase runner reads train_iters from it)"; exit 2; }
(
  flock 9
  if [ "$(at "$WIKI")" != "$WIKI_ITERS" ]; then
    say "1/3 wiki e8 from scratch + reservoir (replay: wiki subset) -> $WIKI"
    env RUN_ID=mres-wiki TRAIN_WEIGHTS="$WIKI" TRAIN_ITERS=$WIKI_ITERS SAVE_INTERVAL=$WIKI_ITERS EVAL_INTERVAL=$WIKI_ITERS NUM_LAYERS=9 HIDDEN_SIZE=1024 FFN_HIDDEN_SIZE=5472 \
        NUM_EXPERTS=8 MICRO_BATCH_SIZE=$MB_WIKI MASTER_PORT=$((PB+0)) \
        LOCAL_SSD_ROOT="$ROOT/scratch_wiki" \
        PROBE_DATASET="$(probe_dir_for_task wiki)" PROBE_NAME=wiki_probe \
        SECONDARY_PROBE_DATASET="$(probe_dir_for_task code)" SECONDARY_PROBE_NAME=code_probe \
        TERTIARY_PROBE_DATASET="$(probe_dir_for_task conversation)" TERTIARY_PROBE_NAME=conversation_probe \
        MOE_JOINT_REPLAY_LM=1 JOINT_REPLAY_DATASET="$SUB/wiki/train" JOINT_REPLAY_TOTAL_SAMPLES=$REPLAY_SAMPLES \
        EXTRA_MEGATRON_ARGS="$MRES_ARGS" WANDB_PROJECT="" \
      bash "$A/pretrain_wiki_shared_router_hybrid_local_bf16.sh" >> "$ROOT/logs/wiki.log" 2>&1
  else say "1/3 wiki at $WIKI_ITERS, reuse $WIKI"; fi
) 9> "$ROOT/wiki.lock"
[ "$(at "$WIKI")" = "$WIKI_ITERS" ] || { say "FAIL wiki (see $ROOT/logs/wiki.log, $WIKI/logs/run.log)"; exit 1; }

# common 1-phase environment (as run_mres_wcc_chain.sh)
export RESUME_FROM_WEIGHTS="" RESUME_LOAD_OPTIM=0 RESUME_RESET_ITERATION=1
export SHARED_ROUTER_HYBRID_TRAIN_ALL_EXPERTS_AND_ROUTER=0 SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS=1
export SHARED_ROUTER_HYBRID_FREEZE_PREEXISTING_ONLY=0 SHARED_ROUTER_HYBRID_PARTIAL_FREEZE_MASK=""
export SHARED_ROUTER_HYBRID_TOPK_WITH_ALL_NEW_EXPERTS=0
export MOE_JOINT_REPLAY_LM=1 JOINT_REPLAY_DATA_WEIGHT_MODE=equal_dataset JOINT_REPLAY_TOTAL_SAMPLES=$REPLAY_SAMPLES
export JOINT_REPLAY_MICRO_BATCH_SIZE=$REPLAY_MB
export OLD_MODEL_KL_TEMPERATURE=1.0
export MOE_EXPANSION_DISTILL_MODE=none ROUTER_MEMORY_KL_COEFF=0.0 ROUTER_MEMORY_INTERVAL=0
export MOE_NEW_EXPERT_LR_RAMP_STEPS=0 MICRO_BATCH_SIZE=$MB_1P
export LR_DECAY_ITERS=$ITERS LR_WSD_DECAY_ITERS=$((ITERS / 10)) LR_WARMUP_FRACTION=0.01
export TRAIN_LOG_STEP_TIME_ONLY=0 PROBE_MICRO_BATCH_SIZE=32 RUN_INITIAL_PROBE_EVAL=1 RUN_INITIAL_VALID_EVAL=1
export SOURCE_REQUIRED_ITERS=$ITERS LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND=1
export WANDB_PROJECT=""
export EXTRA_MEGATRON_ARGS="$MRES_ARGS $ARM_ARGS"
export "${ARM_ENV[@]}"

# ---------------------------------------------------------------- 2. code (task 1)
if [ "$(at "$CODE")" != "$ITERS" ]; then
  say "2/3 code: expand 8->16 from wiki, 1-phase $ITERS; router-FT replay = wiki${CODE_SECONDARY:++code} (split=$REPLAY_SPLIT), loss $ARM"
  env RUN_ID=mres-$ARM-code TRAIN_WEIGHTS="$CODE" STAGE1_WEIGHTS_DIR="$WIKI" \
      SOURCE_NUM_EXPERTS=8 NUM_EXPERTS=16 MASTER_PORT=$((PB+1)) \
      TRAIN_DATASET="$(dataset_dir_for_task code)" DATASET_NAME=code_train_mres_rft_$ARM \
      DATASET_SOURCE="Code primary LM + wiki router-only replay ($ARM, mass reservoir)" \
      JOINT_REPLAY_DATASET="$SUB/wiki/train" JOINT_REPLAY_SECONDARY_DATASET="$CODE_SECONDARY" \
      JOINT_REPLAY_TERTIARY_DATASET="" EXTRA_MEGATRON_ARGS="$MRES_ARGS $ARM_ARGS $SPLIT_CODE" \
      PROBE_DATASET="$(probe_dir_for_task code)" PROBE_NAME=code_probe \
      SECONDARY_PROBE_DATASET="$(probe_dir_for_task wiki)" SECONDARY_PROBE_NAME=wiki_probe \
      TERTIARY_PROBE_DATASET="$(probe_dir_for_task conversation)" TERTIARY_PROBE_NAME=conversation_probe \
      PROBE_STEP_OFFSET=$WIKI_ITERS SECONDARY_PROBE_STEP_OFFSET=$WIKI_ITERS TERTIARY_PROBE_STEP_OFFSET=$WIKI_ITERS \
      WANDB_STEP_OFFSET=$WIKI_ITERS \
    bash "$P/scripts/experiment/continual_shared_router_hybrid_replaymb_local_bf16.sh" >> "$ROOT/logs/$ARM.code.log" 2>&1
  [ "$(at "$CODE")" = "$ITERS" ] || { say "FAIL code (see $ROOT/logs/$ARM.code.log)"; exit 1; }
else say "2/3 code done, skip"; fi

# STOP_AFTER=code: stop each arm after its code stage (run the same command again without it later
# to add conv -- wiki and code are then skipped)
if [ "${STOP_AFTER:-}" = code ]; then say "STOP_AFTER=code: conv not run (code=$(at "$CODE"))"; exit 0; fi

# ---------------------------------------------------------------- 3. conversation (task 2)
if [ "$(at "$CONV")" != "$ITERS" ]; then
  say "3/3 conv: expand 16->24 from $ARM code, 1-phase $ITERS; router-FT replay = wiki+code${CONV_TERTIARY:++conv} (split=$REPLAY_SPLIT), loss $ARM"
  env RUN_ID=mres-$ARM-conv TRAIN_WEIGHTS="$CONV" STAGE1_WEIGHTS_DIR="$CODE" \
      SOURCE_NUM_EXPERTS=16 NUM_EXPERTS=24 MASTER_PORT=$((PB+2)) \
      TRAIN_DATASET="$(dataset_dir_for_task conversation)" DATASET_NAME=conversation_train_mres_rft_$ARM \
      DATASET_SOURCE="Conversation primary LM + wiki/code router-only replay ($ARM, mass reservoir)" \
      JOINT_REPLAY_DATASET="$SUB/wiki/train" JOINT_REPLAY_SECONDARY_DATASET="$SUB/code/train" \
      JOINT_REPLAY_TERTIARY_DATASET="$CONV_TERTIARY" EXTRA_MEGATRON_ARGS="$MRES_ARGS $ARM_ARGS $SPLIT_CONV" \
      PROBE_DATASET="$(probe_dir_for_task conversation)" PROBE_NAME=conversation_probe \
      SECONDARY_PROBE_DATASET="$(probe_dir_for_task wiki)" SECONDARY_PROBE_NAME=wiki_probe \
      TERTIARY_PROBE_DATASET="$(probe_dir_for_task code)" TERTIARY_PROBE_NAME=code_probe \
      PROBE_STEP_OFFSET=$((WIKI_ITERS + ITERS)) SECONDARY_PROBE_STEP_OFFSET=$((WIKI_ITERS + ITERS)) \
      TERTIARY_PROBE_STEP_OFFSET=$((WIKI_ITERS + ITERS)) WANDB_STEP_OFFSET=$((WIKI_ITERS + ITERS)) \
    bash "$P/scripts/experiment/continual_shared_router_hybrid_replaymb_local_bf16.sh" >> "$ROOT/logs/$ARM.conv.log" 2>&1
  [ "$(at "$CONV")" = "$ITERS" ] || { say "FAIL conv (see $ROOT/logs/$ARM.conv.log)"; exit 1; }
else say "3/3 conv done, skip"; fi

say "DONE wiki=$(at "$WIKI") code=$(at "$CODE") conv=$(at "$CONV")"
