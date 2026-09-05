#!/bin/bash
# Two-phase G2 chain, one representative run per call:
#   run_2phase_chain.sh <ffn|hyb> <kd0|kd1>
#   wiki e8 (HF) -> [code KD init] -> code phase-1 (new experts + all router rows, 1800)
#                -> code router FT (router only; wiki+code 0.1% x 200 epochs, 720)
#                -> [conv KD init] -> conv phase-1 (1800)
#                -> conv router FT (wiki+code+conv 0.1% x 200 epochs, 1080)
# Stock stage scripts / launchers are called unmodified (env only).  Reused sources:
#   hyb kd0 code phase-1 = moe_lpr_hybrid_from_hf_e8_20260826/code_task (same recipe),
#   hyb kd1 code KD init = HF kd_1phase/ffn_attn_shared_router/code/kd_init.
# SMOKE=1: 3 iterations per stage, gbs 64, separate root, 1 GPU.
set -euo pipefail
VARIANT="${1:?ffn|hyb}"; KD="${2:?kd0|kd1}"
A="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; P="$(cd "$A/../../.." && pwd)"; cd "$P"
HF=/data2/seonghyeonnoh/LLM-continual-learning-runs/hf_g2_wiki_code_conversation/g2_wiki_code_conversation
FLAME=/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup
SUB=/data2/seonghyeonnoh/LLM-continual-learning-data/router_finetune_miniset_repeats/seed1234/0p1pctx200
SMOKE="${SMOKE:-0}"
ROOT="${ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/g2_2phase_4runs_20260827$([ "$SMOKE" = 1 ] && echo _smoke || true)/${VARIANT}_${KD}}"
GPUS="${GPUS:-0,1}"; PORT_BASE="${PORT_BASE:-29800}"
export FLAME_DATA_ROOT="$FLAME" LOCAL_SSD_ROOT="$ROOT/scratch" LOCAL_WEIGHTS="$ROOT/weights" G2_ROOT="$ROOT/g2" LOCAL_BASE="$ROOT/local"
export CUDA_VISIBLE_DEVICES="$GPUS" NPROC_PER_NODE="$(awk -F, '{print NF}' <<< "$GPUS")"
export DIRECT_LOCAL_SAVE=1 WANDB_MODE=offline PAUSE_SECONDS=0 PYTHONNOUSERSITE=1
export PATH=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin:/usr/bin:/bin
export TOKENIZER_MODEL=/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6
export NUM_LAYERS=9 HIDDEN_SIZE=1024 FFN_HIDDEN_SIZE=5472 NUM_QUERY_GROUPS=16 MOE_FFN_HIDDEN_SIZE=352 MOE_LAYER_FREQ="[0]*1+[1]*8" MOE_ROUTER_TOPK=4 MOE_ROUTER_DTYPE=fp32 MOE_GROUPED_GEMM=1
if [ "$VARIANT" = hyb ]; then
  export ATTN_LORA_RANK=256 ATTN_LORA_ALPHA=256 ATTN_FULL_RANK_LORA_RANK=256 ATTN_FULL_RANK_LORA_ALPHA=256 ATTN_FULL_RANK_LORA_TARGETS=qkvo ATTN_LORA_GROUPED_GEMM=1
  WIKI="$HF/sources/ffn_attn_shared_router/wiki"
else
  WIKI="$HF/sources/ffn_only/wiki"
fi
export NO_SAVE_OPTIM=1 LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND=0   # HF-downloaded sources have no tfevents
# ---- iteration / batch plan -------------------------------------------------------
if [ "$SMOKE" = 1 ]; then
  IT_P1=3; IT_RFT_CODE=3; IT_RFT_CONV=3; IT_KD=3; export GLOBAL_BATCH_SIZE=64
  MB_P1=32; MB_KD=32; MB_RFT=32
  export PROBE_EVAL_ITERS=2 SECONDARY_PROBE_EVAL_ITERS=2 TERTIARY_PROBE_EVAL_ITERS=2 PROBE_EVAL_INTERVAL=3 SECONDARY_PROBE_EVAL_INTERVAL=3 TERTIARY_PROBE_EVAL_INTERVAL=3
else
  IT_P1=1800; IT_RFT_CODE=720; IT_RFT_CONV=1080; IT_KD=600; export GLOBAL_BATCH_SIZE=2304
  if [ "$VARIANT" = hyb ]; then MB_P1="${MB_P1:-72}"; MB_KD="${MB_KD:-36}"; MB_RFT="${MB_RFT:-72}"; else MB_P1="${MB_P1:-96}"; MB_KD="${MB_KD:-96}"; MB_RFT="${MB_RFT:-96}"; fi
  export PROBE_EVAL_ITERS=25 SECONDARY_PROBE_EVAL_ITERS=25 TERTIARY_PROBE_EVAL_ITERS=25 PROBE_EVAL_INTERVAL=50 SECONDARY_PROBE_EVAL_INTERVAL=50 TERTIARY_PROBE_EVAL_INTERVAL=50
fi
export PROBE_MICRO_BATCH_SIZE=32
CODE_KD="$ROOT/code_kd_init"; CODE_P1="$ROOT/code_task"; CODE_RFT="$ROOT/code_router_ft"
CONV_KD="$ROOT/conv_kd_init"; CONV_P1="$ROOT/conv_task"; CONV_RFT="$ROOT/conv_router_ft"
latest() { [ -f "$1/latest_checkpointed_iteration.txt" ] && tr -d '[:space:]' < "$1/latest_checkpointed_iteration.txt" || echo ""; }
say() { printf '[2P %s %s/%s] %s\n' "$(date '+%F %T')" "$VARIANT" "$KD" "$*"; }
need() { [ "$(latest "$1")" = "$2" ] || { say "FAILED: $1 is at '$(latest "$1")', expected $2"; exit 1; }; }
stub_meta() {  # dir step -- launchers read logs/run_metadata.json['train_iters'] from a source dir
  python3 - "$1" "$2" <<'PY'
import json, os, sys
d, step = sys.argv[1], int(sys.argv[2]); p = os.path.join(d, "logs", "run_metadata.json"); os.makedirs(os.path.dirname(p), exist_ok=True)
m = json.load(open(p)) if os.path.isfile(p) else {}
m.setdefault("train_iters", step); json.dump(m, open(p, "w"), indent=1)
PY
}
prepare_copy() {  # src dst -- router FT trains in place on a copy (never touches the phase-1 checkpoint)
  if [ ! -f "$2/latest_checkpointed_iteration.txt" ]; then
    mkdir -p "$2"; rsync -aH --exclude wandb/ "$1/" "$2/"
    printf 'source=%s\nsource_step=%s\nstage=router_ft\n' "$1" "$(latest "$1")" > "$2/PHASE3_SOURCE.txt"
  fi
}
common_probes_code=(PROBE_DATASET="$FLAME/code/test" PROBE_NAME=code_probe SECONDARY_PROBE_DATASET="$FLAME/wiki/test" SECONDARY_PROBE_NAME=wiki_probe)
common_probes_conv=(PROBE_DATASET="$FLAME/conversation/test" PROBE_NAME=conversation_probe SECONDARY_PROBE_DATASET="$FLAME/wiki/test" SECONDARY_PROBE_NAME=wiki_probe TERTIARY_PROBE_DATASET="$FLAME/code/test" TERTIARY_PROBE_NAME=code_probe)
mkdir -p "$ROOT/scratch" "$ROOT/weights" "$ROOT/g2" "$ROOT/logs" "$ROOT/local"
[ -f "$WIKI/latest_checkpointed_iteration.txt" ] || { say "wiki e8 missing: $WIKI"; exit 1; }
stub_meta "$WIKI" 1800
for t in wiki code conversation; do [ -s "$SUB/$t/train/train_text_document.bin" ] || { say "missing 0.1% subset: $SUB/$t"; exit 1; }; done
say "root=$ROOT gpus=$GPUS smoke=$SMOKE wiki=$WIKI"
# =====================================================================================
# stage 1: code KD init (kd1 only)
# =====================================================================================
if [ "$KD" = kd1 ]; then
  if [ "$VARIANT" = hyb ] && [ "$SMOKE" != 1 ] && [ "$(latest "$HF/kd_1phase/ffn_attn_shared_router/code/kd_init")" = 600 ]; then
    CODE_KD="$HF/kd_1phase/ffn_attn_shared_router/code/kd_init"; stub_meta "$CODE_KD" 600; say "code KD init: reuse HF ($(latest "$CODE_KD"))"
  elif [ "$(latest "$CODE_KD")" != "$IT_KD" ]; then
    say "code KD init ($IT_KD it)"
    if [ "$VARIANT" = hyb ]; then
      env STAGE1_WEIGHTS_DIR="$WIKI" SOURCE_REQUIRED_ITERS=1800 TRAIN_ITERS=$IT_KD SAVE_INTERVAL=$IT_KD MICRO_BATCH_SIZE=$MB_KD \
          RUN_ID=2p-${VARIANT}-${KD}-code-kd-init TRAIN_WEIGHTS="$CODE_KD" MASTER_PORT=$((PORT_BASE+1)) \
          bash "$A/run_g2_shared_router_code_expert_distill_init_mha.sh" logits
    else
      env SOURCE_WEIGHTS_DIR="$WIKI" SOURCE_REQUIRED_ITERS=1800 TRAIN_ITERS=$IT_KD SAVE_INTERVAL=$IT_KD MICRO_BATCH_SIZE=$MB_KD \
          RUN_ID=2p-${VARIANT}-${KD}-code-kd-init TRAIN_WEIGHTS="$CODE_KD" MASTER_PORT=$((PORT_BASE+1)) \
          bash "$A/run_g2_ffn_only_code_expert_distill_init_mha.sh" logits
    fi
    need "$CODE_KD" "$IT_KD"
  else say "code KD init: skip (done)"; fi
fi
# =====================================================================================
# stage 2: code phase-1  (new experts 8:16 + all router rows, code LM only)
# =====================================================================================
LPR_CODE=/data2/seonghyeonnoh/LLM-continual-learning-runs/moe_lpr_hybrid_from_hf_e8_20260826/code_task
if [ "$VARIANT" = hyb ] && [ "$KD" = kd0 ] && [ "$SMOKE" != 1 ] && [ "$(latest "$LPR_CODE")" = "$IT_P1" ]; then
  CODE_P1="$LPR_CODE"; say "code phase-1: reuse LPR hybrid code_task ($(latest "$CODE_P1"))"
elif [ "$(latest "$CODE_P1")" != "$IT_P1" ]; then
  say "code phase-1 ($IT_P1 it)"
  if [ "$VARIANT" = hyb ]; then
    resume=(); [ "$KD" = kd1 ] && resume=(RESUME_FROM_WEIGHTS="$CODE_KD" RESUME_LOAD_OPTIM=0 RESUME_RESET_ITERATION=1)
    env STAGE1_WEIGHTS_DIR="$WIKI" SOURCE_REQUIRED_ITERS=1800 SOURCE_NUM_EXPERTS=8 NUM_EXPERTS=16 "${resume[@]}" \
        TRAIN_DATASET="$FLAME/code/train" DATASET_NAME=code_exact DATASET_SOURCE="Python code exact train" "${common_probes_code[@]}" \
        MOE_JOINT_REPLAY_LM=0 MOE_EXPANSION_DISTILL_MODE=none SHARED_ROUTER_HYBRID_TRAIN_ALL_EXPERTS_AND_ROUTER=0 SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS=1 \
        ROUTER_MEMORY_KL_COEFF=0.0 ROUTER_MEMORY_INTERVAL=0 \
        RUN_ID=2p-${VARIANT}-${KD}-code-task TRAIN_WEIGHTS="$CODE_P1" TRAIN_ITERS=$IT_P1 SAVE_INTERVAL=$IT_P1 MICRO_BATCH_SIZE=$MB_P1 MASTER_PORT=$((PORT_BASE+2)) \
        bash scripts/experiment/continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh
  elif [ "$KD" = kd1 ]; then
    env SOURCE_WEIGHTS_DIR="$CODE_KD" TRAIN_ITERS=$IT_P1 SAVE_INTERVAL=$IT_P1 MICRO_BATCH_SIZE=$MB_P1 \
        RUN_ID=2p-${VARIANT}-${KD}-code-task TRAIN_WEIGHTS="$CODE_P1" MASTER_PORT=$((PORT_BASE+2)) \
        bash "$A/run_g2_ffn_only_code_from_distill_init_mha.sh" logits
  else
    env SOURCE_WEIGHTS_DIR="$WIKI" SOURCE_REQUIRED_ITERS=1800 TRAIN_ITERS=$IT_P1 SAVE_INTERVAL=$IT_P1 MICRO_BATCH_SIZE=$MB_P1 \
        RUN_ID=2p-${VARIANT}-${KD}-code-task TRAIN_WEIGHTS="$CODE_P1" MASTER_PORT=$((PORT_BASE+2)) \
        bash "$A/code_from_wiki_ffn_moe_g2matched_attn_freeze_mha_a100_bf16.sh"
  fi
  need "$CODE_P1" "$IT_P1"
else say "code phase-1: skip (done)"; fi
# =====================================================================================
# stage 3: code router FT  (router only; wiki + code 0.1% x 200 epochs)
# =====================================================================================
RFT_CODE_STEP=$(( $(latest "$CODE_P1") + IT_RFT_CODE ))
if [ "$(latest "$CODE_RFT")" != "$RFT_CODE_STEP" ]; then
  say "code router FT ($IT_RFT_CODE it, wiki+code subsets)"; prepare_copy "$CODE_P1" "$CODE_RFT"
  rft_common=(TRAIN_WEIGHTS="$CODE_RFT" SOURCE_STEP="$(latest "$CODE_P1")" RETUNE_ITERS=$IT_RFT_CODE SAVE_INTERVAL=$IT_RFT_CODE
              SOURCE_NUM_EXPERTS=8 NUM_EXPERTS=16 RESUME_FROM_NUM_EXPERTS=8 MICRO_BATCH_SIZE=$MB_RFT
              TRAIN_DATASET_WIKI="$SUB/wiki/train" TRAIN_DATASET_CODE="$SUB/code/train" TRAIN_DATASET_CONVERSATION="" MIXED_DATA_WEIGHT_MODE=equal_dataset
              DATASET_NAME=wiki_code_fixed_0p1pctx200 DATASET_SOURCE="wiki+code 0.1% fixed subsets x200 epochs, equal" MOE_AUX_LOSS_COEFF=0.0 MOE_Z_LOSS_COEFF=0.0
              "${common_probes_code[@]}" RUN_INITIAL_PROBE_EVAL=1 RUN_ID=2p-${VARIANT}-${KD}-code-router-ft MASTER_PORT=$((PORT_BASE+3)))
  if [ "$VARIANT" = hyb ]; then
    env "${rft_common[@]}" MOE_LPR_LOSS_COEFF=0 bash "$A/phase3_router_only_retune_shared_router_hybrid_lpr_local_bf16.sh"
  else
    env "${rft_common[@]}" MODEL_CONFIG_SCRIPT=scripts/experiment/a100/flame-moe-bf16-no-shared.sh bash "$A/phase3_router_only_retune_moe_mixed_local_bf16.sh"
  fi
  need "$CODE_RFT" "$RFT_CODE_STEP"
else say "code router FT: skip (done)"; fi
stub_meta "$CODE_RFT" "$RFT_CODE_STEP"
# =====================================================================================
# stage 4: conv KD init (kd1 only): e16 (router-FT'd) -> e24 on real wiki+code, logits KD
# =====================================================================================
if [ "$KD" = kd1 ]; then
  if [ "$(latest "$CONV_KD")" != "$IT_KD" ]; then
    say "conv KD init ($IT_KD it)"
    if [ "$VARIANT" = hyb ]; then
      env STAGE1_WEIGHTS_DIR="$CODE_RFT" SOURCE_REQUIRED_ITERS=$RFT_CODE_STEP TRAIN_ITERS=$IT_KD SAVE_INTERVAL=$IT_KD MICRO_BATCH_SIZE=$MB_KD \
          RUN_ID=2p-${VARIANT}-${KD}-conv-kd-init TRAIN_WEIGHTS="$CONV_KD" MASTER_PORT=$((PORT_BASE+4)) \
          bash "$A/run_g2_shared_router_conversation_expert_distill_init_wikicode_mha.sh" logits
    else
      env SOURCE_WEIGHTS_DIR="$CODE_RFT" SOURCE_REQUIRED_ITERS=$RFT_CODE_STEP TRAIN_ITERS=$IT_KD SAVE_INTERVAL=$IT_KD MICRO_BATCH_SIZE=$MB_KD \
          RUN_ID=2p-${VARIANT}-${KD}-conv-kd-init TRAIN_WEIGHTS="$CONV_KD" MASTER_PORT=$((PORT_BASE+4)) \
          bash "$A/run_g2_ffn_only_conversation_expert_distill_init_wikicode_mha.sh" logits
    fi
    need "$CONV_KD" "$IT_KD"
  else say "conv KD init: skip (done)"; fi
fi
# =====================================================================================
# stage 5: conv phase-1 (new experts 16:24 + all router rows, conv LM only)
# =====================================================================================
if [ "$(latest "$CONV_P1")" != "$IT_P1" ]; then
  say "conv phase-1 ($IT_P1 it)"
  if [ "$VARIANT" = hyb ]; then
    resume=(); [ "$KD" = kd1 ] && resume=(RESUME_FROM_WEIGHTS="$CONV_KD" RESUME_LOAD_OPTIM=0 RESUME_RESET_ITERATION=1)
    env STAGE1_WEIGHTS_DIR="$CODE_RFT" SOURCE_REQUIRED_ITERS=$RFT_CODE_STEP SOURCE_NUM_EXPERTS=16 NUM_EXPERTS=24 "${resume[@]}" \
        TRAIN_DATASET="$FLAME/conversation/train" DATASET_NAME=conversation_exact DATASET_SOURCE="Conversation exact train" "${common_probes_conv[@]}" \
        MOE_JOINT_REPLAY_LM=0 MOE_EXPANSION_DISTILL_MODE=none SHARED_ROUTER_HYBRID_TRAIN_ALL_EXPERTS_AND_ROUTER=0 SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS=1 \
        ROUTER_MEMORY_KL_COEFF=0.0 ROUTER_MEMORY_INTERVAL=0 \
        RUN_ID=2p-${VARIANT}-${KD}-conv-task TRAIN_WEIGHTS="$CONV_P1" TRAIN_ITERS=$IT_P1 SAVE_INTERVAL=$IT_P1 MICRO_BATCH_SIZE=$MB_P1 MASTER_PORT=$((PORT_BASE+5)) \
        bash scripts/experiment/continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh
  else
    kdflags=(); src="$CODE_RFT"
    [ "$KD" = kd1 ] && { kdflags=(FREEZE_DENSE_ATTENTION_LORA_WITH_NEW_EXPERTS=1 TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY=1 LOAD_EXPANDED_SOURCE=1); src="$CONV_KD"; }
    env SOURCE_TASK=code TARGET_TASK=conversation FREEZE_SHARED=1 TRAIN_ATTENTION_WITH_NEW_EXPERTS=0 "${kdflags[@]}" \
        SOURCE_WEIGHTS_DIR="$src" SOURCE_NUM_EXPERTS=16 NUM_EXPERTS=24 MODEL_CONFIG_SCRIPT=scripts/experiment/a100/flame-moe-bf16-no-shared.sh \
        ENABLE_OLD_MODEL_KL=0 OLD_MODEL_KL_COEFF=0.0 MOE_EXPANSION_DISTILL_MODE=none MOE_JOINT_REPLAY_LM=0 \
        TRAIN_DATASET="$FLAME/conversation/train" "${common_probes_conv[@]}" \
        RUN_ID=2p-${VARIANT}-${KD}-conv-task TRAIN_WEIGHTS="$CONV_P1" TRAIN_ITERS=$IT_P1 SAVE_INTERVAL=$IT_P1 MICRO_BATCH_SIZE=$MB_P1 MASTER_PORT=$((PORT_BASE+5)) \
        bash "$A/run_continual_moe_a100_bf16.sh"
  fi
  need "$CONV_P1" "$IT_P1"
else say "conv phase-1: skip (done)"; fi
# =====================================================================================
# stage 6: conv router FT (router only; wiki + code + conv 0.1% x 200 epochs)
# =====================================================================================
RFT_CONV_STEP=$(( IT_P1 + IT_RFT_CONV ))
if [ "$(latest "$CONV_RFT")" != "$RFT_CONV_STEP" ]; then
  say "conv router FT ($IT_RFT_CONV it, wiki+code+conv subsets)"; prepare_copy "$CONV_P1" "$CONV_RFT"
  rft_common=(TRAIN_WEIGHTS="$CONV_RFT" SOURCE_STEP=$IT_P1 RETUNE_ITERS=$IT_RFT_CONV SAVE_INTERVAL=$IT_RFT_CONV
              SOURCE_NUM_EXPERTS=16 NUM_EXPERTS=24 RESUME_FROM_NUM_EXPERTS=16 MICRO_BATCH_SIZE=$MB_RFT
              TRAIN_DATASET_WIKI="$SUB/wiki/train" TRAIN_DATASET_CODE="$SUB/code/train" TRAIN_DATASET_CONVERSATION="$SUB/conversation/train" MIXED_DATA_WEIGHT_MODE=equal_dataset
              DATASET_NAME=wiki_code_conv_fixed_0p1pctx200 DATASET_SOURCE="wiki+code+conv 0.1% fixed subsets x200 epochs, equal" MOE_AUX_LOSS_COEFF=0.0 MOE_Z_LOSS_COEFF=0.0
              "${common_probes_conv[@]}" RUN_INITIAL_PROBE_EVAL=1 RUN_ID=2p-${VARIANT}-${KD}-conv-router-ft MASTER_PORT=$((PORT_BASE+6)))
  if [ "$VARIANT" = hyb ]; then
    env "${rft_common[@]}" MOE_LPR_LOSS_COEFF=0 bash "$A/phase3_router_only_retune_shared_router_hybrid_lpr_local_bf16.sh"
  else
    env "${rft_common[@]}" MODEL_CONFIG_SCRIPT=scripts/experiment/a100/flame-moe-bf16-no-shared.sh bash "$A/phase3_router_only_retune_moe_mixed_local_bf16.sh"
  fi
  need "$CONV_RFT" "$RFT_CONV_STEP"
else say "conv router FT: skip (done)"; fi
say "ALL DONE: code_task=$CODE_P1 code_router_ft=$CODE_RFT conv_task=$CONV_P1 conv_router_ft=$CONV_RFT"
