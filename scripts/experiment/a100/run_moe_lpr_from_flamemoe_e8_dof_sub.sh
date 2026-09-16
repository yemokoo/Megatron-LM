#!/bin/bash
# MoE-LPR baseline from the SAME wiki checkpoint every other chain here starts
# from, so its hidden drift can be paired with the FFN-only e8 reference:
#
#   e8 wiki (flamemoe, fixed) -> code 1800 (new experts + router only)
#       -> LPR router retune 360 on wiki+code, supervised task-group loss
#       -> conv 1800 (16->24, new experts + router only)
#       -> LPR router retune 360 on wiki+code+conv
#
# The stock chain (run_g2_ffn_only_lpr_1800_360_1800_360_chain_mha.sh) starts
# from the a100 g2matched e8 that is not on this host, keeps outputs under
# ~/.local, and stages datasets on /tmp (1.7 GB free).  This wrapper keeps the
# stage scripts and only changes where things come from and go to.
#
# SOURCE_WEIGHTS_DIR is passed to the code stage alone: the retune stage works
# on its own rsync'd copy and the conv stage sets its source from FFN_SOURCE,
# so a global export would be picked up by the wrong stage.
set -euo pipefail
A="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$A/../../.."

# DoF 스윕: wiki 소스를 env 로 갈아끼울 수 있게 개방 (기본값은 기존 e8 그대로)
E8="${LPR_SOURCE_WEIGHTS:-/data2/seonghyeonnoh/LLM-continual-learning-runs/flamemoe/ffn_experts_only/wiki/pretrain/lm/full_training/g2_olddata_kd_9run_20260808__wiki_ffn_only_e8_step1800}"
ROOT="${LPR_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/moe_lpr_from_flamemoe_e8_20260826}"
LPR_COEFF="${LPR_COEFF:-0.1}"
# 여러 감마 셀을 동시에 돌릴 때 고정 포트가 겹쳐 EADDRINUSE 로 죽었다.
# PORT_BASE 로 셀마다 다른 대역을 주도록 상대화한다(기본값은 기존과 동일).
# PORT_BASE 를 안 주면 비어 있는 4연속 포트를 스스로 찾는다.  큐 워커가 이미 떠 있어
# 함수 정의를 바꿔도 반영되지 않는 상황에서도 셀 간 충돌을 막기 위함이다.
if [ -z "${PORT_BASE:-}" ]; then
  for base in $(seq 29940 4 30300); do
    busy=0
    for off in 0 1 2 3; do
      if ss -Hltn "sport = :$((base+off))" 2>/dev/null | grep -q .; then busy=1; break; fi
    done
    [ "$busy" = 0 ] && { PORT_BASE=$base; break; }
  done
  PORT_BASE="${PORT_BASE:-29940}"
fi
echo "[CONFIG] PORT_BASE=$PORT_BASE"
GPUS="${GPUS:-0,1}"

export FLAME_DATA_ROOT=/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup
export LOCAL_SSD_ROOT="$ROOT/scratch"          # phase3 retune rsyncs datasets here; /tmp cannot hold them
export LOCAL_WEIGHTS="$ROOT/weights"            # nothing should land in ~/.local
export G2_ROOT="$ROOT/g2"
export CUDA_VISIBLE_DEVICES="$GPUS"
export NPROC_PER_NODE="$(awk -F, '{print NF}' <<< "$GPUS")"
export STAGE_INPUTS_TO_SCRATCH=0 DIRECT_LOCAL_SAVE=1
export WANDB_MODE=offline PAUSE_SECONDS=0
export PYTHONNOUSERSITE=1
export PATH=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin:/usr/bin:/bin

[ -f "$E8/latest_checkpointed_iteration.txt" ] || { echo "[ERROR] e8 source missing: $E8" >&2; exit 1; }
mkdir -p "$ROOT/scratch" "$ROOT/weights" "$ROOT/g2" "$ROOT/logs"

CODE_OUT="$ROOT/code_task"
CODE_LPR_OUT="$ROOT/code_router_lpr"
CONV_OUT="$ROOT/conversation_task"
CONV_LPR_OUT="$ROOT/conversation_router_lpr"

latest() { [ -f "$1/latest_checkpointed_iteration.txt" ] && tr -d '[:space:]' < "$1/latest_checkpointed_iteration.txt" || echo ""; }
say() { printf '[LPR-CHAIN %s] %s\n' "$(date '+%F %T')" "$*"; }

say "e8=$E8"; say "root=$ROOT gpus=$GPUS nproc=$NPROC_PER_NODE lpr_coeff=$LPR_COEFF"

# ---- 1. code: 8 -> 16, new experts + router only, no replay -------------------
if [ "$(latest "$CODE_OUT")" != 1800 ]; then
  say "stage 1: code 1800"
  SOURCE_WEIGHTS_DIR="$E8" SOURCE_REQUIRED_ITERS=1800 \
  RUN_ID=lpr-code-e8to16 TRAIN_WEIGHTS="$CODE_OUT" TRAIN_ITERS=1800 SAVE_INTERVAL=1800 \
  MICRO_BATCH_SIZE="${CODE_MB:-96}" MASTER_PORT=$((PORT_BASE+0)) \
    bash "$A/code_from_wiki_ffn_moe_g2matched_attn_freeze_mha_a100_bf16.sh"
  [ "$(latest "$CODE_OUT")" = 1800 ] || { say "stage 1 did not reach 1800"; exit 1; }
else say "stage 1: skip (done)"; fi

# ---- 2. LPR retune on wiki+code (routers only, task-group loss) ---------------
if [ "$(latest "$CODE_LPR_OUT")" != 2160 ]; then
  say "stage 2: code LPR retune 360"
  RUN_ID=lpr-code-router-360 LPR_COEFF="$LPR_COEFF" RETUNE_ITERS=360 \
  MICRO_BATCH_SIZE="${CODE_LPR_MB:-96}" MASTER_PORT=$((PORT_BASE+1)) \
    bash "$A/run_g2_ffn_only_task_group_lpr_router_retune_mha_sub.sh" code "$CODE_OUT" "$CODE_LPR_OUT"
  [ "$(latest "$CODE_LPR_OUT")" = 2160 ] || { say "stage 2 did not reach 2160"; exit 1; }
else say "stage 2: skip (done)"; fi

# ---- 3. conversation: 16 -> 24 from the LPR-retuned code model ---------------
# The phase4 script validates its exp1/exp2 sources (step 3600) even when
# RUN_ONLY_STAGE=ffn_only never touches them, so point both at a stub tracker.
mkdir -p "$ROOT/g2/unused_source_stub_3600"; echo 3600 > "$ROOT/g2/unused_source_stub_3600/latest_checkpointed_iteration.txt"
if [ "$(latest "$CONV_OUT")" != 1800 ]; then
  say "stage 3: conversation 1800"
  CONVERSATION_TRAIN="$FLAME_DATA_ROOT/conversation/train" \
  CONVERSATION_PROBE_DATASET="$FLAME_DATA_ROOT/conversation/test" \
  CODE_PROBE_DATASET="$FLAME_DATA_ROOT/code/test" \
  WIKI_PROBE_DATASET="$FLAME_DATA_ROOT/wiki/test" \
  EXP1_FREEZE_WIKI_SOURCE="$ROOT/g2/unused_source_stub_3600" \
  EXP2_UNFREEZE_WIKI_SOURCE="$ROOT/g2/unused_source_stub_3600" \
  RUN_ONLY_STAGE=ffn_only FFN_SOURCE="$CODE_LPR_OUT" FFN_SOURCE_REQUIRED_ITERS=2160 \
  FFN_RUN_ID=lpr-conv-e16to24 FFN_TRAIN_WEIGHTS="$CONV_OUT" TRAIN_ITERS=1800 SAVE_INTERVAL=1800 \
  FFN_MICRO_BATCH_SIZE="${CONV_MB:-64}" SOURCE_LOGICAL_STEP=2160 FFN_MASTER_PORT=$((PORT_BASE+2)) \
    bash "$A/run_g2_phase4_conversation_after_router_finetune_offline_chain_mha.sh" ffn_only
  [ "$(latest "$CONV_OUT")" = 1800 ] || { say "stage 3 did not reach 1800"; exit 1; }
else say "stage 3: skip (done)"; fi

# ---- 4. LPR retune on wiki+code+conv ------------------------------------------
if [ "$(latest "$CONV_LPR_OUT")" != 2160 ]; then
  say "stage 4: conversation LPR retune 360"
  RUN_ID=lpr-conv-router-360 LPR_COEFF="$LPR_COEFF" RETUNE_ITERS=360 \
  MICRO_BATCH_SIZE="${CONV_LPR_MB:-64}" MASTER_PORT=$((PORT_BASE+3)) \
    bash "$A/run_g2_ffn_only_task_group_lpr_router_retune_mha_sub.sh" conversation "$CONV_OUT" "$CONV_LPR_OUT"
  [ "$(latest "$CONV_LPR_OUT")" = 2160 ] || { say "stage 4 did not reach 2160"; exit 1; }
else say "stage 4: skip (done)"; fi

say "DONE"
say "  code      : $CODE_OUT      (iter 1800)"
say "  code+LPR  : $CODE_LPR_OUT  (iter 2160)"
say "  conv      : $CONV_OUT      (iter 1800)"
say "  conv+LPR  : $CONV_LPR_OUT  (iter 2160)"
