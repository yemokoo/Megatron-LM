#!/bin/bash
# MoE-LPR on the FFN + QKVO-attention-expert shared-router model -- the same
# expert layout as Ours-attn (HF kd_1phase/ffn_attn_shared_router), trained the
# MoE-LPR way instead of KD-init + one-phase replay:
#
#   HF hybrid wiki e8 (fixed)
#     -> code 1800        8->16, new FFN+attention experts + router only, no replay/KD
#     -> LPR retune 360   router only, wiki+code equal mix, supervised task-group loss
#     -> conv 1800        16->24, same recipe
#     -> LPR retune 360   router only, wiki+code+conv
#
# Stage scripts are the stock ones: the hybrid expand launcher
# (continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh, which
# trains new experts + router only unless told otherwise) and the hybrid
# router-only retune (phase3_router_only_retune_shared_router_hybrid_lpr_local_bf16.sh,
# to which the MOE_LPR_* plumbing and the equal_dataset mix were added today).
set -euo pipefail
A="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P="$(cd "$A/../../.." && pwd)"; cd "$P"

HFW=/data2/seonghyeonnoh/LLM-continual-learning-runs/hf_g2_wiki_code_conversation/g2_wiki_code_conversation/sources/ffn_attn_shared_router/wiki
ROOT="${LPR_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/moe_lpr_hybrid_from_hf_e8_20260826}"
LPR_COEFF="${LPR_COEFF:-0.1}"
GPUS="${GPUS:-0,1,2,3}"
FLAME=/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup

export FLAME_DATA_ROOT="$FLAME"
export LOCAL_SSD_ROOT="$ROOT/scratch" LOCAL_WEIGHTS="$ROOT/weights" G2_ROOT="$ROOT/g2"
export CUDA_VISIBLE_DEVICES="$GPUS" NPROC_PER_NODE="$(awk -F, '{print NF}' <<< "$GPUS")"
export GLOBAL_BATCH_SIZE=2304 DIRECT_LOCAL_SAVE=1 WANDB_MODE=offline PAUSE_SECONDS=0
export PYTHONNOUSERSITE=1
export PATH=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin:/usr/bin:/bin
export TOKENIZER_MODEL=/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6
# architecture = Ours-attn (from the HF checkpoints' common.pt)
export NUM_LAYERS=9 HIDDEN_SIZE=1024 FFN_HIDDEN_SIZE=5472 NUM_QUERY_GROUPS=16 MOE_FFN_HIDDEN_SIZE=352 \
       MOE_LAYER_FREQ="[0]*1+[1]*8" MOE_ROUTER_TOPK=4 MOE_ROUTER_DTYPE=fp32 \
       ATTN_LORA_RANK=256 ATTN_LORA_ALPHA=256 ATTN_FULL_RANK_LORA_RANK=256 ATTN_FULL_RANK_LORA_ALPHA=256 \
       ATTN_FULL_RANK_LORA_TARGETS=qkvo MOE_GROUPED_GEMM=1 ATTN_LORA_GROUPED_GEMM=1
# MoE-LPR stage 1: no replay, no expansion distillation, whole router trainable
# --no-save-optim: saving the optimizer state of the partially-trainable expert
# groups fails dist-ckpt sharding validation ("Invalid access pattern for
# ShardedTensor optimizer.state.exp_avg...experts...") at DP=2 and lost a full
# stage-1 run.  Every later stage loads with --no-load-optim, so nothing needs it.
export NO_SAVE_OPTIM=1
export MOE_JOINT_REPLAY_LM=0 MOE_EXPANSION_DISTILL_MODE=none \
       SHARED_ROUTER_HYBRID_TRAIN_ALL_EXPERTS_AND_ROUTER=0 SHARED_ROUTER_HYBRID_TRAIN_ALL_ROUTER_ROWS=1 \
       ROUTER_MEMORY_KL_COEFF=0.0 ROUTER_MEMORY_INTERVAL=0

[ -f "$HFW/latest_checkpointed_iteration.txt" ] || { echo "[ERROR] hybrid e8 missing: $HFW" >&2; exit 1; }
mkdir -p "$ROOT/scratch" "$ROOT/weights" "$ROOT/g2" "$ROOT/logs"

CODE_OUT="$ROOT/code_task"; CODE_LPR_OUT="$ROOT/code_router_lpr"
CONV_OUT="$ROOT/conversation_task"; CONV_LPR_OUT="$ROOT/conversation_router_lpr"

latest() { [ -f "$1/latest_checkpointed_iteration.txt" ] && tr -d '[:space:]' < "$1/latest_checkpointed_iteration.txt" || echo ""; }
say() { printf '[LPR-HYB %s] %s\n' "$(date '+%F %T')" "$*"; }
nbin() { find "$1" -maxdepth 1 -name '*.bin' | wc -l; }
ensure_train_iters_metadata() {   # dir step -- the expand launcher reads logs/run_metadata.json['train_iters'] from its source
  python3 - "$1" "$2" <<'PY'
import json, os, sys
d, step = sys.argv[1], int(sys.argv[2]); p = os.path.join(d, "logs", "run_metadata.json")
os.makedirs(os.path.dirname(p), exist_ok=True)
m = json.load(open(p)) if os.path.isfile(p) else {}
m.setdefault("train_iters", step); json.dump(m, open(p, "w"), indent=1)
PY
}
prepare_retune_copy() {   # source dest -- the retune trains in place on a copy
  if [ ! -f "$2/latest_checkpointed_iteration.txt" ]; then
    mkdir -p "$2"; rsync -aH --exclude wandb/ "$1/" "$2/"
    printf 'source=%s\nsource_step=%s\nstage=lpr\n' "$1" "$(latest "$1")" > "$2/LPR_SOURCE.txt"
  fi
}

say "e8=$HFW"; say "root=$ROOT gpus=$GPUS nproc=$NPROC_PER_NODE lpr_coeff=$LPR_COEFF"
if [ "${PLAN_ONLY:-0}" = 1 ]; then say "PLAN_ONLY: stages code=$(latest "$CODE_OUT") code+lpr=$(latest "$CODE_LPR_OUT") conv=$(latest "$CONV_OUT") conv+lpr=$(latest "$CONV_LPR_OUT")"; exit 0; fi

# ---- 1. code: 8 -> 16 ----------------------------------------------------------
if [ "$(latest "$CODE_OUT")" != 1800 ]; then
  say "stage 1: code 1800"
  STAGE1_WEIGHTS_DIR="$HFW" SOURCE_NUM_EXPERTS=8 NUM_EXPERTS=16 \
  TRAIN_DATASET="$FLAME/code/train" DATASET_NAME=code_exact DATASET_SOURCE="Python code exact train" \
  PROBE_DATASET="$FLAME/code/test" PROBE_NAME=code_probe \
  SECONDARY_PROBE_DATASET="$FLAME/wiki/test" SECONDARY_PROBE_NAME=wiki_probe \
  RUN_ID=lprhyb-code-e8to16 TRAIN_WEIGHTS="$CODE_OUT" TRAIN_ITERS=1800 SAVE_INTERVAL=1800 \
  MICRO_BATCH_SIZE="${CODE_MB:-96}" MASTER_PORT=29950 \
    bash scripts/experiment/continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh
  [ "$(latest "$CODE_OUT")" = 1800 ] || { say "stage 1 did not reach 1800"; exit 1; }
else say "stage 1: skip (done)"; fi

# ---- 2. LPR retune on wiki+code ------------------------------------------------
if [ "$(latest "$CODE_LPR_OUT")" != 2160 ]; then
  say "stage 2: code LPR retune 360"
  prepare_retune_copy "$CODE_OUT" "$CODE_LPR_OUT"
  RUN_ID=lprhyb-code-router-360 TRAIN_WEIGHTS="$CODE_LPR_OUT" \
  SOURCE_NUM_EXPERTS=8 NUM_EXPERTS=16 RESUME_FROM_NUM_EXPERTS=8 RETUNE_ITERS=360 SAVE_INTERVAL=360 \
  TRAIN_DATASET_WIKI="$FLAME/wiki/train" TRAIN_DATASET_CODE="$FLAME/code/train" TRAIN_DATASET_CONVERSATION="" \
  MIXED_DATA_WEIGHT_MODE=equal_dataset \
  MOE_LPR_LOSS_COEFF="$LPR_COEFF" MOE_LPR_DATASET_PREFIX_COUNTS="$(nbin "$FLAME/wiki/train"),$(nbin "$FLAME/code/train")" \
  MOE_LPR_TASK_EXPERT_RANGES="0:8,-" MOE_AUX_LOSS_COEFF=0.0 MOE_Z_LOSS_COEFF=0.0 \
  PROBE_DATASET="$FLAME/code/test" PROBE_NAME=code_probe \
  SECONDARY_PROBE_DATASET="$FLAME/wiki/test" SECONDARY_PROBE_NAME=wiki_probe RUN_INITIAL_PROBE_EVAL=1 \
  MICRO_BATCH_SIZE="${RETUNE_MB:-72}" MASTER_PORT=29951 \
    bash "$A/phase3_router_only_retune_shared_router_hybrid_lpr_local_bf16.sh"
  [ "$(latest "$CODE_LPR_OUT")" = 2160 ] || { say "stage 2 did not reach 2160"; exit 1; }
else say "stage 2: skip (done)"; fi
ensure_train_iters_metadata "$CODE_LPR_OUT" 2160

# ---- 3. conversation: 16 -> 24 from the LPR-retuned code model -----------------
if [ "$(latest "$CONV_OUT")" != 1800 ]; then
  say "stage 3: conversation 1800"
  STAGE1_WEIGHTS_DIR="$CODE_LPR_OUT" SOURCE_NUM_EXPERTS=16 NUM_EXPERTS=24 \
  TRAIN_DATASET="$FLAME/conversation/train" DATASET_NAME=conversation_exact DATASET_SOURCE="Conversation exact train" \
  PROBE_DATASET="$FLAME/conversation/test" PROBE_NAME=conversation_probe \
  SECONDARY_PROBE_DATASET="$FLAME/wiki/test" SECONDARY_PROBE_NAME=wiki_probe \
  TERTIARY_PROBE_DATASET="$FLAME/code/test" TERTIARY_PROBE_NAME=code_probe TERTIARY_PROBE_EVAL_INTERVAL=50 \
  RUN_ID=lprhyb-conv-e16to24 TRAIN_WEIGHTS="$CONV_OUT" TRAIN_ITERS=1800 SAVE_INTERVAL=1800 \
  MICRO_BATCH_SIZE="${CONV_MB:-96}" MASTER_PORT=29952 \
    bash scripts/experiment/continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh
  [ "$(latest "$CONV_OUT")" = 1800 ] || { say "stage 3 did not reach 1800"; exit 1; }
else say "stage 3: skip (done)"; fi

# ---- 4. LPR retune on wiki+code+conv --------------------------------------------
if [ "$(latest "$CONV_LPR_OUT")" != 2160 ]; then
  say "stage 4: conversation LPR retune 360"
  prepare_retune_copy "$CONV_OUT" "$CONV_LPR_OUT"
  RUN_ID=lprhyb-conv-router-360 TRAIN_WEIGHTS="$CONV_LPR_OUT" \
  SOURCE_NUM_EXPERTS=16 NUM_EXPERTS=24 RESUME_FROM_NUM_EXPERTS=16 RETUNE_ITERS=360 SAVE_INTERVAL=360 \
  TRAIN_DATASET_WIKI="$FLAME/wiki/train" TRAIN_DATASET_CODE="$FLAME/code/train" TRAIN_DATASET_CONVERSATION="$FLAME/conversation/train" \
  MIXED_DATA_WEIGHT_MODE=equal_dataset \
  MOE_LPR_LOSS_COEFF="$LPR_COEFF" MOE_LPR_DATASET_PREFIX_COUNTS="$(nbin "$FLAME/wiki/train"),$(nbin "$FLAME/code/train"),$(nbin "$FLAME/conversation/train")" \
  MOE_LPR_TASK_EXPERT_RANGES="0:8,8:16,-" MOE_AUX_LOSS_COEFF=0.0 MOE_Z_LOSS_COEFF=0.0 \
  PROBE_DATASET="$FLAME/conversation/test" PROBE_NAME=conversation_probe \
  SECONDARY_PROBE_DATASET="$FLAME/wiki/test" SECONDARY_PROBE_NAME=wiki_probe \
  TERTIARY_PROBE_DATASET="$FLAME/code/test" TERTIARY_PROBE_NAME=code_probe TERTIARY_PROBE_EVAL_INTERVAL=50 RUN_INITIAL_PROBE_EVAL=1 \
  MICRO_BATCH_SIZE="${RETUNE_MB:-72}" MASTER_PORT=29953 \
    bash "$A/phase3_router_only_retune_shared_router_hybrid_lpr_local_bf16.sh"
  [ "$(latest "$CONV_LPR_OUT")" = 2160 ] || { say "stage 4 did not reach 2160"; exit 1; }
else say "stage 4: skip (done)"; fi

say "DONE"
say "  code      : $CODE_OUT      (iter 1800)"
say "  code+LPR  : $CODE_LPR_OUT  (iter 2160)"
say "  conv      : $CONV_OUT      (iter 1800)"
say "  conv+LPR  : $CONV_LPR_OUT  (iter 2160)"
