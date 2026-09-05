#!/bin/bash
# Arm G: conversation stage of the HF kd_1phase/ffn_attn_shared_router chain with the
# wiki+code OLD data replaced by the model's own BoS+anchor samples (1% of train each,
# bos_samples_20260827/replay_1pct).  Everything else = the HF chain
# (run_g2_shared_router_joint_lm_code_kd_conversation_chain_mha.sh, mode=logits):
#   stage 2  code e16 -> conv slots e24, output-logit KD on (gen wiki + gen code) equal mix, 600 it
#   stage 3  conv LM (new experts 16:24 + all router rows) + (gen wiki + gen code) LM router-only, 1800 it
# Stock stage scripts + launcher untouched; GPUs 0,1; --no-save-optim (hybrid save bug at DP=2).
set -euo pipefail
A="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P="$(cd "$A/../../.." && pwd)"; cd "$P"
HFC=/data2/seonghyeonnoh/LLM-continual-learning-runs/hf_g2_wiki_code_conversation/g2_wiki_code_conversation/kd_1phase/ffn_attn_shared_router/code/one_phase
GEN=/data2/seonghyeonnoh/LLM-continual-learning-runs/bos_samples_20260827/replay_1pct/datasets
ROOT="${G_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/selfgen_replay_G_20260827}"
GPUS="${GPUS:-0,1}"; MODE=logits
FLAME=/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup
export FLAME_DATA_ROOT="$FLAME"
export LOCAL_SSD_ROOT="$ROOT/scratch" LOCAL_WEIGHTS="$ROOT/weights" G2_ROOT="$ROOT/g2"
export CUDA_VISIBLE_DEVICES="$GPUS" NPROC_PER_NODE="$(awk -F, '{print NF}' <<< "$GPUS")"
export GLOBAL_BATCH_SIZE=2304 DIRECT_LOCAL_SAVE=1 WANDB_MODE=offline PAUSE_SECONDS=0
export PYTHONNOUSERSITE=1
export PATH=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin:/usr/bin:/bin
export TOKENIZER_MODEL=/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6
export NUM_LAYERS=9 HIDDEN_SIZE=1024 FFN_HIDDEN_SIZE=5472 NUM_QUERY_GROUPS=16 MOE_FFN_HIDDEN_SIZE=352 \
       MOE_LAYER_FREQ="[0]*1+[1]*8" MOE_ROUTER_TOPK=4 MOE_ROUTER_DTYPE=fp32 \
       ATTN_LORA_RANK=256 ATTN_LORA_ALPHA=256 ATTN_FULL_RANK_LORA_RANK=256 ATTN_FULL_RANK_LORA_ALPHA=256 \
       ATTN_FULL_RANK_LORA_TARGETS=qkvo MOE_GROUPED_GEMM=1 ATTN_LORA_GROUPED_GEMM=1
export NO_SAVE_OPTIM=1
# HF chain probe schedule and step offsets (wiki 1800 + code-init 600 + code 1800 = 4200)
PROBE_INTERVAL=50; PROBE_ITERS=25; PROBE_MB=32; CONV_INIT_OFFSET=4200; CONV_OFFSET=4800
KD_OUT="$ROOT/conversation_kd_init_gen"; CONV_OUT="$ROOT/conversation_1phase_gen"
latest() { [ -f "$1/latest_checkpointed_iteration.txt" ] && tr -d '[:space:]' < "$1/latest_checkpointed_iteration.txt" || echo ""; }
say() { printf '[G %s] %s\n' "$(date '+%F %T')" "$*"; }
[ "$(latest "$HFC")" = 1800 ] || { say "HF code/one_phase not at 1800: $HFC"; exit 1; }
for t in wiki code; do [ -s "$GEN/$t/train_text_document.bin" ] || { say "missing generated set $GEN/$t"; exit 1; }; done
mkdir -p "$ROOT/scratch" "$ROOT/weights" "$ROOT/g2" "$ROOT/logs"
# the launcher reads logs/run_metadata.json['train_iters'] from the source; HF download has none
python3 - "$HFC" <<'PY'
import json, os, sys
d = sys.argv[1]; p = os.path.join(d, "logs", "run_metadata.json"); os.makedirs(os.path.dirname(p), exist_ok=True)
m = json.load(open(p)) if os.path.isfile(p) else {}
m.setdefault("train_iters", 1800); m.setdefault("note", "stub for STAGE1 metadata read; HF-downloaded checkpoint has no logs")
json.dump(m, open(p, "w"), indent=1)
PY
say "source=$HFC"; say "gen wiki=$GEN/wiki gen code=$GEN/code"; say "root=$ROOT gpus=$GPUS nproc=$NPROC_PER_NODE"
# ---- stage 2: conv KD init on generated wiki+code ---------------------------------
if [ "$(latest "$KD_OUT")" != 600 ]; then
  say "stage 2: conversation KD init (logits KD, gen wiki+code equal), 600 it, mb ${KD_MB:-36}"
  env STAGE1_WEIGHTS_DIR="$HFC" SOURCE_REQUIRED_ITERS=1800 CODE_SOURCE_MODE=$MODE CODE_SOURCE_ITERS=1800 CODE_SOURCE_MB=96 \
      TRAIN_DATASET="$GEN/wiki" TRAIN_DATASET_SECONDARY="$GEN/code" \
      DATASET_NAME=selfgen_wiki_code_equal_expansion_distill DATASET_SOURCE="BoS+anchor self-generated wiki + code (1% each), equal mix, for conv expert KD init" \
      TRAIN_ITERS=600 MICRO_BATCH_SIZE="${KD_MB:-36}" GLOBAL_BATCH_SIZE=2304 SAVE_INTERVAL=600 \
      PROBE_EVAL_INTERVAL=$PROBE_INTERVAL SECONDARY_PROBE_EVAL_INTERVAL=$PROBE_INTERVAL TERTIARY_PROBE_EVAL_INTERVAL=$PROBE_INTERVAL \
      PROBE_EVAL_ITERS=$PROBE_ITERS SECONDARY_PROBE_EVAL_ITERS=$PROBE_ITERS TERTIARY_PROBE_EVAL_ITERS=$PROBE_ITERS PROBE_MICRO_BATCH_SIZE=$PROBE_MB \
      PROBE_STEP_OFFSET=$CONV_INIT_OFFSET SECONDARY_PROBE_STEP_OFFSET=$CONV_INIT_OFFSET TERTIARY_PROBE_STEP_OFFSET=$CONV_INIT_OFFSET WANDB_STEP_OFFSET=$CONV_INIT_OFFSET \
      RUN_ID=G-selfgen-conv-kd-init-e16to24-logits-mb${KD_MB:-36}-600 TRAIN_WEIGHTS="$KD_OUT" MASTER_PORT=29983 \
      bash "$A/run_g2_shared_router_conversation_expert_distill_init_wikicode_mha.sh" $MODE
  [ "$(latest "$KD_OUT")" = 600 ] || { say "stage 2 did not reach 600"; exit 1; }
else say "stage 2: skip (done)"; fi
# ---- stage 3: conv 1-phase with generated replay -------------------------------------
if [ "$(latest "$CONV_OUT")" != 1800 ]; then
  say "stage 3: conversation 1-phase (conv LM + gen wiki/code router-only LM), 1800 it, mb ${CONV_MB:-72}"
  env STAGE1_WEIGHTS_DIR="$HFC" SOURCE_REQUIRED_ITERS=1800 RESUME_FROM_WEIGHTS="$KD_OUT" DISTILL_SOURCE_REQUIRED_ITERS=600 DISTILL_SOURCE_ITERS=600 \
      CODE_SOURCE_MODE=$MODE CODE_SOURCE_ITERS=1800 CODE_SOURCE_MB=96 \
      JOINT_REPLAY_DATASET="$GEN/wiki" JOINT_REPLAY_SECONDARY_DATASET="$GEN/code" \
      DATASET_NAME=conversation_train_with_selfgen_wikicode_joint_replay DATASET_SOURCE="Conversation primary LM + self-generated wiki/code router-only replay LM" \
      TRAIN_ITERS=1800 MICRO_BATCH_SIZE="${CONV_MB:-72}" GLOBAL_BATCH_SIZE=2304 SAVE_INTERVAL="${CONV_SAVE_INTERVAL:-600}" \
      PROBE_EVAL_INTERVAL=$PROBE_INTERVAL SECONDARY_PROBE_EVAL_INTERVAL=$PROBE_INTERVAL TERTIARY_PROBE_EVAL_INTERVAL=$PROBE_INTERVAL \
      PROBE_EVAL_ITERS=$PROBE_ITERS SECONDARY_PROBE_EVAL_ITERS=$PROBE_ITERS TERTIARY_PROBE_EVAL_ITERS=$PROBE_ITERS PROBE_MICRO_BATCH_SIZE=$PROBE_MB \
      PROBE_STEP_OFFSET=$CONV_OFFSET SECONDARY_PROBE_STEP_OFFSET=$CONV_OFFSET TERTIARY_PROBE_STEP_OFFSET=$CONV_OFFSET WANDB_STEP_OFFSET=$CONV_OFFSET \
      RUN_ID=G-selfgen-conv-1phase-e24-logits-mb${CONV_MB:-72}-1800 TRAIN_WEIGHTS="$CONV_OUT" MASTER_PORT=29984 \
      bash "$A/run_g2_shared_router_conversation_wikicode_joint_lm_allrouter_mha.sh" $MODE
  [ "$(latest "$CONV_OUT")" = 1800 ] || { say "stage 3 did not reach 1800"; exit 1; }
else say "stage 3: skip (done)"; fi
say "ALL DONE: kd_init=$KD_OUT conv=$CONV_OUT"
