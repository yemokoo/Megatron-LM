#!/bin/bash
# G-new: fully replay-free chain (no real wiki/code data after the wiki pretrain; probes = test sets, measurement only).
#   s0  HF hybrid wiki e8
#   [gen1]  BoS+anchor wiki samples from s0 (1% = 41,472 x 512, KV-cached)
#   [code KD]  e8->e16 logit self-KD on gen1 wiki (600)      -> [code 1-phase] code LM + gen1-wiki replay (router only), 1800 = G_code
#   [gen2]  wiki + code samples from G_code (1% each)
#   [conv KD]  e16->e24 self-KD on gen2 wiki+code (600)     -> [conv 1-phase] conv LM + gen2 wiki/code replay, 1800       = G_conv
# Stock stage scripts unmodified (env only).  Only the final iteration of each stage is saved (--no-save-optim).
#   GPUS=0,1 bash scripts/experiment/a100/run_selfgen_replay_Gnew_full_chain.sh        (SMOKE=1: 3 it/stage, 64-seq gens)
#   PLAN_ONLY=1 ... prints the stage plan and exits.
set -euo pipefail
A="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; P="$(cd "$A/../../.." && pwd)"; cd "$P"
HF=/data2/seonghyeonnoh/LLM-continual-learning-runs/hf_g2_wiki_code_conversation/g2_wiki_code_conversation
WIKI="$HF/sources/ffn_attn_shared_router/wiki"
ANCH=/data2/seonghyeonnoh/LLM-continual-learning-runs/bos_samples_20260827/anchors   # first-token histograms (wiki top-1024, code top-64)
FLAME=/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup
SMOKE="${SMOKE:-0}"; GPUS="${GPUS:-0,1}"; PORT_BASE="${PORT_BASE:-29860}"
ROOT="${ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/selfgen_replay_Gnew_20260828$([ "$SMOKE" = 1 ] && echo _smoke || true)}"
export FLAME_DATA_ROOT="$FLAME" LOCAL_SSD_ROOT="$ROOT/scratch" LOCAL_WEIGHTS="$ROOT/weights" G2_ROOT="$ROOT/g2" LOCAL_BASE="$ROOT/local"
export CUDA_VISIBLE_DEVICES="$GPUS" NPROC_PER_NODE="$(awk -F, '{print NF}' <<< "$GPUS")"
export DIRECT_LOCAL_SAVE=1 WANDB_MODE=offline PAUSE_SECONDS=0 PYTHONNOUSERSITE=1
export PATH=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin:/usr/bin:/bin
export TOKENIZER_MODEL=/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6
export NUM_LAYERS=9 HIDDEN_SIZE=1024 FFN_HIDDEN_SIZE=5472 NUM_QUERY_GROUPS=16 MOE_FFN_HIDDEN_SIZE=352 MOE_LAYER_FREQ="[0]*1+[1]*8" MOE_ROUTER_TOPK=4 MOE_ROUTER_DTYPE=fp32 MOE_GROUPED_GEMM=1 \
       ATTN_LORA_RANK=256 ATTN_LORA_ALPHA=256 ATTN_FULL_RANK_LORA_RANK=256 ATTN_FULL_RANK_LORA_ALPHA=256 ATTN_FULL_RANK_LORA_TARGETS=qkvo ATTN_LORA_GROUPED_GEMM=1
export NO_SAVE_OPTIM=1 LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND=0
if [ "$SMOKE" = 1 ]; then IT_KD=3; IT_P1=3; NGEN=64; GBATCH=64; export GLOBAL_BATCH_SIZE=64; MB_KD=32; MB_P1=32
  export PROBE_EVAL_ITERS=2 SECONDARY_PROBE_EVAL_ITERS=2 TERTIARY_PROBE_EVAL_ITERS=2 PROBE_EVAL_INTERVAL=3 SECONDARY_PROBE_EVAL_INTERVAL=3 TERTIARY_PROBE_EVAL_INTERVAL=3
else IT_KD=600; IT_P1=1800; NGEN=41472; GBATCH=512; export GLOBAL_BATCH_SIZE=2304; MB_KD="${MB_KD:-36}"; MB_P1="${MB_P1:-72}"
  export PROBE_EVAL_ITERS=25 SECONDARY_PROBE_EVAL_ITERS=25 TERTIARY_PROBE_EVAL_ITERS=25 PROBE_EVAL_INTERVAL=50 SECONDARY_PROBE_EVAL_INTERVAL=50 TERTIARY_PROBE_EVAL_INTERVAL=50; fi
export PROBE_MICRO_BATCH_SIZE=32
GEN1="$ROOT/gen1"; CODE_KD="$ROOT/code_kd_init"; CODE_P1="$ROOT/code_1phase"; GEN2="$ROOT/gen2"; CONV_KD="$ROOT/conv_kd_init"; CONV_P1="$ROOT/conv_1phase"
latest() { [ -f "$1/latest_checkpointed_iteration.txt" ] && tr -d '[:space:]' < "$1/latest_checkpointed_iteration.txt" || echo ""; }
say() { printf '[Gnew %s] %s\n' "$(date '+%F %T')" "$*"; }
need() { [ "$(latest "$1")" = "$2" ] || { say "FAILED: $1 at '$(latest "$1")', expected $2"; exit 1; }; }
stub_meta() { python3 - "$1" "$2" <<'PY'
import json, os, sys
d, step = sys.argv[1], int(sys.argv[2]); p = os.path.join(d, "logs", "run_metadata.json"); os.makedirs(os.path.dirname(p), exist_ok=True)
m = json.load(open(p)) if os.path.isfile(p) else {}
m.setdefault("train_iters", step); json.dump(m, open(p, "w"), indent=1)
PY
}
gen() {  # ckpt ne src task out_dir  -- BoS + per-sequence anchor sampling, KV-cached; writes <out>/train_text_document.{bin,idx}
  local ckpt=$1 ne=$2 src=$3 task=$4 out=$5; local json="$ANCH/${task}_first_token_top$([ "$task" = wiki ] && echo 1024 || echo 64).json"
  [ -s "$out/train_text_document.bin" ] && { say "gen $task: skip (exists)"; return; }
  say "gen $task from $ckpt (experts=$ne src=$src, N=$NGEN)"
  local g0; g0="$(cut -d, -f1 <<< "$GPUS")"
  GPU=$g0 CKPT="$ckpt" NE=$ne SRC=$src OUT="$out/raw" N=$NGEN T=512 BATCH=$GBATCH KV=1 ANCHOR_JSON="$json" LABEL="Gnew_${task}" SEED=7 PORT=$((PORT_BASE+9)) \
    bash scripts/analysis/run_bos_sample_hybrid.sh
  [ -s "$out/raw/gen_text_document.idx" ] || { say "FAILED: gen $task"; exit 1; }
  cp "$out/raw/gen_text_document.bin" "$out/train_text_document.bin"; cp "$out/raw/gen_text_document.idx" "$out/train_text_document.idx"   # real copies (launcher rsync -l breaks symlinks)
}
probes_code=(PROBE_DATASET="$FLAME/code/test" PROBE_NAME=code_probe SECONDARY_PROBE_DATASET="$FLAME/wiki/test" SECONDARY_PROBE_NAME=wiki_probe)
probes_conv=(PROBE_DATASET="$FLAME/conversation/test" PROBE_NAME=conversation_probe SECONDARY_PROBE_DATASET="$FLAME/wiki/test" SECONDARY_PROBE_NAME=wiki_probe TERTIARY_PROBE_DATASET="$FLAME/code/test" TERTIARY_PROBE_NAME=code_probe)
mkdir -p "$ROOT"/{scratch,weights,g2,logs,local}; [ "$(latest "$WIKI")" = 1800 ] || { say "wiki e8 missing"; exit 1; }; stub_meta "$WIKI" 1800
say "root=$ROOT gpus=$GPUS smoke=$SMOKE"
if [ "${PLAN_ONLY:-0}" = 1 ]; then
  for s in "gen1 wiki <- $WIKI" "code_kd_init ($IT_KD) <- gen1 wiki" "code_1phase ($IT_P1) code LM + gen1 wiki replay" "gen2 wiki+code <- code_1phase" "conv_kd_init ($IT_KD) <- gen2" "conv_1phase ($IT_P1) conv LM + gen2 replay"; do echo "  - $s"; done; exit 0; fi
# ---- 1. gen1: wiki samples from the wiki-only model ---------------------------------------
gen "$WIKI" 8 8 wiki "$GEN1/wiki"
# ---- 2. code KD init on gen1 wiki --------------------------------------------------------
if [ "$(latest "$CODE_KD")" != "$IT_KD" ]; then say "code KD init ($IT_KD it, gen1 wiki)"
  env STAGE1_WEIGHTS_DIR="$WIKI" SOURCE_REQUIRED_ITERS=1800 TRAIN_DATASET="$GEN1/wiki" DATASET_NAME=selfgen_wiki_expansion_distill DATASET_SOURCE="self-generated wiki (from wiki e8) for code expert KD init" \
      TRAIN_ITERS=$IT_KD SAVE_INTERVAL=$IT_KD MICRO_BATCH_SIZE=$MB_KD RUN_ID=Gnew-code-kd-init TRAIN_WEIGHTS="$CODE_KD" MASTER_PORT=$((PORT_BASE+1)) \
      bash "$A/run_g2_shared_router_code_expert_distill_init_mha.sh" logits
  need "$CODE_KD" "$IT_KD"; else say "code KD init: skip"; fi
# ---- 3. code 1-phase: code LM (new experts + all router rows) + gen1 wiki replay (router only) ----
if [ "$(latest "$CODE_P1")" != "$IT_P1" ]; then say "code 1-phase ($IT_P1 it, replay = gen1 wiki)"
  env STAGE1_WEIGHTS_DIR="$WIKI" SOURCE_REQUIRED_ITERS=1800 RESUME_FROM_WEIGHTS="$CODE_KD" DISTILL_SOURCE_REQUIRED_ITERS=$IT_KD DISTILL_SOURCE_ITERS=$IT_KD \
      JOINT_REPLAY_DATASET="$GEN1/wiki" JOINT_REPLAY_SECONDARY_DATASET="" DATASET_NAME=code_train_with_selfgen_wiki_joint_replay DATASET_SOURCE="Code LM + self-generated wiki router-only replay" \
      "${probes_code[@]}" TRAIN_ITERS=$IT_P1 SAVE_INTERVAL=$IT_P1 MICRO_BATCH_SIZE=$MB_P1 RUN_ID=Gnew-code-1phase TRAIN_WEIGHTS="$CODE_P1" MASTER_PORT=$((PORT_BASE+2)) \
      bash "$A/run_g2_shared_router_code_wiki_joint_lm_allrouter_mha.sh" logits
  need "$CODE_P1" "$IT_P1"; else say "code 1-phase: skip"; fi
stub_meta "$CODE_P1" "$IT_P1"
# ---- 4. gen2: wiki + code samples from G_code ------------------------------------------------
gen "$CODE_P1" 16 8 wiki "$GEN2/wiki"; gen "$CODE_P1" 16 8 code "$GEN2/code"
# ---- 5. conv KD init on gen2 wiki+code --------------------------------------------------------
if [ "$(latest "$CONV_KD")" != "$IT_KD" ]; then say "conv KD init ($IT_KD it, gen2 wiki+code)"
  env STAGE1_WEIGHTS_DIR="$CODE_P1" SOURCE_REQUIRED_ITERS=$IT_P1 CODE_SOURCE_ITERS=$IT_P1 TRAIN_DATASET="$GEN2/wiki" TRAIN_DATASET_SECONDARY="$GEN2/code" \
      DATASET_NAME=selfgen_wiki_code_equal_expansion_distill DATASET_SOURCE="self-generated wiki+code (from G_code), equal, for conv expert KD init" \
      TRAIN_ITERS=$IT_KD SAVE_INTERVAL=$IT_KD MICRO_BATCH_SIZE=$MB_KD RUN_ID=Gnew-conv-kd-init TRAIN_WEIGHTS="$CONV_KD" MASTER_PORT=$((PORT_BASE+3)) \
      bash "$A/run_g2_shared_router_conversation_expert_distill_init_wikicode_mha.sh" logits
  need "$CONV_KD" "$IT_KD"; else say "conv KD init: skip"; fi
# ---- 6. conv 1-phase: conv LM + gen2 wiki/code replay (router only) ------------------------------
if [ "$(latest "$CONV_P1")" != "$IT_P1" ]; then say "conv 1-phase ($IT_P1 it, replay = gen2 wiki+code)"
  env STAGE1_WEIGHTS_DIR="$CODE_P1" SOURCE_REQUIRED_ITERS=$IT_P1 CODE_SOURCE_ITERS=$IT_P1 RESUME_FROM_WEIGHTS="$CONV_KD" DISTILL_SOURCE_REQUIRED_ITERS=$IT_KD DISTILL_SOURCE_ITERS=$IT_KD \
      JOINT_REPLAY_DATASET="$GEN2/wiki" JOINT_REPLAY_SECONDARY_DATASET="$GEN2/code" DATASET_NAME=conversation_train_with_selfgen_wikicode_joint_replay DATASET_SOURCE="Conversation LM + self-generated wiki/code router-only replay" \
      "${probes_conv[@]}" TRAIN_ITERS=$IT_P1 SAVE_INTERVAL=$IT_P1 MICRO_BATCH_SIZE=$MB_P1 RUN_ID=Gnew-conv-1phase TRAIN_WEIGHTS="$CONV_P1" MASTER_PORT=$((PORT_BASE+4)) \
      bash "$A/run_g2_shared_router_conversation_wikicode_joint_lm_allrouter_mha.sh" logits
  need "$CONV_P1" "$IT_P1"; else say "conv 1-phase: skip"; fi
say "ALL DONE: G_code=$CODE_P1 G_conv=$CONV_P1 (gen1=$GEN1 gen2=$GEN2)"
