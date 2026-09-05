#!/usr/bin/env bash
set -uo pipefail
# CKA-32 top-4% GT chain, one objective per invocation:
#   1. Code   : Wiki KD-init (16E) + Code LM, replay = Code32 GT (router-only)
#   2. Expand : E16->E24, output-logit KD on Wiki+Code (existing recipe, unchanged)
#   3. Conv   : (24E) + Conv LM, replay = Conv32 GT (router-only)
# OBJECTIVE=hidden_mse | lm  selects the replay loss for stages 1 and 3.
#   OBJECTIVE=hidden_mse GPUS=0,1,2,3 ./run_cka32_top4_chain_mha.sh
#   OBJECTIVE=lm         GPUS=4,5,6,7 ./run_cka32_top4_chain_mha.sh
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$R"
OBJECTIVE="${OBJECTIVE:?set OBJECTIVE=hidden_mse|lm}"
GPUS="${GPUS:?set GPUS, e.g. 0,1,2,3}"
NP=$(echo "$GPUS" | tr ',' '\n' | wc -l)
PORT_BASE="${PORT_BASE:-38000}"
D=/data2/seonghyeonnoh/LLM-continual-learning-data
V=$D/flamedata2.data2-verified-backup
HIDDEN_MSE_COEFF="${HIDDEN_MSE_COEFF:-10}"
GT_PCT="${GT_PCT:-4}"
GT_SET="${GT_SET:-top${GT_PCT}}"
MB="${MB:-96}"
TAG="${OBJECTIVE}"; [[ "$OBJECTIVE" == hidden_mse ]] && TAG="hidden_mse_c${HIDDEN_MSE_COEFF}"
STUDY=/data2/seonghyeonnoh/LLM-continual-learning-runs/cka32_${GT_SET}_chain_${TAG}_20260819
WIKI_KD_INIT=/data2/seonghyeonnoh/LLM-continual-learning-runs/flamemoe/ffn_experts_only/code/expansion_kd_init/kd_init/full_training/g2_olddata_kd_9run_20260808__code_e8_to_e16_wiki_kd_step600
CODE_OUT="$STUDY/01_code_${TAG}_1800"
KD24="$STUDY/02_expansion_kd_init_e16_to_e24_step600"
CONV_OUT="$STUDY/03_conv_${TAG}_1800"
FLAME_ENV=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100
mkdir -p "$STUDY/logs"; ST="$STUDY/logs/status.tsv"
say(){ echo "$(date -Is) $*" | tee -a "$ST"; }
done_at(){ [[ -d "$1/iter_$(printf %07d "$2")" && "$(cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null)" == "$2" ]]; }

plan(){ cat <<PLAN
[CHAIN] gt=${GT_SET} objective=$OBJECTIVE (tag=$TAG, mse_coeff=$HIDDEN_MSE_COEFF, primary_mb=$MB)  gpus=$GPUS (nproc=$NP)  study=$STUDY
[1] Code   src=$(basename $WIKI_KD_INIT) (16E, it600)
           primary=Code LM 1800 | replay=$D/cka32_${GT_SET}_code_20260819  | replay loss=$OBJECTIVE, router-only
           out=$CODE_OUT
[2] Expand src=[1] it1800 (16E) -> 24E | logit-KD on Wiki+Code equal, 600 steps | trainable=new experts+router rows 16:24 | out=$KD24
[3] Conv   src=[2] it600 (24E)
           primary=Conv LM 1800 | replay=$D/cka32_${GT_SET}_conv_20260819  | replay loss=$OBJECTIVE, router-only
           out=$CONV_OUT
PLAN
}
plan
[[ "${PLAN_ONLY:-0}" == 1 ]] && { echo; echo "--- stage-1 launcher PLAN ---"; PLAN_ONLY=1 TASK=code TRAIN_ITERS=1800 CUDA_VISIBLE_DEVICES="$GPUS" NPROC_PER_NODE=$NP OBJECTIVE=$OBJECTIVE SOURCE=$WIKI_KD_INIT REPLAY_MINISET_DIR=$D/cka32_${GT_SET}_code_20260819/train REPLAY_GT_PATH=$D/cka32_${GT_SET}_code_20260819/gt STUDY_ROOT=$STUDY LABEL=$(basename $CODE_OUT) bash scripts/experiment/a100/run_g2_cka_gt_contextual_replay_mha.sh | grep -E "PREFLIGHT|task|source|teacher|replay dataset|replay mask|replay budget|pool repet|objective|GPUs"; exit 0; }

# ---------------- 1. Code ----------------
if done_at "$CODE_OUT" 1800; then say "SKIP 1 code"; else
  say "1-CODE-START"
  env TASK=code TRAIN_ITERS=1800 CUDA_VISIBLE_DEVICES="$GPUS" NPROC_PER_NODE=$NP MASTER_PORT=$((PORT_BASE+1)) \
      OBJECTIVE=$OBJECTIVE MB=$MB SOURCE=$WIKI_KD_INIT SOURCE_REQUIRED_ITERS=600 \
      REPLAY_MINISET_DIR=$D/cka32_${GT_SET}_code_20260819/train REPLAY_GT_PATH=$D/cka32_${GT_SET}_code_20260819/gt \
      STUDY_ROOT=$STUDY LABEL=$(basename $CODE_OUT) \
      bash scripts/experiment/a100/run_g2_cka_gt_contextual_replay_mha.sh >> "$STUDY/logs/01_code.log" 2>&1
  say "1-CODE-DONE rc=$?"; done_at "$CODE_OUT" 1800 || { say "1-CODE-INCOMPLETE stop"; exit 1; }
fi
# ---------------- 2. Expand + KD-init ----------------
if done_at "$KD24" 600; then say "SKIP 2 expand"; else
  say "2-EXPAND-START"
  env CUDA_VISIBLE_DEVICES="$GPUS" NPROC_PER_NODE=$NP MASTER_PORT=$((PORT_BASE+2)) PYTHONNOUSERSITE=1 \
      FLAME_ENV=$FLAME_ENV PYTHON_BIN=$FLAME_ENV/bin/python PATH=$FLAME_ENV/bin:/usr/bin:/bin CUDA_HOME=$FLAME_ENV \
      FLAME_DATA_ROOT=$V WANDB_MODE=offline \
      TOKENIZER_MODEL=/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6 \
      SOURCE_WEIGHTS_DIR=$CODE_OUT SOURCE_REQUIRED_ITERS=1800 SOURCE_NUM_EXPERTS=16 NUM_EXPERTS=24 OLD_MODEL_KL_NUM_EXPERTS=16 \
      OLD_MODEL_KL_COEFF=1 OLD_MODEL_KL_TEMPERATURE=1 TRAIN_ITERS=600 MICRO_BATCH_SIZE=32 GLOBAL_BATCH_SIZE=2304 \
      SAVE_INTERVAL=600 EVAL_INTERVAL=600 PROBE_EVAL_INTERVAL=100 SECONDARY_PROBE_EVAL_INTERVAL=100 TERTIARY_PROBE_EVAL_INTERVAL=100 \
      RUN_ID=cka32-${TAG}-expansion-kd-init-e16to24 TRAIN_WEIGHTS=$KD24 LOCAL_BASE=$STUDY/local LOCAL_SSD_ROOT=$STUDY/scratch/kd_init \
      bash scripts/experiment/a100/run_g2_ffn_only_conversation_expert_distill_init_wikicode_mha.sh >> "$STUDY/logs/02_expand.log" 2>&1
  say "2-EXPAND-DONE rc=$?"; done_at "$KD24" 600 || { say "2-EXPAND-INCOMPLETE stop"; exit 1; }
fi
# ---------------- 3. Conv ----------------
if done_at "$CONV_OUT" 1800; then say "SKIP 3 conv"; else
  say "3-CONV-START"
  env TASK=conversation TRAIN_ITERS=1800 CUDA_VISIBLE_DEVICES="$GPUS" NPROC_PER_NODE=$NP MASTER_PORT=$((PORT_BASE+3)) \
      OBJECTIVE=$OBJECTIVE MB=$MB SOURCE=$KD24 SOURCE_REQUIRED_ITERS=600 \
      REPLAY_MINISET_DIR=$D/cka32_${GT_SET}_conv_20260819/train REPLAY_GT_PATH=$D/cka32_${GT_SET}_conv_20260819/gt \
      STUDY_ROOT=$STUDY LABEL=$(basename $CONV_OUT) \
      bash scripts/experiment/a100/run_g2_cka_gt_contextual_replay_mha.sh >> "$STUDY/logs/03_conv.log" 2>&1
  say "3-CONV-DONE rc=$?"
fi
say "CHAIN-COMPLETE"
