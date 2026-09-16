#!/usr/bin/env bash
# Table-1 baseline (ewc | seq_lora | olora ...) LoRA-rank sweep on TRACE, alpha = 2r.
# Same recipe as tab1_fixed_20260913 (lr 2e-4, global 64, seed 2025, all7 targets,
# EWC lambda 400 online / fisher 1000): only TAB1_RANK/TAB1_ALPHA/output change.
#   usage: METHOD=ewc RANK=16 GPUS=0,1,2,3 bash run_tab1_rank.sh   # train -> sparse-15 -> lm-eval
set -uo pipefail
METHOD=${METHOD:?ewc|seq_lora|olora}; RANK=${RANK:?}; ALPHA=${ALPHA:-$((RANK*2))}; GPUS=${GPUS:?}
ROOT=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace
PY=$ROOT/.venv-runtime/bin/python
SWEEP=${SWEEP_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/rank_sweep_20260916}
NAME=${METHOD}_r${RANK}; RUN=$SWEEP/$NAME
OWN=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/tab3_20260913/.gpu_owner
W=/data2/seonghyeonnoh/LLM-continual-learning-runtime/lmeval
NGPU=$(awk -F, '{print NF}' <<< "$GPUS"); MICRO=${MICRO:-$(( 64 / NGPU > 16 ? 16 : 64 / NGPU ))}
mkdir -p $SWEEP/logs $OWN
export SLORA_LLAMA31_PATH=/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct
say(){ printf '[TAB1-RANK %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$SWEEP/progress.log"; }
for g in ${GPUS//,/ }; do echo $$ > $OWN/$g; done
release(){ for g in ${GPUS//,/ }; do [ "$(cat $OWN/$g 2>/dev/null)" = "$$" ] && rm -f $OWN/$g; done; }; trap release EXIT
read -r -a G <<< "${GPUS//,/ }"

if [ ! -f $RUN/7/tab1_meta.json ]; then
  say "$NAME train start: r$RANK a$ALPHA, $NGPU GPUs ($GPUS) micro $MICRO global 64"
  ( cd $ROOT && env TAB1_RANK=$RANK TAB1_ALPHA=$ALPHA TAB1_GPUS=$GPUS TAB1_PORT=${PORT:-29630} TAB1_MICRO_BATCH=$MICRO TAB1_GLOBAL_BATCH=64 \
      TAB1_OUTPUT_ROOT=$RUN bash scripts/baselines/llama31/tab1_${METHOD}.sh train ) > $SWEEP/logs/$NAME.train.log 2>&1
  [ -f $RUN/7/tab1_meta.json ] || { say "$NAME TRAIN FAILED (logs/$NAME.train.log)"; exit 1; }
  say "$NAME train done"
fi
say "$NAME eval start"
( cd $ROOT && $PY scripts/run_tab1_sparse15.py --evaluator evaluate_tab1.py --method $NAME --run-dir $RUN --gpus $GPUS --py150-batch 8 --meetingbank-batch 1 ) > $SWEEP/logs/$NAME.eval.log 2>&1
say "$NAME eval exit=$? $( $PY -c "import json;d=json.load(open('$RUN/sparse15_summary.json'));print('AA %.2f F %.2f'%(d['final_average'],-d['BWT']))" 2>/dev/null )"
export HF_HOME=/data2/seonghyeonnoh/huggingface HF_DATASETS_CACHE=/data2/seonghyeonnoh/LLM-continual-learning-data/lmeval_datasets HF_DATASETS_OFFLINE=1 HF_DATASETS_TRUST_REMOTE_CODE=1 TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=${G[0]} $W/../lmeval-venv/bin/python $W/run_lmeval_trace.py --ckpt $RUN/7 --out $W/tab1_mmlu/$NAME --tasks mmlu --batch_size 16 > $W/tab1_mmlu/$NAME.log 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=${G[1]} $W/../lmeval-venv/bin/python $W/run_lmeval_trace.py --ckpt $RUN/7 --out $W/tab1_gsm8k_piqa/$NAME --tasks gsm8k,piqa --batch_size 16 > $W/tab1_gsm8k_piqa/$NAME.log 2>&1 &
wait $p1 $!
say "$NAME lm-eval done: $(grep -m1 '^|mmlu ' $W/tab1_mmlu/$NAME.log | tr -s ' ') $(grep -E '^\|(gsm8k|piqa) ' $W/tab1_gsm8k_piqa/$NAME.log | tr -s ' ' | tr '\n' ' ')"
