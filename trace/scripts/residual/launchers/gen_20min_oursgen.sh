#!/usr/bin/env bash
# 20Minuten self-generated replay for Ours-gen, same two-stage pipeline as
# run_selfgen_cl_frozen.sh (anchor prompt sampling -> greedy answer pass),
# produced by the checkpoint right after 20Minuten = model/7.
set -uo pipefail
TRACE=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace
PY=$TRACE/.venv-runtime/bin/python
SG=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/selfgen_v3_20260829
CKPT=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/selfgen_cl_frozen_20260901/model/7
R=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/residual_router_20260915
D=$R/data/oursgen_20Minuten; mkdir -p $D
G=${GPU:?}
export OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
cd $TRACE
pfx=$($PY -c "import json;print(json.load(open('$SG/anchors.json'))['7']['anchor'],end='')")
[ -f $D/stageA/stats.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/analysis/bos_sample_v3.py --checkpoint $CKPT --mode anchor \
  --prefix-text "$pfx" --num-seqs 640 --max-seqs 200000 --max-new-tokens 1024 --batch 128 --no-routing-probe \
  --seed 319 --out-dir $D/stageA --label oursgen_20Minuten > $R/logs/gen20min_stageA.log 2>&1 || exit 1
CUDA_VISIBLE_DEVICES=$G $PY scripts/analysis/answer_pass_v3_fix.py --checkpoint $CKPT --stage-a $D/stageA \
  --out $D/records.jsonl --prompt-cue $'\n\nSimplification:' --max-answer-tokens 512 --batch 64 \
  > $R/logs/gen20min_stageB.log 2>&1
echo "records: $(wc -l < $D/records.jsonl)"
