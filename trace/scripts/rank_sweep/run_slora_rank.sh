#!/usr/bin/env bash
# S-LoRA-Pre LoRA-rank (DoF) sweep on TRACE, fixed merge scaling (builder.py 2026-09-14),
# alpha = 2r (same ratio as the r64 Table-1 run and as the Ours sweep).
# Recipe = tab1_slora_r64_scalefix: lr 2e-4, global batch 64, seed 2025, epochs 5,3,7,5,3,5,5,7.
#   usage: RANK=16 GPUS=0,1,2,3 bash run_slora_rank.sh          # train (orders 1-8) + sparse-15 eval (TRACE only, no general benchmarks)
#          RANK=256 GPUS=4,5,6,7 MICRO=8 bash run_slora_rank.sh # micro 8 x 4 x accum 2 = 64 (memory)
#   GPU count must divide 64 with micro in {8,16}: 2 -> 16x2x2, 4 -> 16x4x1 (or 8x4x2), 8 -> 8x8x1.
set -uo pipefail
RANK=${RANK:?}; ALPHA=${ALPHA:-$((RANK*2))}; GPUS=${GPUS:?}
ROOT=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace
SWEEP=${SWEEP_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/rank_sweep_20260916}
NAME=slora_pre_r${RANK}; RUN=$SWEEP/$NAME; PRE=$RUN/llama31/pre
OWN=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/tab3_20260913/.gpu_owner
W=/data2/seonghyeonnoh/LLM-continual-learning-runtime/lmeval
NGPU=$(awk -F, '{print NF}' <<< "$GPUS"); MICRO=${MICRO:-$(( NGPU >= 8 ? 8 : 16 ))}
ACCUM=$(( 64 / (MICRO * NGPU) )); (( MICRO * NGPU * ACCUM == 64 )) || { echo "[ERROR] $NGPU GPUs x micro $MICRO cannot make 64"; exit 2; }
mkdir -p $RUN $SWEEP/logs $OWN
export SLORA_LLAMA31_PATH=/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct
say(){ printf '[SLORA-RANK %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$SWEEP/progress.log"; }
grep -q "original_scaling = lora_config.lora_alpha / lora_config.r" $ROOT/implementations/SLoRA-upstream-port/src/model/builder.py || { say "ABORT: merge fix missing"; exit 1; }
for g in ${GPUS//,/ }; do echo $$ > $OWN/$g; done
release(){ for g in ${GPUS//,/ }; do [ "$(cat $OWN/$g 2>/dev/null)" = "$$" ] && rm -f $OWN/$g; done; }; trap release EXIT
read -r -a G <<< "${GPUS//,/ }"

if [ ! -f $PRE/order8/max.safetensors ]; then
  say "$NAME train start: r$RANK a$ALPHA, $NGPU GPUs ($GPUS) x micro $MICRO x accum $ACCUM = 64"
  ( cd $ROOT && env SKIP_COMPLETED=1 PET_MASTER_PORT=${PORT:-29620} CUDA_VISIBLE_DEVICES=$GPUS WORLD_SIZE=$NGPU MICRO_BATCH=$MICRO GRAD_ACCUM=$ACCUM \
      SLORA_LORA_R=$RANK SLORA_LORA_ALPHA=$ALPHA SLORA_RELEASED_OUTPUT_ROOT=$RUN PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      bash scripts/run_experiment.sh train slora_pre_released llama31 ) >> $SWEEP/logs/$NAME.train.log 2>&1
  [ -f $PRE/order8/max.safetensors ] || { say "$NAME TRAIN FAILED (logs/$NAME.train.log)"; exit 1; }
  say "$NAME train done"
fi
say "$NAME eval start ($NGPU shards)"
pids=(); i=0
for g in "${G[@]}"; do
  env SLORA_OUTPUT_ROOT=$RUN CUDA_VISIBLE_DEVICES=$g EVAL_SHARD_COUNT=$NGPU EVAL_SHARD_INDEX=$i EVAL_SPARSE_15=1 \
    bash $ROOT/implementations/SLoRA-upstream-port/scripts/repro/eval_trace_pertask.sh pre llama31 > $SWEEP/logs/$NAME.eval$i.log 2>&1 &
  pids+=($!); i=$((i+1))
done
erc=0; for p in "${pids[@]}"; do wait $p || erc=1; done
say "$NAME eval exit=$erc"
# general benchmarks (MMLU/GSM8K/PIQA) intentionally NOT run for the rank sweep -- TRACE only
