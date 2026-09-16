#!/usr/bin/env bash
# TRACE LoRA-rank (DoF) sweep for Ours-rep (v3_new_replay1to1, KD-init + 1-phase
# joint real replay, shared-router QKVO+FFN experts).
#
# Baseline = Table-1 Ours-rep run
#   /data2/seonghyeonnoh/LLM-continual-learning-runs/trace/v3_replay1to1/v3_new_replay1to1_st_top1
#   (r64, alpha 128, 4 GPUs x micro 8 x accum 1 = global 32, seed 2025).
# Every argument is copied from that run's train.command.txt; only these change:
#   --lora_moe_rank R   --lora_moe_alpha 2R   (alpha/r ratio fixed at 2, attention
#   rank follows the FFN rank automatically)   --output_dir / --data_output_path
#   --nproc_per_node / --per_device_train_batch_size  (global batch kept at 32)
#
#   usage: RANK=16 GPUS=0,1,2,3 bash run_rank_sweep.sh            # train + eval
#          RANK=16 GPUS=0,1,2,3 PHASE=train bash run_rank_sweep.sh
#          RANK=16 GPUS=0,1,2,3 PHASE=eval  bash run_rank_sweep.sh
#          RANK=16 CHECK=1 bash run_rank_sweep.sh                 # print argv diff vs baseline, no run
#          RANK=256 GPUS=0,1 SMOKE=1 bash run_rank_sweep.sh       # C-STANCE only, for OOM check
#   env:   SWEEP_ROOT (default runs/trace/rank_sweep_20260916), PORT, GLOBAL_BATCH (32)
set -uo pipefail
RANK=${RANK:?RANK required (e.g. 16 32 128 256)}
ALPHA=${ALPHA:-$((RANK * 2))}
PHASE=${PHASE:-both}; CHECK=${CHECK:-0}; SMOKE=${SMOKE:-0}
GLOBAL_BATCH=${GLOBAL_BATCH:-32}
TRACE=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace
LLMCL=$TRACE/implementations/llmcl_benchmark
PY=$TRACE/.venv-runtime/bin/python
BASELINE=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/v3_replay1to1/v3_new_replay1to1_st_top1
# argv source: the baseline run's train.command.txt if that run exists on this host,
# otherwise the verbatim copy committed next to this script (same file, 2026-09-16).
BASELINE_CMD=${BASELINE_CMD:-$BASELINE/train.command.txt}
[ -f "$BASELINE_CMD" ] || BASELINE_CMD=$(dirname "${BASH_SOURCE[0]}")/baseline_train.command.txt
[ -f "$BASELINE_CMD" ] || { echo "[ERROR] no baseline train.command.txt"; exit 2; }
SWEEP_ROOT=${SWEEP_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/rank_sweep_20260916}
NAME=ours_rep_r${RANK}; [ "$SMOKE" = 1 ] && NAME=${NAME}_smoke
OUT=$SWEEP_ROOT/$NAME
OWN=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/tab3_20260913/.gpu_owner
export SLORA_LLAMA31_PATH=/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct
export PYTHONNOUSERSITE=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
say(){ printf '[RANK-SWEEP %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$SWEEP_ROOT/progress.log"; }

# ---- build argv from the baseline command, overriding only what the sweep changes
build_argv(){
  local ngpu=$1 micro=$2
  $PY - "$BASELINE_CMD" "$RANK" "$ALPHA" "$OUT" "$ngpu" "$micro" "${PORT:-29881}" "$SMOKE" <<'PYEOF'
import sys
cmd, rank, alpha, out, ngpu, micro, port, smoke = sys.argv[1:]
toks = open(cmd).read().split()
i = toks.index("training/main_Ours_LoRA_MoE.py"); args = toks[i + 1:]
def setv(k, v):
    j = args.index(k); args[j + 1] = v
setv("--lora_moe_rank", rank); setv("--lora_moe_alpha", alpha)
setv("--output_dir", out); setv("--data_output_path", out + "/data_cache")
setv("--per_device_train_batch_size", micro); setv("--gradient_accumulation_steps", "1")
args = [a.replace("\\,", ",") for a in args]
if smoke == "1":
    args += ["--stop_after_task", "C-STANCE"]
print(" ".join([f"--nproc_per_node={ngpu}", f"--master_port={port}", "training/main_Ours_LoRA_MoE.py"] + args))
PYEOF
}

if [ "$CHECK" = 1 ]; then
  echo "== argv diff vs Table-1 baseline (4 GPUs assumed) =="
  diff <(tr ' ' '\n' < $BASELINE_CMD | sed -n '/main_Ours_LoRA_MoE.py/,$p') \
       <(build_argv 4 8 | tr ' ' '\n' | sed -n '/main_Ours_LoRA_MoE.py/,$p') || true
  exit 0
fi

GPUS=${GPUS:?GPUS required, e.g. 0,1,2,3}
NGPU=$(awk -F, '{print NF}' <<< "$GPUS")
MICRO=$((GLOBAL_BATCH / NGPU))
(( MICRO * NGPU == GLOBAL_BATCH )) || { echo "[ERROR] $NGPU GPUs cannot make global batch $GLOBAL_BATCH"; exit 2; }
mkdir -p $OUT $SWEEP_ROOT $OWN
for g in ${GPUS//,/ }; do echo $$ > $OWN/$g; done
release(){ for g in ${GPUS//,/ }; do [ "$(cat $OWN/$g 2>/dev/null)" = "$$" ] && rm -f $OWN/$g; done; }
trap release EXIT

if [ "$PHASE" != eval ] && [ ! -f $OUT/7/lora_moe_meta.json ]; then
  ARGV=$(build_argv $NGPU $MICRO)
  echo "$PY -m torch.distributed.run $ARGV" > $OUT/train.command.txt
  say "$NAME train start: rank $RANK alpha $ALPHA, $NGPU GPUs ($GPUS) x micro $MICRO = global $GLOBAL_BATCH"
  ( cd $LLMCL && CUDA_VISIBLE_DEVICES=$GPUS $PY -m torch.distributed.run $ARGV ) > $OUT/train.log 2>&1
  rc=$?
  last=$( [ "$SMOKE" = 1 ] && echo 0 || echo 7 )
  if [ $rc -ne 0 ] || [ ! -f $OUT/$last/lora_moe_meta.json ]; then say "$NAME TRAIN FAILED rc=$rc ($OUT/train.log)"; exit 1; fi
  say "$NAME train done"
fi
[ "$SMOKE" = 1 ] && { say "$NAME smoke ok (round 0 saved)"; exit 0; }
[ "$PHASE" = train ] && exit 0

# ---- sparse-15 eval, same driver/knobs as run_v3_job.sh (what scored Table 1)
LOG_DIR=$SWEEP_ROOT/logs/${NAME}_sparse15; mkdir -p $LOG_DIR
say "$NAME eval start on $GPUS"
OURS_LORAMOE_OUTPUT_ROOT=$OUT SPARSE15_METHODS=ours_lora_moe_v3_new_replay1to1 SPARSE15_GPUS=$GPUS \
SPARSE15_NUM_SAMPLE_SHARDS=${SPARSE15_NUM_SAMPLE_SHARDS:-4} SPARSE15_EXACT_STOP_MARKERS=1 \
SPARSE15_EVAL_BATCH=96 SPARSE15_SCIENCEQA_BATCH=256 SPARSE15_20MINUTEN_BATCH=64 \
SPARSE15_MEETINGBANK_BATCH=4 SPARSE15_PY150_BATCH=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
SPARSE15_LOG_ROOT=$LOG_DIR/queue_logs SPARSE15_STATUS_INTERVAL=120 SPARSE15_CONTINUE_ON_CELL_ERROR=1 \
TRACE_PYTHON=$PY $PY -u $TRACE/scripts/run_ours_sparse15_optimized.py > $LOG_DIR/eval.log 2>&1
say "$NAME eval exit=$? $( $PY -c "import json;d=json.load(open('$OUT/sparse15_summary.json'));print('AA %.2f F %.2f'%(d['final_average'],-d['BWT']))" 2>/dev/null )"
