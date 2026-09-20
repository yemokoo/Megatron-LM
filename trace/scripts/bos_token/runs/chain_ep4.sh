#!/usr/bin/env bash
set -uo pipefail
T=/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning/trace; PY=$T/.venv-runtime/bin/python
D=/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace
CK0=/data2/seonghyeonnoh/paper/ablation/1phase_kd_rep/0; CK1=/data2/seonghyeonnoh/paper/bos_token/fomc_arms/bos_gen/model/1
R=/data2/seonghyeonnoh/paper/bos_token/cstance_1phase_kd_rep; A=/data2/seonghyeonnoh/paper/bos_token/after_fomc
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say(){ printf '[EP4 %s] %s\n' "$(date '+%F %T')" "$*" | tee -a $R/chain_ep4.log; }
say "train ep4 lr1e-3 on 8 GPUs"
cd $T && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PY -m torch.distributed.run --nproc_per_node=8 --master_port=29902 scripts/bos_token/train_bos_token.py \
  --checkpoint $CK0 --train-json $D/C-STANCE/train.json --eval-json $D/C-STANCE/eval.json --eval-num 200 \
  --lr 1e-3 --epochs 4 --micro-batch 4 --grad-accum 8 --eval-every 40 --out-dir $R/ep4_lr1e-3 > $R/ep4_lr1e-3.log 2>&1
[ -f $R/ep4_lr1e-3/bos_token.pt ] || { say "TRAIN FAILED"; exit 1; }
say "train done; generating"
gen(){ CUDA_VISIBLE_DEVICES=$1 $PY scripts/bos_token/gen_doc.py --checkpoint $2 --bos-token-file $3 --num-seqs 50 --batch 50 --max-new-tokens 320 --out-dir $4 > $4.log 2>&1; }
gen 0 $CK1 $R/ep4_lr1e-3/bos_token.pt $A/gen_bos_cstance_ep4_on_r1 &
gen 1 $CK0 $R/ep4_lr1e-3/bos_token.pt $A/gen_bos_cstance_ep4_on_r0 &
gen 2 $CK1 $A/bos_cstance_base/bos_token.pt $A/gen_bos_cstance_base_on_r1 &
gen 3 $CK0 $A/bos_cstance_base/bos_token.pt $A/gen_bos_cstance_base_on_r0 &
wait; say "ALL DONE"
