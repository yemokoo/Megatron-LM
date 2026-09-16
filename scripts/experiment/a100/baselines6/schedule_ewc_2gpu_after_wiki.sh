#!/usr/bin/env bash
# Preserve both 1-GPU Wiki+Fisher stages, then prioritize w5632 on GPUs 6,7.
# Start w2816 on the first LPR pair returned by the gamma sweep.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOF=/data2/seonghyeonnoh/LLM-continual-learning-runs/dof_sweep_20260908
LOG="$DOF/logs/ewc_2gpu_transition.log"
RUN="$HERE/run_ewc_dof_width.sh"

say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
ready() {
    local w=$1 base="$DOF/b6_ewc_dof_fixed/w$1/wiki"
    [ "$(tr -d '[:space:]' < "$base/latest_checkpointed_iteration.txt" 2>/dev/null || true)" = 1800 ] &&
        [ -f "$base/continual_state_ewc/manifest.json" ]
}
stop_old_width() {
    local w=$1
    # Stop only the old single-GPU torchrun for this exact width. Its parent
    # chain exits on the nonzero return and the pool worker remains harmless.
    pkill -TERM -f "torch.distributed.run.*--nproc_per_node 1.*pretrain_gpt_baselines6.py.*--ffn-hidden-size $w" 2>/dev/null || true
    sleep 5
    pkill -KILL -f "torch.distributed.run.*--nproc_per_node 1.*pretrain_gpt_baselines6.py.*--ffn-hidden-size $w" 2>/dev/null || true
}
launch_width() {
    local w=$1 mb=$2 pair=$3 port=$4
    say "launch w$w on $pair MB$mb"
    setsid nohup env W="$w" MB="$mb" GPUS="$pair" CUDA_VISIBLE_DEVICES="$pair" \
        NPROC_PER_NODE=2 MASTER_PORT="$port" SKIP_WIKI_STAGE=1 \
        bash "$RUN" >> "$DOF/logs/ewc_dof_w${w}_2gpu.log" 2>&1 < /dev/null &
    echo "RUNNING 2gpu=$pair mb=$mb $(date '+%F %T')" > "$DOF/state/ewc_dof_w$w"
}
pair_free() {
    local a=$1 b=$2 ma mb
    ma=$(nvidia-smi -i "$a" --query-gpu=memory.used --format=csv,noheader,nounits)
    mb=$(nvidia-smi -i "$b" --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$ma" -lt 2000 ] && [ "$mb" -lt 2000 ]
}

say "waiting for both Wiki checkpoints and Fishers"
until ready 2816 && ready 5632; do sleep 30; done
say "both Wiki+Fisher stages ready; stopping old single-GPU chains"
stop_old_width 2816
stop_old_width 5632
until pair_free 6 7; do sleep 10; done
launch_width 5632 32 6,7 43860

say "waiting for an LPR pair for w2816"
if pgrep -f "torch.distributed.run.*--nproc_per_node 2.*pretrain_gpt_baselines6.py.*--ffn-hidden-size 2816" >/dev/null; then
    say "w2816 already running on 2 GPUs; transition scheduler complete"
    exit 0
fi
while true; do
    for pair in 0,1 2,3 4,5; do
        a=${pair%,*}; b=${pair#*,}
        if pair_free "$a" "$b"; then
            launch_width 2816 64 "$pair" 43861
            say "transition scheduler complete"
            exit 0
        fi
    done
    sleep 30
done
