#!/usr/bin/env bash
# One controlled MoE-LPR gamma cell: 2 GPUs, MB48, non-grouped MoE.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAMMA="${GAMMA:?set GAMMA}"
NAME="${NAME:?set NAME}"
GPUS="${GPUS:?set a two-GPU pair, e.g. 0,1}"
ROOT="${SWEEP_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/moelpr_gamma_2gpu_nongrouped_20260911}"

[ "$(awk -F, '{print NF}' <<< "$GPUS")" -eq 2 ] || { echo "GPUS must contain exactly two GPUs" >&2; exit 2; }

export MODEL_CONFIG_SCRIPT=scripts/experiment/a100/flame-moe-bf16-no-shared.sh
export GLOBAL_BATCH_SIZE=2304 TRAIN_ITERS=1800 SAVE_INTERVAL=1800
export SEED=1234 PAUSE_SECONDS=0 GUARD_GRACE_SECONDS=100000
export LPR_ROOT="$ROOT/$NAME" LPR_COEFF="$GAMMA"
export CODE_MB=48 CODE_LPR_MB=48 CONV_MB=48 CONV_LPR_MB=48

echo "[MOELPR] name=$NAME gamma=$GAMMA gpus=$GPUS mb=48 grouped_gemm=off start=$(date '+%F %T')"
bash "$HERE/run_moe_lpr_from_flamemoe_e8.sh"
echo "[MOELPR] name=$NAME complete=$(date '+%F %T')"
